import torch
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
import torch.nn as nn
import numpy as np



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        # client_dict["model_paras"].update({"KD":True})
        self.model = self.model_type(
            **client_dict["model_paras"]).to(self.device)
        # self.criterion = My_loss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.nll = nn.NLLLoss().to(self.device)
        # self.client_infos = client_dict["client_infos"]
        # self.client_cnts = self.init_client_infos()

    # def init_client_infos(self):
    #     client_cnts = {}
    #     # print(self.client_infos)

    #     for client, info in self.client_infos.items():
    #         cnts = []
    #         for c in range(self.num_classes):
    #             if c in info.keys():
    #                 num = info[c]
    #             else:
    #                 num = 0
    #             cnts.append(num)

    #         cnts = torch.FloatTensor(np.array(cnts))
    #         client_cnts.update({client: cnts})
    #     # print(client_cnts)
    #     return client_cnts

    def one_hot(y, num_class):
        return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)

    class PairedLoss(torch.nn.Module):
        def __init__(self, T, client_dist):
            super(PairedLoss, self).__init__()
            self.T = T
            self.client_dist = client_dist
            self.adds = self.T * torch.pow(self.client_dist, -0.25)

        def forward(self, inputs, targets):
            targets_onehot = one_hot(targets, inputs.shape[1])
            inputs = inputs - self.adds
            loss = - torch.log(torch.exp(inputs[targets]) / torch.sum(torch.exp(inputs[targets_onehot == 0]), dim=1))
            return torch.mean(loss)

    def get_cdist(self, idx):
        client_dis = self.client_cnts[idx]

        client_dis = client_dis**(-0.25)*self.args.mu
        client_dis = torch.where(torch.isinf(client_dis), torch.full_like(client_dis, 0), client_dis)
        cdist = client_dis.reshape((1, -1))


        return cdist.to(self.device)

    def train(self):

        cidst = self.get_cdist(self.client_index)
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                logits = self.model(images)

                calibrated_logits = logits-cidst



                loss = self.criterion(calibrated_logits, labels)



                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(
                    self.client_index, epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        # 此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = {key: value for key,
                   value in self.model.cpu().state_dict().items()}
        return weights


    def criterion(self,x,y):

        # print(x.shape,y.shape)

        targ1hot = torch.zeros(x.shape,device=self.device).scatter(1, y.reshape((-1,1)), 1.0)

        up = x.exp()
        # print(up)

        select_item = (up*targ1hot).sum(dim=1,keepdim=True)
        down = up.sum(dim=1,keepdim=True)-select_item

        # probs = select_item/down

        log_probs = -(torch.log(select_item)-torch.log(down))

        # log_probs_selletc = (log_probs*targ1hot).sum(dim=1,keepdim=True)

        loss = log_probs.mean()


        if torch.isinf(loss).any() or torch.isnan(loss).any():
            print("up_old",up)
            print("select_item",select_item)
            print("up",up.sum(dim=1,keepdim=True))
            print("down",down)
            print("log_probs",log_probs)

                    

        # input("wait")

        # loss =self.nll(log_probs,y)

        return loss
 
    def test(self):

        if len(self.acc_dataloader) == 0:
            return 0

        cidst = self.get_cdist(self.client_index)
        self.model.to(self.device)
        self.model.eval()

        preds = None
        labels = None
        acc = None
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                logits = self.model(x)

                calibrated_logits = logits-cidst
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(calibrated_logits, 1)
                if preds is None:
                    preds = predicted.cpu()
                    labels = target.cpu()
                else:
                    preds = torch.concat((preds,predicted.cpu()),dim=0)
                    labels = torch.concat((labels,target.cpu()),dim=0)
        for c in range(self.num_classes):
            temp_acc = (((preds == labels) * (labels == c)).float() / (max((labels == c).sum(), 1))).sum().cpu()
            if acc is None:
                acc = temp_acc.reshape((1,-1))
            else:
                acc = torch.concat((acc,temp_acc.reshape((1,-1))),dim=0) 
        weighted_acc = (acc.reshape((1,-1))*self.weight_test.cpu()).sum()
        logging.info(
                "************* Client {} Acc = {:.2f} **************".format(self.client_index, weighted_acc.item()))
        return weighted_acc


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(**server_dict["model_paras"])
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
