import torch
import logging
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
import numpy as np
from torch.cuda.amp import autocast as autocast

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        client_dict["model_paras"].update({"KD":True})
        self.model = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.hypers = client_dict["hypers"]

        # self.client_infos = client_dict["client_infos"]
        # self.client_cnts = self.init_client_infos()

    def init_client_infos(self):
        client_cnts = {}
        # print(self.client_infos)

        for client,info in self.client_infos.items():
            cnts = []
            for c in range(self.num_classes):
                if c in info.keys():
                    num = 1
                else:
                    num = 0
                cnts.append(num)

            cnts = torch.FloatTensor(np.array(cnts))
            client_cnts.update({client:cnts})    
        # print(client_cnts)
        return client_cnts

    def get_cdist_inner(self,idx):
        client_dis = self.client_cnts[idx]

        cdist = client_dis * (1.0 - self.hypers["mu"]) + self.hypers["mu"]
        cdist = cdist.reshape((1, -1))

        return cdist.to(self.device)
 
    def train(self): 

        cidst = self.get_cdist_inner(self.client_index)
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
                hs,_ = self.model(images)
                ws = self.model.fc.weight

                logits = cidst * hs.mm(ws.transpose(0, 1))
                loss = self.criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        
        #此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = {key:value for key,value in self.model.cpu().state_dict().items()}
        return weights

    def test(self):

        cidst = self.get_cdist_inner(self.client_index)
        self.model.to(self.device)
        self.model.eval() 

        preds = None
        labels = None
        acc = None
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                with autocast():
                    hs,_ = self.model(x)
                    ws = self.model.fc.weight

                    logits = cidst * hs.mm(ws.transpose(0, 1))
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(logits, 1)
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
        weighted_acc = acc.reshape((1,-1)).mean()
        logging.info(
                "************* Client {} Acc = {:.2f} **************".format(self.client_index, weighted_acc.item()))
        return weighted_acc

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(**server_dict["model_paras"])
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

