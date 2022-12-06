import os
import torch

import logging
from methods.base import Base_Client, Base_Server
from models.resnet_balance import resnet_fedbalance_experimental as resnet_fedbalance
from models.resnet_balance import resnet_fedbalance_server_experimental as resnet_fedbalance_server
from torch.multiprocessing import current_process
import numpy as np
from torch.cuda.amp import autocast as autocast

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model_new = self.model_type(
            **client_dict["model_paras"]["new"]).to(self.device)
        model_local_type, paras = client_dict["model_paras"]["local"].values()
        self.model_local = model_local_type(**paras).to(self.device)
        self.model = resnet_fedbalance(self.model_local, self.model_new,KD=True)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

        self.private_epochs = 2

        self.upload_keys = [
            x for x in self.model.state_dict().keys() if not 'local' in x]

    def load_client_state_dict(self, server_state_dict):
        paras_old = self.model.state_dict()
        paras_new = server_state_dict

        for key in self.upload_keys:

            paras_old[key] = paras_new[key]
            # print(key)

        self.model.load_state_dict(paras_old)

    def get_cdist_inner(self, idx):
        client_dis = self.client_cnts[idx]

        dist = client_dis / client_dis.sum()  # 个数的比例
        cdist = dist#/dist.max()
        cdist = cdist.reshape((1, -1))

        # logging.info("Client is {}\t, distance is {}\t".format(idx, cdist))

        return cdist.to(self.device)

    def train(self):

        cidst = self.get_cdist_inner(self.client_index)
        # train the local model
        self.model.to(self.device)
        self.model.train()

        epoch_loss = []
        epoch_KL = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            batch_KL = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                h_local,h_gloabl,probs = self.model(images, cidst)
                probs_l = torch.softmax(h_local,-1)
                probs_g = torch.softmax(h_gloabl,-1)
                # # probs_g = torch.softmax(h_gloabl,-1)
                
                # if epoch < self.private_epochs:
                #     loss = self.criterion(probs_l, labels)
                # else:
                h_combine = probs_l*cidst + probs_g
                loss = self.criterion(h_combine, labels)
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                kl = torch.kl_div(probs_l.log(),probs_g.detach()).mean()
                batch_KL.append(kl)

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_KL.append(sum(batch_KL) / len(batch_KL))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}\tKL: {:.6f}  Thread {}  Map {}'.format(
                    self.client_index, epoch, sum(epoch_loss) / len(epoch_loss),sum(epoch_KL) / len(epoch_KL), current_process()._identity[0], self.client_map[self.round]))
        weights = {key: value for key, value in self.model.cpu(
        ).state_dict().items() if key in self.upload_keys}
        return weights

    def test(self):

        cidst = self.get_cdist_inner(self.client_index)
        self.model.to(self.device)
        self.model.eval()

        preds = None
        labels = None
        acc = None

        # test_correct = 0.0
        # # test_loss = 0.0
        # test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                _,_,probs = self.model(x, cidst)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(probs, 1)
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
        self.model_server = self.model_type(**server_dict["model_paras"])
        self.model = resnet_fedbalance_server(self.model_server)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
