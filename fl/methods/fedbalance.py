import os
import torch

import logging
from methods.base import Base_Client, Base_Server
from models.resnet_balance import resnet_fedbalance_experimental as resnet_fedbalance
from models.resnet_balance import resnet_fedbalance_server_experimental as resnet_fedbalance_server
from torch.multiprocessing import current_process
import numpy as np
from models.resnet import resnet8 as resnet8
from models.resnet import resnet20 as resnet20
from models.alexnet import alexnet
from models.lenet import lenet


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model_global = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.hypers = client_dict["hypers"]
        self.model_local = self.init_local_net().to(self.device)#
        self.model = resnet_fedbalance(self.model_local, self.model_global)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)
        self.upload_keys = [
            x for x in self.model.state_dict().keys() if not 'local' in x]

    def init_local_net(self):
        if self.hypers["model_type"]== "resnet20":
            return resnet20(self.num_classes)
        elif self.hypers["model_type"]== "resnet8":
            return resnet8(self.num_classes)
        elif self.hypers["model_type"]== "alexnet":
            return alexnet(self.num_classes)
        elif self.hypers["model_type"]== "lenet":
            return lenet(self.num_classes)
        else:
            raise Exception("Invalid Local Model! ",self.hypers["model_type"])

    def load_client_state_dict(self, server_state_dict):
        paras_old = self.model.state_dict()
        paras_new = server_state_dict
        for key in self.upload_keys:
            paras_old[key] = paras_new[key]
        self.model.load_state_dict(paras_old)

    def get_cdist_inner(self, idx):
        client_dis = self.client_cnts[idx]

        dist = client_dis / client_dis.sum()  # 个数的比例
        cdist = dist#/dist.max()
        cdist = cdist.reshape((1, -1))
        # cdist = torch.log(cdist)

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
                
                probs = self.model(images, cidst)
                loss = self.criterion(probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(
                    self.client_index, epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
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
                
                logits = self.model.model_global(x)
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
        self.model_global = self.model_type(**server_dict["model_paras"])
        self.model = resnet_fedbalance_server(self.model_global)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
