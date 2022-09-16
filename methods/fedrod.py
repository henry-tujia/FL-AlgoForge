import random
import torch
import torch.nn.functional as F

import logging
from methods.base import Base_Client, Base_Server
import torch.nn.functional as F
from models.resnet_balance import resnet_fedbalance_experimental as  resnet_fedbalance
from models.resnet_balance import resnet_fedbalance_server_experimental as  resnet_fedbalance_server
from torch.multiprocessing import current_process
import numpy as np
import random
import copy


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        client_dict["model_paras"].update({"KD":True})
        self.model = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.predictor = copy.deepcopy(self.model.fc)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

        self.optimizer_prd = torch.optim.SGD(self.predictor.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)


        self.client_infos = client_dict["client_infos"]

        self.client_cnts = self.init_client_infos()

    def load_client_state_dict(self,server_state_dict):
        paras_old = self.model.state_dict()
        paras_new = server_state_dict

        # print(paras_new.keys())

        for key in self.upload_keys:
  
            paras_old[key] = paras_new[key]
            # print(key)
        
        self.model.load_state_dict(paras_old)

    def init_client_infos(self):
        client_cnts = {}
        # print(self.client_infos)

        for client,info in self.client_infos.items():
            cnts = []
            for c in range(self.num_classes):
                if c in info.keys():
                    num = info[c]
                else:
                    num = 0
                cnts.append(num)

            cnts = torch.FloatTensor(np.array(cnts))
            client_cnts.update({client:cnts})
        # print(client_cnts)
        return client_cnts

    def get_cdist(self,idx):
        client_dis = self.client_cnts[idx]

        dist = client_dis / client_dis.sum() #个数的比例
        cdist = dist# 
        # cdist = cdist * (1.0 - self.args.alpha) + self.args.alpha
        cdist = cdist.reshape((1, -1))

        # if torch.any(torch.isnan(cdist)):
        logging.info("Client is {}\t, distance is {}\t".format(idx,cdist))
    
        return cdist.to(self.device)

    def train(self): 

        cdist = self.get_cdist(self.client_index)
        # train the local model
        self.model.to(self.device)
        self.model.train()
        # for name, param in self.model.named_parameters():
        #     if "local" in name :
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                feature, log_probs = self.model(images)
                loss_bsm = self.balanced_softmax_loss(labels, log_probs, cdist)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                log_probs_pred = self.predictor(feature.detach())
                loss = self.criterion(log_probs_pred+log_probs.detach(),labels)
                self.optimizer_prd.zero_grad()
                loss.backward()
                self.optimizer_prd.step()

                batch_loss.append(loss.item())
            # #此处交换参数以及输出新字典
            # self.model.change_paras()
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        
        weights = {key:value for key,value in self.model.cpu().state_dict().items()}
        return epoch_loss, weights
    # https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
    def balanced_softmax_loss(self,labels, logits, sample_per_class, reduction="mean"):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        sample_per_class: A int tensor of size [no of classes].
        reduction: string. One of "none", "mean", "sum"
        Returns:
        loss: A float tensor. Balanced Softmax Loss.
        """
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
        return loss

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model_server = self.model_type(**server_dict["model_paras"])
        self.model = resnet_fedbalance_server(self.model_server)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

