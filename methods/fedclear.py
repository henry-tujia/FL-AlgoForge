import copy
import random
import torch
import torch.nn.functional as F

import logging
from methods.base import Base_Client, Base_Server
import torch.nn.functional as F
from  torch import nn
from torch.multiprocessing import current_process
import numpy as np
import random



class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        client_dict["model_paras"]["KD"] = True
        self.model = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.client_infos = client_dict["client_infos"]
        self.client_cnts = self.init_client_infos()
        self.cut_size = client_dict["cut_size"]
        self.predictor = nn.Sequential(
            nn.Linear(self.cut_size,128).to(self.device),
            nn.Linear(128,self.num_classes).to(self.device)
        )
         
        self.optimizer = torch.optim.SGD([{"params":self.model.parameters()},{"params":self.predictor.parameters()}]
            , lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

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
        cdist = dist/ dist.max()

        cdist = cdist.reshape((1, -1))
        return cdist.to(self.device)

    def train(self): 

        cidst = self.get_cdist(self.client_index)
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.predictor.train()

        epoch_loss = []
        for epoch in range(self.args.epochs): 
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                self.predictor.zero_grad()
                features,log_probs = self.model(images)
                # log_probs = nn.functional.softmax(log_probs, dim=1)
                local_logits = self.predictor(features[:,:self.cut_size].detach())
                
                loss_1 = self.criterion(local_logits, labels)
                logits = log_probs+local_logits*cidst
                loss = self.criterion(logits, labels)+0.1*loss_1
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        # if self.client_index==0:
        #     self.acc_dataloader = self.train_dataloader
        #     self.testAndplot(cidst)
        weights = {key:value for key,value in self.model.cpu().state_dict().items()}
        return epoch_loss, weights
    
    def testAndplot(self,cidst):

        self.model.to(self.device)
        self.model.eval()
        self.predictor.eval()
        inner_list = [[] for x in range(self.num_classes)]
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                features,log_probs = self.model(x)
                # log_probs = nn.functional.softmax(log_probs, dim=1)
                local_logits = nn.functional.relu(self.predictor(features[:,:self.cut_size]))*cidst
                logits = log_probs+local_logits

                for index,label in enumerate(target):
                    inner_list[label.data].append([local_logits[index].cpu().numpy(),log_probs[index].cpu().numpy(),logits[index].cpu().numpy(),cidst[0].cpu().numpy()])
        plot_data_list = []
        import pandas

        for i in range(self.num_classes):
            if len(inner_list[i]) == 0:
                inner_list[i] = [np.zeros(self.num_classes).tolist(),np.zeros(self.num_classes).tolist(),cidst[0].cpu().numpy().tolist()]
            else:
                inner_array=  np.array(inner_list[i]).mean(axis=0).tolist()
            plot_data_list.append([x for x in inner_array])
                
        logging.info("{}\n".format(plot_data_list))
        df_save = pandas.DataFrame(plot_data_list)
        df_save.to_csv(str(self.round)+"_pos_here.csv")
    def loss_pos(self,x):
        y = torch.where(x<0, x,torch.zeros_like(x).to(self.device))

        return F.mse_loss(y,torch.zeros_like(y).to(self.device))

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(**server_dict["model_paras"])
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)