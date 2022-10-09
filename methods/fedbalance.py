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
import wandb
import pandas


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model_new = self.model_type(**client_dict["model_paras"]["new"]).to(self.device)
        model_local_type,paras = client_dict["model_paras"]["local"].values()#
        self.model_local = model_local_type(**paras).to(self.device)
        self.model = resnet_fedbalance(self.model_local,self.model_new)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

        self.client_infos = client_dict["client_infos"]

        self.client_cnts = self.init_client_infos()

        self.upload_keys = [x for x in self.model.state_dict().keys() if not 'local' in x ]

        self.softmax = torch.nn.Softmax(dim=1)

        # print(self.upload_keys)

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
        cdist = dist/dist.max()#/ # 
        # cdist = cdist * (1.0 - self.args.alpha) + self.args.alpha
        cdist = cdist.reshape((1, -1))

        # if torch.any(torch.isnan(cdist)):
        logging.info("Client is {}\t, distance is {}\t".format(idx,cdist))
    
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
                probs = self.model(images,cidst)
                # if self.client_index==0 and batch_idx==0:
                #     with open('logits.log', 'a+') as out_file:
                #         out_file.write("Round {}\tEpoch {}\tLabel {}\n".format(self.round,epoch,labels[0]))
                #         for name,data in zip(("local","global","combine"),(local_probs[0],global_probs[0],log_probs[0])):
                #             out_file.write("{} logits is {}\n".format(name,data))
                loss = self.criterion(probs, labels)
   
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            # #此处交换参数以及输出新字典
            # self.model.change_paras()
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        # self.acc_dataloader = self.test_dataloader
        # self.test_inner(cidst)
        # #此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = {key:value for key,value in self.model.cpu().state_dict().items() if key in self.upload_keys}
        return epoch_loss, weights

    def test_inner(self,cdist):
        self.model.to(self.device)
        self.model.eval()

        test_correct = [0.0]*3
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                local_probs,global_probs,log_probs = self.model(x,cdist)

                for index,probs in enumerate([local_probs,global_probs,log_probs]):
                    _, predicted = torch.max(probs, 1)
                    correct = predicted.eq(target).sum()

                    test_correct[index] += correct.item()
                # test_loss += loss.item()
                test_sample_number += target.size(0)
            accs = (np.array(test_correct) / test_sample_number)*100
            with open('localval.log', 'a+') as out_file:
                out_file.write("Client {}\tRound {}\tLocal Model acc :{}\tGlobal Model acc :{}\tCombine Model acc :{}\n".format(self.client_index,self.round,*accs))


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model_server = self.model_type(**server_dict["model_paras"])
        self.model = resnet_fedbalance_server(self.model_server)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

