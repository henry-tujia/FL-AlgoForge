"""
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
"""

import torch
import torch.nn as nn

# import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process


class FedDecorrLoss(nn.Module):
    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        # z标准化
        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=self.args.wd,
            nesterov=True,
        )
        self.hypers = client_dict["hypers"]
        self.extra_loss = FedDecorrLoss()

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        for epoch in range(self.args.epochs):
            epoch_DC = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                feature, log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)

                extra_loss = self.extra_loss(feature)
                epoch_DC.append(extra_loss.item())

                # loss = loss + extra_loss*self.hypers["mu"]

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.loggers[self.client_idx].info(
                    "(Local Training Epoch: {} \tLoss: {:.6f} DC: {:.6f} Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        sum(epoch_DC) / len(epoch_DC),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
        weights = self.model.cpu().state_dict()
        return weights


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
