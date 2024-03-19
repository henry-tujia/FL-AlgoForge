"""
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
"""

import pandas
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
import copy
import numpy
from torch.multiprocessing import current_process
from torch.optim.optimizer import Optimizer


class sgddelta(Optimizer):
    def __init__(self, params, defaults: Dict[str, Any]) -> None:
        super().__init__(params, defaults)

    def compute_dif_norms(self, pre_optim):
        for group, pre_group in zip(self.param_groups, pre_optim.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group["params"], pre_group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data

                grad_dif_norm += torch.norm(p=2, input=d_p - prev_d_p).item()
                param_dif_norm += torch.norm(p=2, input=p.data - prev_p.data).item()
            group["grad_dif_norm"] = grad_dif_norm
            group["param_dif_norm"] = param_dif_norm

    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        loss = None
        if closure is None:
            loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"]
            damping = group["damping"]
            amplifier = group["amplifier"]
            theta = group["theta"]
            grad_dif_norm = group["grad_dif_norm"]
            param_dif_norm = group["param_dif_norm"]

            if param_dif_norm > 0 and grad_dif_norm > 0:
                lr_new = (
                    min(
                        lr * numpy.sqrt(1 + amplifier * theta),
                        param_dif_norm / (damping * grad_dif_norm),
                    )
                    + eps
                )
            else:
                lr_new = lr * numpy.sqrt(1 + amplifier * theta)
            theta = lr_new / lr
            group["theta"] = theta
            group["lr"] = lr_new
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["weight_decay"] != 0:
                    d_p.add_(group["weight_decay"], p.data)
                p.data.add_(d_p, alpha=-lr_new)
        return loss


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
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.local_setting.lr,
            momentum=0.9,
            weight_decay=self.args.local_setting.wd,
            nesterov=True,
        )
        self.hypers = args.method.hyperparams
        self.extra_loss = FedDecorrLoss()

    def train(self):
        # list_for_df = []
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []

        for epoch in range(self.args.local_setting.epochs):
            epoch_DC = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    feature, log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)

                    extra_loss = self.extra_loss(feature)
                epoch_DC.append(extra_loss.item())

                # loss = loss + extra_loss*self.hypers.mu"]

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f} DC: {:.6f} Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        sum(epoch_DC) / len(epoch_DC),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
                # list_for_df.append(
                # [self.round, epoch, sum(epoch_loss) / len(epoch_loss),sum(epoch_DC) / len(epoch_DC)])
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        weights = self.model.cpu().state_dict()
        return weights, {"train_loss_epoch": epoch_loss}


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
