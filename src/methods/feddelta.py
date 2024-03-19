from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
import numpy
from torch.multiprocessing import current_process
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import copy


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


class DeltaScheduler:
    def __init__(self, optimizer: Optimizer, model, theta, gamma, delta) -> None:
        super().__init__()
        self.pre_model = copy.deepcopy(model)
        self.theta = theta
        self.gamma = gamma
        self.delta = delta
        self.optimizer = optimizer
        self.gradient_pre = 0

    def adjust_lr(self, model):
        self.diff_model = (
            -model.span_model_params_to_vec()
            + self.pre_model.span_model_params_to_vec()
        )
        self.l2_diff = torch.norm(p=2, input=self.diff_model)
        self.gradient_cur = self.diff_model / self.optimizer.param_groups[0]["lr"]
        self.gradient_diff = self.gradient_cur - self.gradient_pre
        debug_a = torch.max(self.gradient_diff)
        debug_b = torch.min(self.gradient_diff)
        self.l2_gradient_diff = 2 * torch.norm(p=2, input=self.gradient_diff)

        self.lr = min(
            self.gamma * self.l2_diff / self.l2_gradient_diff,
            ((1 + self.delta * self.theta) ** 0.5)
            * self.optimizer.param_groups[0]["lr"],
        )

        self.theta = self.lr / self.optimizer.param_groups[0]["lr"]
        self.pre_model = copy.deepcopy(model)
        self.gradient_pre = self.gradient_cur

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.hypers = args.method.hyperparams
        self.optimizer = sgddelta(
            self.model.parameters(),
            {
                "lr": self.args.local_setting.lr,
                "weight_decay": self.args.local_setting.wd,
                "amplifier": self.hypers.amplifier,
                "theta": self.hypers.theta,
                "damping": self.hypers.damping,
                "eps": eval(self.hypers.eps),
            },
        )

        self.prev_model = copy.deepcopy(self.model)
        self.prev_optim = sgddelta(
            self.prev_model.parameters(), {"weight_decay": self.args.local_setting.wd}
        )

    def train(self):
        self.model.to(self.device)
        self.model.train()

        self.prev_model.to(self.device)
        self.prev_model.train()

        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                self.prev_optim.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                loss.backward()

                prev_log_probs = self.prev_model(images)
                prev_loss = self.criterion(prev_log_probs, labels)
                prev_loss.backward()

                self.optimizer.compute_dif_norms(pre_optim=self.prev_optim)
                self.prev_model.load_state_dict(self.model.state_dict())
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
        weights = self.model.cpu().state_dict()

        return weights, {"train_loss_epoch": epoch_loss}


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
