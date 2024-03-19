import torch
import torch.nn as nn
import torch.nn.functional as F

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
import pandas
import copy
from torch.multiprocessing import current_process


def l2_norm_batch(v):
    norms = (v**2).sum([1, 2, 3]) ** 0.5
    return norms


def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def get_input_grad(model, X, y, eps, delta_init="none", backprop=False):
    if delta_init == "none":
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == "random_uniform":
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == "random_corner":
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError("wrong delta init")

    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad


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
        # self.extra_loss = FedDecorrLoss()

    def train(self):
        # list_for_df = []
        # train the local model
        self.model.to(self.device)
        self.model.train()
        model_global = copy.deepcopy(self.model)
        model_global.to(self.device)
        model_global.eval()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            batch_extra_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)

                grad1 = get_input_grad(
                    self.model,
                    images,
                    labels,
                    eval(self.hypers.eps),
                    delta_init="none",
                    backprop=True,
                )

                grad2 = get_input_grad(
                    self.model,
                    images,
                    labels,
                    eval(self.hypers.eps),
                    delta_init="random_uniform",
                    backprop=False,
                )

                grad3 = get_input_grad(
                    model_global,
                    images,
                    labels,
                    eval(self.hypers.eps),
                    delta_init="random_uniform",
                    backprop=False,
                )

                grads_nnz_idx = ((grad1**2).sum([1, 2, 3]) ** 0.5 != 0) * (
                    (grad2**2).sum([1, 2, 3]) ** 0.5 != 0
                )
                grad1, grad2, grad3 = (
                    grad1[grads_nnz_idx],
                    grad2[grads_nnz_idx],
                    grad3[grads_nnz_idx],
                )
                grad1_norms, grad2_norms, grad3_norms = (
                    l2_norm_batch(grad1),
                    l2_norm_batch(grad2),
                    l2_norm_batch(grad3),
                )
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                grad3_normalized = grad3 / grad3_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))

                cos_ = torch.sum(grad1_normalized * grad3_normalized, (1, 2, 3))

                logits = torch.cat(
                    (cos_.reshape(-1, 1), (1 - cos).reshape(-1, 1)), dim=1
                )
                logits /= self.hypers.temp
                labels = torch.zeros(images.size(0)).to(self.device).long()

                extra_loss = self.hypers.grad_align_cos_lambda * self.criterion(
                    logits, labels
                )

                # extra_loss = self.hypers.grad_align_cos_lambda"] * (
                #     cos.mean()
                # ) - self.hypers.grad_align_cos_lambda2"] * (cos_.mean())
                loss += extra_loss
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                batch_extra_loss.append(extra_loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f} \tExtra: {}  Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        sum(batch_extra_loss) / len(batch_extra_loss),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
                # list_for_df.append(
                # [self.round, epoch, sum(epoch_loss) / len(epoch_loss)])
        weights = self.model.cpu().state_dict()
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        return weights, {"train_loss_epoch": epoch_loss}


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
