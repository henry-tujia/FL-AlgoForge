'''
Code credit to https://github.com/lxcnju/FedRepo/blob/main/algorithms/fednova.py
for thier implementation of FedNova.
'''

import torch
import logging
from methods.base import Base_Client, Base_Server
import copy
from torch.multiprocessing import current_process

from torch.optim.optimizer import Optimizer


class NovaOptimizer(Optimizer):
    """ gmf: global momentum
        prox_mu: mu of proximal term
        ratio: client weight
    """

    def __init__(
        self, params, lr, ratio, gmf, prox_mu=0,
        momentum=0, dampening=0, weight_decay=0, nesterov=False, variance=0
    ):
        self.gmf = gmf
        self.ratio = ratio
        self.prox_mu = prox_mu
        self.momentum = momentum
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr < 0.0:
            raise ValueError("Invalid lr: {}".format(lr))

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, variance=variance
        )
        super(NovaOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NovaOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # weight_decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # save the first parameter w0
                param_state = self.state[p]
                if "old_init" not in param_state:
                    param_state["old_init"] = torch.clone(p.data).detach()

                # momentum:
                # v_{t+1} = rho * v_t + g_t
                # g_t = v_{t+1}
                # rho = momentum
                local_lr = group["lr"]
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                        # update momentum buffer !!!
                        param_state["momentum_buffer"] = buf

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # add proximal updates: g_t = g_t + prox_mu * (w - w0)
                if self.prox_mu != 0:
                    d_p.add_(self.prox_mu, p.data - param_state["old_init"])

                # updata accumulated local updates
                # sum(g_0, g_1, ..., g_t)
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)
                else:
                    param_state["cum_grad"].add_(local_lr, d_p)

                # update: w_{t+1} = w_t - lr * g_t
                p.data.add_(-1.0 * local_lr, d_p)

        # compute local normalizing vec, a_i
        # For momentum: a_i = [(1 - rho)^{tau_i - 1}/(1 - rho), ..., 1]
        # 1, 1 + rho, 1 + rho + rho^2, ...
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        # proximal: a_i = [(1 - eta * mu)^{\tau_i - 1}, ..., 1]
        # 1, 1 - eta * mu, (1 - eta * mu)^2 + 1, ...
        self.etamu = local_lr * self.prox_mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        # FedAvg: no momentum, no proximal, [1, 1, 1, ...]
        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1
        return


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.hypers = client_dict["hypers"]
        self.optimizer = NovaOptimizer(
            self.model.parameters(),
            lr=self.args.lr,
            gmf=self.hypers["gmf"],
            prox_mu=self.hypers["prox_mu"],
            ratio=self.hypers["ratio"],
            momentum=0.9,
            weight_decay=self.args.wd,
        )
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader, self.test_dataloader = self.get_dataloader(
                self.args.datadir, self.args.batch_size, self.train_data, client_idx=client_idx, train=True)
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            self.client_cnts = self.init_client_infos()
            num_samples = len(self.train_dataloader)*self.args.batch_size
            last_round = self.get_last_round(client_idx)
            local_grad, tau_eff = self.train()
            if self.args.local_valid:  # and self.round == last_round:
                self.weight_test = self.get_cdist_test(
                    client_idx).reshape((1, -1))
                self.acc_dataloader = self.test_dataloader
                after_test_acc = self.test()
            else:
                after_test_acc = 0

            client_results.append({'local_grads': local_grad, "tau_eff": tau_eff, 'num_samples': num_samples,
                                  'results': after_test_acc, 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()
        self.round += 1
        return client_results

    def train(self):
        init_state_dict = copy.deepcopy(self.model.state_dict())
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                             epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        local_grad = self.get_local_grad_(
            opt=self.optimizer,
            cur_params=self.model.state_dict(),
            init_params=init_state_dict
        )

        if self.hypers["prox_mu"] != 0:
            tau_eff = self.optimizer.local_steps * self.optimizer.ratio
        else:
            tau_eff = self.optimizer.local_normalizing_vec * self.optimizer.ratio

        return local_grad, tau_eff
        # weights = self.model.cpu().state_dict()
        # return weights

    def get_local_grad_(self, opt, cur_params, init_params):
        weight = opt.ratio

        grad_dict = {}
        for k in cur_params.keys():
            scale = 1.0 / opt.local_normalizing_vec
            cum_grad = init_params[k] - cur_params[k]
            try:
                cum_grad.mul_(weight * scale)
            except Exception:
                cum_grad = (cum_grad * weight * scale).long()
            grad_dict[k] = cum_grad
        return grad_dict


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(**server_dict["model_paras"])
        self.hypers = server_dict['hypers']
        self.global_momentum_buffer = {}
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        tau_eff = sum([c['tau_eff'] for c in client_info])
        local_grads = [c['local_grads'] for c in client_info]

        cum_grad = local_grads[0]

        ssd = self.model.state_dict()

        for k in local_grads[0].keys():
            for i in range(0, len(local_grads)):
                if i == 0:
                    cum_grad[k] = (local_grads[i][k] *
                                   tau_eff).to(torch.device("cpu"))
                else:
                    cum_grad[k] += (local_grads[i][k] *
                                    tau_eff).to(torch.device("cpu"))

        for k in ssd.keys():
            if self.hypers["gmf"] != 0:
                if k not in self.global_momentum_buffer:
                    self.global_momentum_buffer[k] = torch.clone(
                        cum_grad[k]
                    ).detach()
                    buf = self.global_momentum_buffer[k]
                    buf.div_(self.args.lr)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.hypers["gmf"]).add_(
                        1.0 / self.args.lr, cum_grad[k]
                    )
                try:
                    ssd[k] = ssd[k].to(torch.device("cpu"))
                    ssd[k].sub_(self.args.lr, buf)
                except Exception:
                    ssd[k] = ssd[k].to(torch.device("cpu"))
                    ssd[k] = (ssd[k] - self.args.lr * buf).long()
            else:
                try:
                    ssd[k] = ssd[k].to(torch.device("cpu"))
                    ssd[k].sub_(cum_grad[k])
                except Exception:
                    ssd[k] = ssd[k].to(torch.device("cpu"))
                    ssd[k] = (ssd[k] - cum_grad[k]).long()

        # global_model.load_state_dict(params, strict=True)
        # client_sd = [c['weights'] for c in client_info]
        # cw = [c['num_samples']/sum([x['num_samples']
        #                            for x in client_info]) for c in client_info]

        # ssd = self.model.state_dict()
        # for key in ssd:
        #     ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        # if self.args.save_client and self.round == self.args.comm_round-1:
        #     for client in client_info:
        #         torch.save(
        #             client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]
