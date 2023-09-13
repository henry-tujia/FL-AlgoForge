import copy
import logging
import math

import torch
from methods.base import Base_Client, Base_Server
from torch.autograd import Variable
from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(
            **client_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        return_loss = self.train_once()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(
                    self.client_index, epoch, sum(epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))

        # 此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = copy.deepcopy(self.model)

        # {key: value for key,
        #            value in self.model.cpu().state_dict().items()}
        return weights, return_loss

    def train_once(self):
        total_loss = 0
        total_sample_number = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss * labels.shape[0]
                total_sample_number += labels.shape[0]
            loss = total_loss/total_sample_number
        return Variable(loss, requires_grad=False)

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

            weights, l_loacl = self.train()
            if self.args.local_valid:  # and self.round == last_round:
                self.weight_test = self.get_cdist_test(
                    client_idx).reshape((1, -1))
                self.acc_dataloader = self.test_dataloader
                after_test_acc = self.test()
            else:
                after_test_acc = 0

            client_results.append({'weights': weights, 'l_local': l_loacl, 'num_samples': num_samples,
                                  'results': after_test_acc, 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def test(self):

        cidst = self.get_cdist(self.client_index)
        self.model.to(self.device)
        self.model.eval()

        preds = None
        labels = None
        acc = None
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                logits = self.model(x)

                calibrated_logits = logits-cidst
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(calibrated_logits, 1)
                if preds is None:
                    preds = predicted.cpu()
                    labels = target.cpu()
                else:
                    preds = torch.concat((preds, predicted.cpu()), dim=0)
                    labels = torch.concat((labels, target.cpu()), dim=0)
        for c in range(self.num_classes):
            temp_acc = (((preds == labels) * (labels == c)).float() /
                        (max((labels == c).sum(), 1))).sum().cpu()
            if acc is None:
                acc = temp_acc.reshape((1, -1))
            else:
                acc = torch.concat((acc, temp_acc.reshape((1, -1))), dim=0)
        weighted_acc = acc.reshape((1, -1)).mean()
        logging.info(
            "************* Client {} Acc = {:.2f} **************".format(self.client_index, weighted_acc.item()))
        return weighted_acc


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(**server_dict["model_paras"])
        self.local_gradient_round = {}
        self.hypers = server_dict['hypers']
        self.alpha = self.hypers['alpha']
        self.tau = self.hypers['tau']

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'].to(torch.device("cpu")) for c in client_info]
        old_model = self.model.cpu().span_model_params_to_vec()
        g_locals = []

        for model_para in client_sd:
            g_locals.append(old_model-model_para.span_model_params_to_vec())
        for g_l, info in zip(g_locals, client_info):
            self.local_gradient_round.update(
                {info['client_index']: [self.round, g_l]})

        order_grads = copy.deepcopy(g_locals)
        order = [_ for _ in range(len(order_grads))]
        temp = sorted(
            list(zip([x['l_local'] for x in client_info], order)), key=lambda x: x[0])
        order = [x[1] for x in temp]
        keep_original = []
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order)-1)*(1-self.alpha)):]
        g_locals_l2_norm = []
        for g_l in g_locals:
            g_locals_l2_norm.append(torch.norm(g_l)**2)
        for i in range(len(order_grads)):
            if i in keep_original:
                continue
            for j in order:
                if i == j:
                    continue
                else:
                    dot = g_locals[j] @ order_grads[i]
                    if dot < 0:
                        order_grads[i] -= dot/g_locals_l2_norm[j]*g_locals[j]
        weights = torch.Tensor([1/len(order_grads)] *
                               len(order_grads)).float().to(self.device)
        gt = weights@(torch.stack(order_grads).to(self.device))

        if self.round >= self.tau:
            for k in range(self.tau-1, -1, -1):
                gcs = [value[1] for key, value in self.local_gradient_round.items() if key ==
                       self.round-k and gt @ value[1].to(self.device) < 0]
                if gcs:
                    gcs = torch.vstack(gcs)
                    g_con = torch.sum(gcs, dim=0)
                    dot = gt@g_con
                    if dot < 0:
                        gt -= dot/(torch.norm(g_con)**2)*g_con
        gnorm = torch.norm(weights@torch.stack(g_locals).to(self.device))
        gt = gt/torch.norm(gt)*gnorm

        self.model.to(self.device)
        gt.to(self.device)

        for i, p in enumerate(self.model.parameters()):
            p.data -= gt[self.model.Loc_reshape_list[i]]

        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]
