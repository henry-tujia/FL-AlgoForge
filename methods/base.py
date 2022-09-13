import numpy as np
import torch
import logging
import json
from torch.multiprocessing import current_process


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type']
        self.num_classes = client_dict['num_classes']
        self.args = args
        self.round = 0
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        self.get_dataloader = client_dict['get_dataloader']
        self.before_val = False
        self.after_val = False
        self.distances = None
        self.alpha = None


    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            # get_client_dataloader(args.datadir, args.batch_size, dict_client_idexes,train=False)
            # train_dl_local, val_dl_local  = get_client_dataloader(args.datadir, args.local_bs, dict_client_idexes, client_idx = idx)
            self.train_dataloader, self.test_dataloader = self.get_dataloader(
                self.args.datadir, self.args.batch_size, self.train_data, client_idx=client_idx, train=True)
            # self.test_dataloader = self.test_data[client_idx]
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader)*self.args.batch_size

            if self.before_val:
                self.acc_dataloader = self.train_dataloader
                before_train_loss, before_train_acc = self.test()
                self.acc_dataloader = self.test_dataloader
                before_test_loss, before_test_acc = self.test()
            else:
                before_train_loss, before_train_acc,before_test_loss, before_test_acc = 0,0,0,0
            epoch_loss, weights = self.train()
            if self.after_val:
                self.acc_dataloader = self.train_dataloader
                _, after_train_acc = self.test()
                self.acc_dataloader = self.test_dataloader
                after_test_loss, after_test_acc = self.test()
            else:
                after_train_acc,after_test_loss, after_test_acc = 0,0,0

            client_results.append({'weights': weights, 'num_samples': num_samples, 'results': [before_train_loss, before_train_acc, before_test_loss, before_test_acc, np.array(
                epoch_loss).mean(), after_train_acc, after_test_loss, after_test_acc], 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def train(self):
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
        weights = self.model.cpu().state_dict()
        return epoch_loss, weights

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item()
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number)*100
            logging.info(
                "************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return test_loss/len(self.acc_dataloader), acc


class Base_Server():
    def __init__(self, server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model_type = server_dict['model_type']
        self.num_classes = server_dict['num_classes']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        loss, acc = self.test()
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(self.model.state_dict(),
                       '{}/{}.pt'.format(self.save_path, 'server'))
            self.acc = acc
        return server_outputs, [loss, acc]

    def start(self):
        with open('{}/config.txt'.format(self.save_path), 'a+') as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c['results'][-1]
                         for c in client_info])/len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(
            acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples']/sum([x['num_samples']
                                   for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        if self.args.save_client:
            for client in client_info:
                torch.save(
                    client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def test_inner(self,data):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item()
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number)*100
            # logging.info(
            #     "************* Server Acc = {:.2f} **************".format(acc))
        return test_loss/len(data), acc, test_sample_number

    def test_mutildata(self):
        res = []
        for data in self.test_data:
            res.append(self.test_inner(data))
        # print(res)
        res = np.array(res)
        total = res[:, -1].sum()
        res = (res[:, :-1].transpose()*res[:, -1]).transpose().sum(0)
        # print(res)
        loss, acc = res/total

        return loss, acc

    def test(self):
        if isinstance(self.test_data, list):
            loss, acc = self.test_mutildata()
        else:
            loss, acc, _ = self.test_inner(self.test_data)
        logging.info(
            "************* Server Acc = {:.2f} **************".format(acc))
        return loss, acc
