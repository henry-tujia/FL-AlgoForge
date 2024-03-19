"""
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
"""
import numpy as np
import pandas
import torch

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.prev_model = self.model_type(**client_dict["model_paras"]).to(self.device)
        self.global_model = self.model_type(**client_dict["model_paras"]).to(
            self.device
        )
        self.prev_model.load_state_dict(self.model.state_dict())
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.local_setting.lr,
            momentum=0.9,
            weight_decay=self.args.local_setting.wd,
            nesterov=True,
        )
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.hypers = args.method.hyperparams

    def run(self, received_info):
        client_results = []
        self.global_model.load_state_dict(received_info["global"])
        for client_idx in self.client_map[self.round]:
            self.prev_model.load_state_dict(received_info["prev"][client_idx])
            self.load_client_state_dict(received_info["global"])
            # self.train_dataloader = self.train_data[client_idx]
            # self.test_dataloader = self.test_data[client_idx]
            self.train_dataloader, self.test_dataloader = self.get_dataloader(
                self.args.datasets.datadir,
                self.args.datasets.batch_size,
                self.train_data,
                client_idx=client_idx,
                train=True,
            )
            if (
                self.args.federated_settings.client_sample < 1.0
                and self.train_dataloader._iterator is not None
                and self.train_dataloader._iterator._shutdown
            ):
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.datasets.batch_size
            self.client_cnts = self.init_client_infos()
            weights = self.train()
            # last_round = self.get_last_round(client_idx)
            if self.args.local_setting.local_valid:  # and self.round == last_round:
                self.weight_test = self.get_cdist_test(client_idx).reshape((1, -1))
                self.acc_dataloader = self.test_dataloader
                after_test_acc = self.test()
            else:
                after_test_acc = 0

            client_results.append(
                {
                    "weights": weights,
                    "num_samples": num_samples,
                    "results": after_test_acc,
                    "client_index": self.client_index,
                }
            )
            if (
                self.args.federated_settings.client_sample < 1.0
                and self.train_dataloader._iterator is not None
            ):
                self.train_dataloader._iterator._shutdown_workers()
        self.round += 1
        return client_results

    def train(self):
        # list_for_df = []
        # train the local model
        self.model.to(self.device)
        self.global_model.to(self.device)
        self.prev_model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                #####
                    pro1, out = self.model(x)
                    pro2, _ = self.global_model(x)

                    posi = self.cos(pro1, pro2)
                    logits = posi.reshape(-1, 1)

                    pro3, _ = self.prev_model(x)
                    nega = self.cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    logits /= self.hypers.temp
                    labels = torch.zeros(x.size(0)).to(self.device).long()

                    loss2 = self.hypers.mu * self.criterion(logits, labels)

                    loss1 = self.criterion(out, target)
                    loss = loss1 + loss2
                #####
                loss.backward()
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
                # list_for_df.append(
                # [self.round, epoch, sum(epoch_loss) / len(epoch_loss)])
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients" /
        # "dfs"/f"{self.client_index}.xlsx")
        weights = self.model.cpu().state_dict()
        # self.prev_model.load_state_dict(weights)
        return weights, {"train_loss_epoch": epoch_loss}

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        preds = None
        labels = None
        acc = None
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                if preds is None:
                    preds = predicted.cpu()
                    labels = target.cpu()
                else:
                    preds = torch.concat((preds, predicted.cpu()), dim=0)
                    labels = torch.concat((labels, target.cpu()), dim=0)
        for c in range(self.num_classes):
            temp_acc = (
                (
                    ((preds == labels) * (labels == c)).float()
                    / (max((labels == c).sum(), 1))
                )
                .sum()
                .cpu()
            )
            if acc is None:
                acc = temp_acc.reshape((1, -1))
            else:
                acc = torch.concat((acc, temp_acc.reshape((1, -1))), dim=0)
        weighted_acc = acc.reshape((1, -1)).mean()
        self.logger.info(
            "************* Client {} Acc = {:.2f} **************".format(
                self.client_index, weighted_acc.item()
            )
        )
        return weighted_acc


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.prev_models = {
            x: self.model.cpu().state_dict() for x in range(self.args.federated_settings.client_number)
        }
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        self.logger = self.logger_method(self.args.paths.output_dir, "server", "server")
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            torch.save(
                self.model.state_dict(),
                "{}/{}.pt".format(self.args.paths.output_dir, "server"),
            )
            self.acc = acc
        for x in received_info:
            self.prev_models[x["client_index"]] = x["weights"]
        server_outputs = [
            {"global": g, "prev": self.prev_models} for g in server_outputs
        ]
        return server_outputs, acc

    def start(self):
        return [
            {"global": self.model.cpu().state_dict(), "prev": self.prev_models}
            for x in range(self.args.federated_settings.thread_number)
        ]

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        # test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                _, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item()# * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            self.logger.info(
                "************* Server Acc = {:.2f} **************".format(acc)
            )
        return acc
