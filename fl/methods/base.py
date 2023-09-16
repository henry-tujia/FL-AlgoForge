import numpy as np
import torch
from torch.multiprocessing import current_process


class Base_Client:
    def __init__(self, client_dict, args):
        self.train_data = client_dict["train_data"]
        self.test_data = client_dict["test_data"]
        self.device = client_dict["device"]
        self.model_type = client_dict["model_type"]
        self.num_classes = client_dict["num_classes"]
        self.args = args
        self.round = 0
        self.client_map = client_dict["client_map"]
        self.train_dataloader = None
        self.test_dataloader = None
        self.client_index = None
        self.get_dataloader = client_dict["get_dataloader"]
        self.before_val = False
        self.after_val = False
        self.distances = None
        self.client_infos = client_dict["client_infos"]
        self.weight_test = None
        self.loggers = client_dict["loggers"]

    # def get_last_round(self, client_idx):
    #     return self.last_select_round_dict[client_idx]

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)

    def get_cdist_test(self, client_idx):
        client_dis = self.client_cnts[client_idx]
        dist = client_dis / client_dis.sum()  # 个数的比例
        cdist = dist
        return cdist.to(self.device)

    def init_client_infos(self):
        client_cnts = {}
        for client, info in self.client_infos.items():
            cnts = []
            for c in range(self.num_classes):
                if c in info.keys():
                    num = info[c]
                else:
                    num = 0
                cnts.append(num)

            cnts = torch.FloatTensor(np.array(cnts))
            client_cnts.update({client: cnts})

        return client_cnts

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.logger = self.loggers[client_idx]
            # self.logger = self.logger_method(
            #     self.args.save_path, str(client_idx), mode="client"
            # )
            self.load_client_state_dict(received_info)
            self.train_dataloader, self.test_dataloader = self.get_dataloader(
                self.args.datadir,
                self.args.batch_size,
                self.train_data,
                client_idx=client_idx,
                train=True,
            )
            if (
                self.args.client_sample < 1.0
                and self.train_dataloader._iterator is not None
                and self.train_dataloader._iterator._shutdown
            ):
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            self.client_cnts = self.init_client_infos()
            num_samples = len(self.train_dataloader) * self.args.batch_size

            self.logger.info(
                f"***********************{self.round}***********************"
            )

            weights = self.train()
            if self.args.local_valid:  # and self.round == last_round:
                self.weight_test = self.get_cdist_test(
                    client_idx).reshape((1, -1))
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
                self.args.client_sample < 1.0
                and self.train_dataloader._iterator is not None
            ):
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
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
        weights = self.model.cpu().state_dict()
        return weights

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

                pred = self.model(x)
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
        # print(acc.device,self.weight_test.device)
        weighted_acc = acc.reshape((1, -1)).mean()
        self.logger.info(
            "************* Client {} Acc = {:.2f} **************".format(
                self.client_index, weighted_acc.item()
            )
        )
        return weighted_acc


class Base_Server:
    def __init__(self, server_dict, args):
        self.train_data = server_dict["train_data"]
        self.test_data = server_dict["test_data"]
        self.device = server_dict["device"]
        self.model_type = server_dict["model_type"]
        self.num_classes = server_dict["num_classes"]
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.logger = server_dict["logger"]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        acc = self.test()
        # self.logger = self.logger_method(
        #     self.args.save_path, "server", "server")
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            self.logger.info("Save Best Model...")
            torch.save(
                self.model.state_dict(), "{}/{}.pt".format(self.save_path, "server")
            )
            self.acc = acc
        return server_outputs, acc

    def start(self):
        # with open('{}/config.txt'.format(self.save_path), 'a+') as config:
        #     config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c["results"]
                         for c in client_info]) / len(client_info)
        out_str = "Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n".format(
            acc, client_acc, self.round
        )
        self.logger.info(out_str)
        # with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
        #     out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup["client_index"])
        client_sd = [c["weights"] for c in client_info]
        cw = [
            c["num_samples"] / sum([x["num_samples"] for x in client_info])
            for c in client_info
        ]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)
        # if self.args.save_client and self.round == self.args.comm_round - 1:
        #     for client in client_info:
        #         torch.save(
        #             client["weights"],
        #             "{}/client_{}.pt".format(self.save_path,
        #                                      client["client_index"]),
        #         )
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    def test_inner(self, data):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        # test_loss = 0.0
        test_sample_number = 0.0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item()
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            # logging.info(
            #     "************* Server Acc = {:.2f} **************".format(acc))
        return acc, test_sample_number

    def test(self):
        acc, _ = self.test_inner(self.test_data)
        self.logger.info(
            "************* Server Acc = {:.2f} **************".format(acc))
        return acc
