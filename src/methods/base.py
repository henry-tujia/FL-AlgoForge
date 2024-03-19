import numpy as np
import torch
from torch.multiprocessing import current_process
# import nvidia.dali as dali

class Base_Client:
    def __init__(self, client_dict, args):
        # for data
        self.train_data = client_dict["train_data"]
        self.test_data = client_dict["test_data"]
        self.train_dataloader = None
        self.test_dataloader = None
        self.get_dataloader = client_dict["get_dataloader"]

        # for model
        self.num_classes = args.datasets.num_classes
        # self.model_type = client_dict["model_type"]
        self.device = client_dict["device"]

        # for train
        self.before_val = False
        self.after_val = False
        self.weight_test = None

        # for federated
        self.round = 0
        self.client_index = None
        self.client_map = client_dict["client_map"]
        self.client_infos = client_dict["client_infos"]

        # for log
        self.logger_method = client_dict["logger_method"]

        # for others
        self.distances = None
        self.args = args

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
            self.logger = self.logger_method(
                self.args.paths.output_dir / "clients" / "logs",
                str(client_idx),
                mode="client",
            )
            self.load_client_state_dict(received_info)
            self.train_dataloader, self.test_dataloader = self.get_dataloader(
                self.args.datasets.datadir,
                self.args.datasets.batch_size,
                self.train_data,
                client_idx=client_idx,
                train=True,
            )
            # if (
            #     self.args.federated_settings.client_sample < 1.0
            #     and self.train_dataloader._iterator is not None
            #     and self.train_dataloader._iterator._shutdown
            # ):
            #     self.train_dataloader._iterator = self.train_dataloader._get_iterator()
            self.client_index = client_idx
            self.client_cnts = self.init_client_infos()
            num_samples = len(self.train_dataloader) * self.args.datasets.batch_size

            self.logger.info(
                f"***********************{self.round}***********************"
            )

            weights, train_res = self.train()
            if self.args.local_setting.local_valid:  # and self.round == last_round:
                self.weight_test = self.get_cdist_test(client_idx).reshape((1, -1))
                self.acc_dataloader = self.test_dataloader
                val_res = self.test()
            else:
                val_res = {}

            client_results.append(
                {
                    "weights": weights,
                    "num_samples": num_samples,
                    "client_index": self.client_index,
                    "result": dict(**train_res, **val_res),
                }
            )
            # if (
            #     self.args.federated_settings.client_sample < 1.0
            #     and self.train_dataloader._iterator is not None
            # ):
            #     self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def train(self):
        # list_for_df = []
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tBatch {}: \tLoss: {:.6f}  Thread {}  Map {}".format(
                        epoch,
                        batch_idx,
                        sum(epoch_loss) / len(epoch_loss),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )

            # if isinstance(self.train_dataloader, dali.plugin.pytorch.DALIGenericIterator):
            #     self.train_dataloader.reset()
        weights = self.model.cpu().state_dict()
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        return weights, {"train_loss_epoch": epoch_loss}

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        preds = None
        labels = None
        acc = None
        losses = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                losses.append(loss)
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
        return {
            "val_loss": sum(losses) / len(losses),
            "val_weighted_acc": weighted_acc.item(),
            "val_acc": (
                (preds == labels).float().sum() / (labels).float().sum()
            ).item(),
        }


class Base_Server:
    def __init__(self, server_dict, args):
        self.train_data = server_dict["train_data"]
        self.test_data = server_dict["test_data"]
        self.device = server_dict["device"]
        # self.model_type = server_dict["model_type"]
        self.num_classes = args.datasets.num_classes
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.logger_method = server_dict["logger_method"]

    def run(self, received_info):
        self.logger = self.logger_method(
            self.args.paths.output_dir, "server", mode="server"
        )
        self.start_params = [
            param.clone().detach().cpu() for param in self.model.parameters()
        ]
        server_outputs = self.operations(received_info)
        param_norm = self.compute_grad_norm()
        acc = self.test()
        # self.logger = self.logger_method(
        #     self.args.paths.output_dir, "server", "server")
        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            self.logger.info("Save Best Model...")
            torch.save(
                self.model.state_dict(),
                "{}/{}.pt".format(self.args.paths.output_dir, "server"),
            )
            self.acc = acc
        return server_outputs, {"acc": acc, "param_norm": param_norm}

    def start(self):
        return [
            self.model.cpu().state_dict()
            for x in range(self.args.federated_settings.thread_number)
        ]

    def log_info(self, client_info, acc):
        client_acc = 0  # sum([c.results for c in client_info]) / len(client_info)
        out_str = "Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n".format(
            acc, client_acc, self.round
        )
        self.logger.info(out_str)

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
        return [
            self.model.cpu().state_dict()
            for x in range(self.args.federated_settings.thread_number)
        ]

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
        self.logger.info("************* Server Acc = {:.2f} **************".format(acc))
        return acc

    def compute_grad_norm(
        self,
    ):
        # 计算参数变化量
        params_change = (
            torch.tensor(
                [
                    torch.norm(param.cpu() - init_param).item()
                    for param, init_param in zip(
                        self.model.parameters(), self.start_params
                    )
                ]
            )
            .mean()
            .item()
        )

        return params_change
