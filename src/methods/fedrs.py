import pandas
import torch

# import logging
from src.methods.feddecorr import FedDecorrLoss
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
from torch.multiprocessing import current_process
import numpy as np

features = []

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

        # self.client_infos = client_dict["client_infos"]
        # self.client_cnts = self.init_client_infos()

    def init_client_infos(self):
        client_cnts = {}
        # print(self.client_infos)

        for client, info in self.client_infos.items():
            cnts = []
            for c in range(self.num_classes):
                if c in info.keys():
                    num = 1
                else:
                    num = 0
                cnts.append(num)

            cnts = torch.FloatTensor(np.array(cnts))
            client_cnts.update({client: cnts})
        # print(client_cnts)
        return client_cnts


    def get_cdist_inner(self, idx):
        client_dis = self.client_cnts[idx]

        cdist = client_dis * (1.0 - self.hypers.mu) + self.hypers.mu
        cdist = cdist.reshape((1, -1))

        return cdist.to(self.device)

    def train(self):
        # list_for_df = []
        cidst = self.get_cdist_inner(self.client_index)
        # train the local model
        self.model.to(self.device)
        self.model.train()
        self.model.heads.head.register_forward_hook(feature_extract_hook)
        epoch_loss = []
        epoch_extra_loss = []
        for epoch in range(self.args.local_setting.epochs):
            batch_loss = []
            batch_extra_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    _ = self.model(images)
                    hs = features[-1][0]
                    # ws = self.model.model.fc.weight
                    ws = self.model.heads.head.weight

                    logits = cidst * hs.mm(ws.transpose(0, 1))
                    loss = self.criterion(logits, labels)

                    # extra_loss = self.extra_loss(hs)

                # loss == extra_loss * 0.1
                features.clear()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                # batch_extra_loss.append(extra_loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # epoch_extra_loss.append(sum(batch_extra_loss) / len(batch_extra_loss))
                self.logger.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}".format(
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                        current_process()._identity[0],
                        self.client_map[self.round],
                    )
                )
                ##list_for_df.append(
                # [self.round, epoch, sum(epoch_loss) / len(epoch_loss)])
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        # 此处交换参数以及输出新字典
        # self.model.change_paras()
        weights = {key: value for key, value in self.model.cpu().state_dict().items()}
        return weights, {
            "train_loss_epoch": epoch_loss,
            # "FedDecorrLoss": epoch_extra_loss,
        }

    def test(self):
        cidst = self.get_cdist_inner(self.client_index)
        self.model.to(self.device)
        self.model.eval()
        self.model.heads.head.register_forward_hook(feature_extract_hook)
        preds = None
        labels = None
        acc = None
        losses = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.acc_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)

                _ = self.model(x)
                hs = features[-1][0]
                ws = self.model.heads.head.weight

                logits = cidst * hs.mm(ws.transpose(0, 1))

                _, predicted = torch.max(logits, 1)
                loss = self.criterion(predicted, target)
                if preds is None:
                    preds = predicted.cpu()
                    labels = target.cpu()
                else:
                    preds = torch.concat((preds, predicted.cpu()), dim=0)
                    labels = torch.concat((labels, target.cpu()), dim=0)
                losses.append(loss)
                features.clear()
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
        return {
            "val_loss": sum(losses) / len(losses),
            "val_weighted_acc": weighted_acc.item(),
            "val_acc": (
                (preds == labels).float().sum() / (labels).float().sum()
            ).item(),
        }


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)


def feature_extract_hook(module, input, output):
    features.append(input)
