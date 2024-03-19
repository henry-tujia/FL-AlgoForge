import torch
import torch.nn as nn

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
import pandas
from torch.multiprocessing import current_process


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
            lr=self.args.local_setting.lrcal_setting.lrcal_setting.lrcal_setting.lr,
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
        epoch_extra_loss = []
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
                    feature, log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)

                extra_loss = self.extra_loss(feature)
                loss += extra_loss * self.hypers.mu

                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                batch_extra_loss.append(extra_loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_extra_loss.append(sum(batch_extra_loss) / len(batch_extra_loss))
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
        weights = self.model.cpu().state_dict()
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        return weights, {
            "train_loss_epoch": epoch_loss,
            "FedDecorrLoss": epoch_extra_loss,
        }


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
