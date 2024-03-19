import torch
import torch.nn as nn

# import logging
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
import pandas
from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none").to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.local_setting.lr,
            momentum=0.9,
            weight_decay=self.args.local_setting.wd,
            nesterov=True,
        )
        self.hypers = args.method.hyperparams

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

                scale_rate = (
                    (self.hypers.l_min / loss).float().to(self.device)
                )  # .type(torch.cuda.FloatTensor)
                Indicator_above = (
                    (scale_rate <= 1).float().to(self.device)
                )  # .type(torch.cuda.FloatTensor)
                Indicator_below = (
                    (scale_rate > 1).float().to(self.device)
                )  # .type(torch.cuda.FloatTensor)
                loss_scale_rate = scale_rate * Indicator_below + Indicator_above
                # loss = loss.mul(loss_scale_rate)
                loss *= loss_scale_rate

                loss = loss.mean()

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
        weights = self.model.cpu().state_dict()
        # df_save = pandas.DataFrame(list_for_df)
        # df_save.to_excel(self.args.paths.output_dir/"clients"/#"dfs"/f"{self.client_index}.xlsx")
        return weights, {"train_loss_epoch": epoch_loss}


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
