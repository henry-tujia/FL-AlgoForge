import torch
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model


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


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = Init_Model(args).model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
