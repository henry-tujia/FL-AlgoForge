import copy
import torch
from methods.base import Base_Client, Base_Server


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(
            **client_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd, nesterov=True)


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(
            **server_dict["model_paras"]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.hypers = server_dict["hypers"]
        self.global_optimizer = self._initialize_global_optimizer()

    def _initialize_global_optimizer(self):
        # global optimizer
        if self.hypers["glo_optimizer"] == "SGD":
            # similar as FedAvgM
            global_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hypers["glo_lr"],
                momentum=0.9,
                weight_decay=0.0
            )
        elif self.hypers["glo_optimizer"] == "Adam":
            global_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hypers["glo_lr"],
                betas=(0.9, 0.999),
                weight_decay=0.0
            )
        else:
            raise ValueError("No such glo_optimizer: {}".format(
                self.hypers["glo_optimizer"]
            ))
        return global_optimizer

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        cw = [c['num_samples']/sum([x['num_samples']
                                   for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key]*cw[i] for i, sd in enumerate(client_sd)])

        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.model)
        new_model.load_state_dict(ssd, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                self.model.parameters(), new_model.parameters()
            ):
                param.grad = param.data - new_param.data

        # replace some non-parameters's state dict
        state_dict = self.model.state_dict()
        for name in dict(self.model.named_parameters()).keys():
            ssd[name] = state_dict[name]
        self.model.load_state_dict(ssd, strict=True)

        # optimization
        self.global_optimizer = self._initialize_global_optimizer()
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]
