import torch
from src.methods.base import Base_Client, Base_Server
from src.models.init_model import Init_Model
from src.utils.image_synthesizer import Synthesizer, TensorDataset, DiffAugment
from src.utils.utils import kd_loss
from copy import deepcopy


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
        self.trajectories_list = []
        self.hypers = args.method.hyperparams
        self.synthesizer = Synthesizer(deepcopy(self.model), self.test_data, self.args)

    def run(self, received_info):
        self.logger = self.logger_method(
            self.args.paths.output_dir, "server", mode="server"
        )
        self.start_params = [
            param.clone().detach().cpu() for param in self.model.parameters()
        ]

        if self.round < self.hypers.L:
            self.trajectories_list.append(self.start_params)
        elif self.round > self.hypers.L:
            self.logger.info("Begin finetuning...")
            self.finetune()
        else:
            self.logger.info("Begin synthesizing image...")
            self.synthesizer.synthesize(
                trajectories_list=self.trajectories_list, args=self.args
            )
            self.synthesizer.evaluate(args=self.args)
            self.images_train, self.labels_train = (
                self.synthesizer.image_syn.cpu().detach(),
                self.synthesizer.label_syn.cpu().detach(),
            )

        server_outputs = self.operations(received_info)
        param_norm = self.compute_grad_norm()
        acc = self.test()
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

    def finetune(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.distill_lr,
        )
        ds = TensorDataset(self.images_train, self.labels_train)
        dl = torch.utils.data.Dataloader(
            ds, batch_size=next(self.test_data)[0].shape[0], shuffle=True
        )

        loss_fn = torch.nn.CrossEntropyLoss() if not self.args.ifsoft else kd_loss

        self.model.train()

        for _ in range(self.args.distill_epoch):
            for image, label in dl:
                optimizer.zero_grad()
                image = image.to(self.device)
                label = label.to(self.device)

                if self.args.dsa:
                    image = DiffAugment(
                        image, self.args.dsa_strategy, param=self.args.dsa_param
                    )
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=True
                ):
                    prediction = self.model(image)
                    loss = loss_fn(prediction, label)

                loss.backward()
                optimizer.step()
