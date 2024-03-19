from collections import defaultdict
import time
import logging
from omegaconf import DictConfig, OmegaConf
import rich
import tqdm
import random
import pathlib
import torch
from torch.multiprocessing import Queue, set_start_method
from torch.utils.tensorboard import SummaryWriter
import wandb

from src.utils import custom_multiprocess, tools
from src.models.init_model import Init_Model

torch.multiprocessing.set_sharing_strategy("file_system")


def init_process(q, Client):
    tools.set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])


def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info("exiting")
        return None


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        DEVICE = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = DEVICE
        self.logger_method = tools.set_logger
        self.logger = self.logger_method(cfg.paths.output_dir, "scheduler")
        self.writer = self.setup_tensorboard()

    # def read_config(self):
    #     # shutil.copy(self.config_path, self.save_path)
    #     with self.config_path.open("r", encoding="utf-8") as f:
    #         data = yaml.load(f, Loader=yaml.FullLoader)
    #     with (self.config_path.parent / (f"{self.method}.yaml")).open(
    #         "r", encoding="utf-8"
    #     ) as f:
    #         hypers = yaml.load(f, Loader=yaml.FullLoader)

    #     # shutil.copy(self.config_path.parent / (f"{self.method}.yaml"), self.save_path)
    #     data["hypers"] = hypers
    #     data["tags"] = self.tags

    #     yaml_path = self.save_path / f"{self.method}.yaml"
    #     with yaml_path.open("w") as file:
    #         yaml.dump(data, file)

    #     return data

    def setup_tensorboard(self):
        wandb.init(
            dir=str(self.cfg.paths.output_dir),
            # set the wandb project where this run will be logged
            project="Multi-FL-Training",
            # track hyperparameters and run metadata
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        save_path = self.cfg.paths.output_dir / "tensorboard"
        pathlib.Path.mkdir(save_path, parents=True, exist_ok=True)
        writer = SummaryWriter(save_path)

        # yaml_path = save_path / "hparams.yaml"
        # with yaml_path.open("w") as file:
        #     yaml.dump(self.settings, file)

        self.logger.info(f"INIT::Tensorboard file saved at {save_path}")
        return writer

    def allocate_clients_to_threads(self):
        mapping_dict = defaultdict(list)
        for _ in range(self.cfg.federated_settings.comm_round):
            if self.cfg.federated_settings.client_sample < 1.0:
                num_clients = int(
                    self.cfg.federated_settings.client_number
                    * self.cfg.federated_settings.client_sample
                )
                client_list = random.sample(
                    range(self.cfg.federated_settings.client_number), num_clients
                )
            else:
                num_clients = self.cfg.federated_settings.client_number
                client_list = list(range(num_clients))
            if (
                num_clients % self.cfg.federated_settings.thread_number == 0
                and num_clients > 0
            ):
                clients_per_thread = int(
                    num_clients / self.cfg.federated_settings.thread_number
                )
                for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                    idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                    mapping_dict[c].append(idxs)
            else:
                raise ValueError(
                    "Sampled client number not divisible by number of threads"
                )
        self.mapping_dict = mapping_dict

    # def init_net(self):
    #     model = Init_Model(self.cfg).model
    #     # if self.cfg.datasets.dataset in (
    #     #     "cifar10",
    #     #     "cinic10",
    #     #     "femnist",
    #     #     "svhn",
    #     #     "digits+feature",
    #     #     "office+feature",
    #     #     "covid",
    #     # ):
    #     #     from src.models.Resnet_ import Resnet8 as resnet8

    #     #     Model = resnet8
    #     # elif self.cfg.datasets.dataset == "cifar100":
    #     #     from src.models.Resnet_ import Resnet32 as resnet32

    #     #     Model = resnet32
    #     return model

    def init_dataloaders(self):
        import functools
        match self.cfg.datasets.dataset:
            case "cifar10":
                from src.data_preprocessing.cifar10.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case "cifar100":
                from src.data_preprocessing.cifar100.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case "tinyimagenet":
                from src.data_preprocessing.ImageNet.data_loader import (
                    get_client_dataloader,
                    get_client_idxes_dict,
                )
            case _:
                raise ValueError("Unrecognized Dataset!")
        (
            # train_ds,
            # test_ds,
            self.dict_client_idexes,
            self.class_num,
            self.client_infos,
        ) = get_client_idxes_dict(
            self.cfg.datasets.datadir,
            self.cfg.datasets.partition_method,
            self.cfg.datasets.partition_alpha,
            self.cfg.federated_settings.client_number,
        )
        # get_client_dataloader = functools.partial(get_client_dataloader, train_ds, test_ds)

        self.test_dl = get_client_dataloader(
            self.cfg.datasets.datadir,
            self.cfg.datasets.batch_size,
            self.dict_client_idexes,
            client_idx=None,
            train=False,
        )
        self.get_client_dataloader = get_client_dataloader
        # self.logger.info(f"INIT::Data Partation\n{self.dict_client_idexes}")
        with (self.cfg.paths.output_dir / "dict_client_idexes.log").open("w") as f:
            # with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(self.dict_client_idexes, file=f)

        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.client_infos).fillna(0)  # 使用 0 填充缺失值

        plt.figure(
            figsize=(self.cfg.federated_settings.client_number, self.class_num // 2)
        )
        sns.heatmap(
            df, annot=True, cmap="viridis", square=True, fmt="g", annot_kws={"size": 7}
        )

        plt.xlabel("Clients")
        plt.ylabel("Classes")
        plt.title("Data Distribution")

        plt.savefig(
            self.cfg.paths.output_dir / "Data_Distribution.png", bbox_inches="tight"
        )
        plt.show()

        # 将图像上传到 wandb
        wandb.log(
            {
                "Data Distribution": wandb.Image(
                    str(self.cfg.paths.output_dir / "Data_Distribution.png")
                )
            }
        )

    def init_methods(self):
        # Model = Init_Model(self.cfg).model  # self.init_net()
        # model_paras = {"num_classes": self.cfg.datasets.num_classes}
        # hypers = self.settings["hypers"]
        self.server_dict = {
            "train_data": self.test_dl,
            "test_data": self.test_dl,
            # "model_type": Model,
            "device": self.device,
            "logger_method": self.logger_method,
        }
        self.client_dict = [
            {
                "train_data": self.dict_client_idexes,
                "test_data": self.dict_client_idexes,
                "get_dataloader": self.get_client_dataloader,
                "device": torch.device(f"cuda:{i%torch.cuda.device_count()}")
                if self.device.type == "cuda"
                else self.device,
                "client_map": self.mapping_dict[i],
                # "model_type": Model,
                "client_infos": self.client_infos,
                "logger_method": self.logger_method,
            }
            for i in range(self.cfg.federated_settings.thread_number)
        ]
        match self.cfg.method.method_name:
            case "fedavg":
                import src.methods.fedavg as alg
            case "feddecorr":
                import src.methods.feddecorr as alg
            case "fedunknown":
                import src.methods.fedunknown as alg
            case "fedprox":
                import src.methods.fedprox as alg
            case "moon":
                import src.methods.moon as alg
            case "fedrs":
                import src.methods.fedrs as alg
            case "fedrod":
                import src.methods.fedrod as alg
            case "fedfv":
                import src.methods.fedfv as alg
            case "feddelta":
                import src.methods.feddelta as alg
            case "fedga":
                import src.methods.fedga as alg
            case "fedscale":
                import src.methods.fedscale as alg
            case "dynafed":
                import src.methods.dynafed as alg
            case _:
                raise ValueError(
                    "Invalid --method chosen! Please choose from availible methods."
                )

        self.Server = alg.Server
        self.Client = alg.Client
        # self.server_dict["model_paras"] = model_paras
        # for i in range(len(self.client_dict)):
        #     self.client_dict[i]["model_paras"] = model_paras
        # self.logger.info(f"INIT::Server Dict\n{pp(self.server_dict,output = False)}")
        # self.logger.info(f"INIT::Client Dict\n{pp(self.client_dict[0],output = False)}")
        # self.logger.info(f"INIT::Args\n{pp(self.settings,output = False)}")

    def run_one_round(self, r):
        client_outputs = self.pool.map(run_clients, self.server_outputs)

        client_outputs = [c for sublist in client_outputs for c in sublist]
        res_for_log = {"client_results": client_outputs}
        self.server_outputs, server_res = self.server.run(client_outputs)
        res_for_log.update({"server_results": server_res})
        self.log(round=r + 1, contents=res_for_log)

        return server_res["acc"]

    def log(self, round: int, contents: dict):
        """
        {
            "weights": weights,
            "num_samples": num_samples,
            "client_index": self.client_index,
            "result": dict(**train_res, **val_res),
        }
        """
        # res = contents["client_results"]
        for client_res in contents["client_results"]:
            #  res = client_res["results"]
            for key, value in client_res["result"].items():
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        self.writer.add_scalar(
                            tag=f"client_{client_res['client_index']}/{round}/{key}",
                            scalar_value=item,
                            global_step=index,
                        )
                        # wandb.log(
                        #     {f"client_{client_res['client_index']}/{round}/{key}": item},step=index
                        # )
                else:
                    self.writer.add_scalar(
                        tag=f"client_{client_res['client_index']}/{key}",
                        scalar_value=value,
                        global_step=round,
                    )
                    wandb.log(
                        {f"client_{client_res['client_index']}/{key}": value},
                        step=round,
                    )
        for key, value in contents["server_results"].items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    self.writer.add_scalar(
                        tag=f"server/{round}/{key}",
                        scalar_value=item,
                        global_step=index,
                    )
                    # wandb.log({f"server/{round}/{key}": item},step=index)
            else:
                self.writer.add_scalar(
                    tag=f"server/{key}",
                    scalar_value=value,
                    global_step=round,
                )
                wandb.log(
                    {f"server/{key}": value},
                    step=round,
                )

    def multiprocess_before_run(self):
        try:
            # set_start_method("fork")
            set_start_method("spawn", force=True)
        except RuntimeError as e:
            raise e
        client_info = Queue()
        for i in range(self.cfg.federated_settings.thread_number):
            client_info.put((self.client_dict[i], self.cfg))

        # Start server and get initial outputs
        self.pool = custom_multiprocess.DreamPool(
            self.cfg.federated_settings.thread_number,
            init_process,
            (client_info, self.Client),
            # context=multiprocessing.get_context("spawn")
        )

        self.server = self.Server(self.server_dict, self.cfg)

    def run(self):
        self.init_dataloaders()
        self.logger.info("INIT::Data partation finished...")
        self.allocate_clients_to_threads()
        self.init_methods()
        self.multiprocess_before_run()
        self.logger.info("INIT::Starting server...")
        self.server_outputs = self.server.start()
        time.sleep(5 * (self.cfg.federated_settings.client_number / 100))
        self.logger.info("INIT::Runnging FL...")
        with tqdm.tqdm(range(self.cfg.federated_settings.comm_round)) as t:
            for r in range(self.cfg.federated_settings.comm_round):
                acc = self.run_one_round(r)
                t.set_postfix({"Acc": acc})
                t.set_description(
                    f"""Round: {r+1}/{self.cfg.federated_settings.comm_round}"""
                )
                t.update(1)
        self.pool.close()
        self.pool.join()

        tools.parser_log(self.cfg.paths.output_dir / "server.log")
        self.logger.info("RESULT::Experiment finished...")
        # wandb.close()
