import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from collections import defaultdict
import random
import shutil
import time
import torch
import yaml
from models.Resnet_ import Resnet32 as resnet32
from models.Resnet_ import Resnet8 as resnet8
import methods.moon as moon
import methods.fedrs as fedrs
import methods.fedprox as fedprox
import methods.feddecorr as feddecorr
import methods.fedict as fedict
import methods.fedfv as fedfv
import methods.fedbalance as fedbalance
import methods.fedavg as fedavg
from utils import custom_multiprocess, tools
import logging
from torch.multiprocessing import Queue, set_start_method
import os
from addict import Dict 


torch.multiprocessing.set_sharing_strategy('file_system')

def init_process(q, Client):
    tools.set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None


class Trainer():
    def __init__(self,method,config,save_path) -> None:
        self.method = method
        self.config_path = config
        self.save_path = save_path
        self.settings = self.read_config()
        self.DEVICE = (
            torch.device(self.settings["DEVICE"])
            if torch.cuda.is_available()    
            else torch.device("cpu")
            )
        tools.set_logger(save_path/("train.log"))

    def read_config(self):
        shutil.copy(self.config_path, self.save_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        with open(self.config_path.parent/(f"{self.method}.yaml"), "r", encoding="utf-8") as f:
            hypers = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        shutil.copy(self.config_path.parent/(f"{self.method}.yaml"), self.save_path)
        data["hypers"] = hypers
        return data

    def allocate_clients_to_threads(self):
        mapping_dict = defaultdict(list)
        for round in range(self.settings["comm_round"]):
            if self.settings["client_sample"] < 1.0:
                num_clients = int(self.settings["client_number"]*self.settings["client_sample"])
                client_list = random.sample(range(self.settings["client_number"]), num_clients)
            else:
                num_clients = self.settings["client_number"]
                client_list = list(range(num_clients))
            if num_clients % self.settings["thread_number"] == 0 and num_clients > 0:
                clients_per_thread = int(num_clients/self.settings["thread_number"])
                for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                    idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
                    mapping_dict[c].append(idxs)
            else:
                raise ValueError(
                    "Sampled client number not divisible by number of threads")
        self.mapping_dict = mapping_dict

    def init_net(self):
        if self.settings["dataset"] in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature", "covid"):
            Model = resnet8
        elif self.settings["dataset"] == "cifar100":
            Model = resnet32
        return Model
    
    def init_dataloaders(self):
        match self.settings["dataset"]:
            case "cifar10":
                from data_preprocessing.cifar10.data_loader import (
                    get_client_dataloader, get_client_idxes_dict)
            case  "cifar100":
                from data_preprocessing.cifar100.data_loader import (
                    get_client_dataloader, get_client_idxes_dict)
            case _:
                raise ValueError("Unrecognized Dataset!")
        self.dict_client_idexes, self.class_num, self.client_infos = get_client_idxes_dict(
            self.settings["datadir"], self.settings["partition_method"], self.settings["partition_alpha"], self.settings["client_number"])
        self.test_dl = get_client_dataloader(
            self.settings["datadir"], self.settings["batch_size"], self.dict_client_idexes, client_idx=None, train=False)
        self.get_client_dataloader = get_client_dataloader
        
    def init_methods(self):
        Model = self.init_net()
        model_paras = {"num_classes": self.class_num}
        hypers = self.settings["hypers"]
        self.server_dict = {'train_data': self.test_dl, 'test_data': self.test_dl, "save_path":self.save_path,
                            'model_type': Model, 'model_paras': model_paras, 'num_classes': self.class_num,'device': self.DEVICE}
        self.client_dict = [{'train_data': self.dict_client_idexes, 'test_data': self.dict_client_idexes, 'get_dataloader': self.get_client_dataloader, 'device': self.DEVICE,
                                'client_map': self.mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':self.class_num,  "client_infos":self.client_infos,"hypers":hypers
                                } for i in range(self.settings["thread_number"])]
        match self.method:
            case 'fedavg':
                self.Server = fedavg.Server
                self.Client = fedavg.Client
            case 'feddecorr':
                self.Server = feddecorr.Server
                self.Client = feddecorr.Client
                model_paras["KD"] = True
            case 'fedprox':
                self.Server = fedprox.Server
                self.Client = fedprox.Client
            case 'moon':
                self.Server = moon.Server
                self.Client = moon.Client
                model_paras["KD"] = True  
                model_paras["projection"] = True
            case 'fedrs':
                self.Server = fedrs.Server
                self.Client = fedrs.Client
            case 'fedrod':
                self.Server = fedrs.Server
                self.Client = fedrs.Client
            case 'fedfv':
                self.Server = fedfv.Server
                self.Client = fedfv.Client
            case _:
                raise ValueError(
                    'Invalid --method chosen! Please choose from availible methods.')
        self.server_dict["model_paras"] = model_paras
    def run_one_round(self,r):
        logging.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = self.pool.map(run_clients, self.server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        # res = np.array([x['results'] for x in client_outputs])  # .sum()
        server_outputs, acc = self.server.run(client_outputs)
        # logging.info(f"Acc: {acc}")
        # wandb.log({"local_test_acc": res}, step=r)
        # wandb.log({'global_test_acc': acc}, step=r)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end-round_start))

    def mutilprocess_before_run(self):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        client_info = Queue()
        for i in range(self.settings["thread_number"]):
            client_info.put((self.client_dict[i], Dict(self.settings)))

        # Start server and get initial outputs
        self.pool = custom_multiprocess.DreamPool(self.settings["thread_number"], init_process,
                            (client_info, self.Client))
        
        self.server = self.Server(self.server_dict, Dict(self.settings))
        
    def run(self):
        self.init_dataloaders()
        self.allocate_clients_to_threads()
        self.init_methods()
        self.mutilprocess_before_run()

        self.server_outputs = self.server.start()
        time.sleep(15*(self.settings["client_number"]/100))
        for r in range(self.settings["comm_round"]):
            self.run_one_round(r)
        self.pool.close()
        self.pool.join()