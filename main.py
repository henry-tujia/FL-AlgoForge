'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.multiprocessing import Queue, set_start_method

import data_preprocessing.custom_multiprocess as cm
import methods.fedavg as fedavg
import methods.fedbalance as fedbalance
import methods.fedprox as fedprox
import methods.fedrs as fedrs
import methods.moon as moon
import methods.fedmc as fedmc

# from torch.utils.tensorboard import SummaryWriter
import wandb
from models.alexnet import alexnet as alexnet
from models.preresnet import preresnet20 as preresnet

# from models.resnet_model import resnet32 as resnet
from models.resnet_model import resnet8 as resnet8
from models.resnet_model import resnet20 as resnet


torch.multiprocessing.set_sharing_strategy('file_system')

# methods
sys.path.append('/mnt/data/th')


def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedmc', metavar='N',
                        help='Options are: fedavg, fedprox, moon, mixup, stochdepth, gradaug, fedalign')

    parser.add_argument('--experi', type=str, default='0',
                        help='the times of experi')

    parser.add_argument('--dataset', type=str,
                        default='cifar10', help="name of dataset")

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.2, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=20, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;',
                        type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=0.45, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=2, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=0.1, metavar='MT',
                        help='Fraction of clients to sample')

    args = parser.parse_args()

    return args

# Setup Functions


def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: If you want every run to be exactly the same each time
    # uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions


def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])


def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number*args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients/args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError(
                "Sampled client number not divisible by number of threads")
    return mapping_dict


def init_net():
    if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
        Model = resnet8
    elif args.dataset == "cifar100":
        Model = resnet
    return Model


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed()
    # get arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    root_path = "/mnt/data/th/FedTH/data"
    args.datadir = os.path.join(root_path, "dataset", args.dataset)

    if args.dataset == "cifar10":
        from FedML.fedml_api.data_preprocessing.cifar10.data_loader import (
            get_client_dataloader, get_client_idxes_dict)
    elif args.dataset == "cifar100":
        from FedML.fedml_api.data_preprocessing.cifar100.data_loader import (
            get_client_dataloader, get_client_idxes_dict)
    elif args.dataset == "cinic10":
        from FedML.fedml_api.data_preprocessing.cinic10.data_loader import (
            get_client_dataloader, get_client_idxes_dict)
    elif args.dataset == "imagenet":
        from FedML.fedml_api.data_preprocessing.ImageNet.data_loader import (
            get_client_dataloader, get_client_idxes_dict)
    elif args.dataset == "emnist":
        from FedML.fedml_api.data_preprocessing.emnist.data_loader import (
            get_client_dataloader, get_client_idxes_dict)
    elif args.dataset == "svhn":
        from utils.utils import get_client_dataloader, get_client_idxes_dict
    elif "feature" in args.dataset:
        from FedTH.data.digits_feature import (get_client_dataloader,
                                               get_client_idxes_dict)

    dict_client_idexes, class_num, client_infos = get_client_idxes_dict(
        args.datadir, args.partition_method, args.partition_alpha, args.client_number)
    test_dl = get_client_dataloader(
        args.datadir, args.batch_size, dict_client_idexes, client_idx=None, train=False)

    mapping_dict = allocate_clients_to_threads(args)
    # init method and model type
    if args.method == 'fedavg':
        Server = fedavg.Server
        Client = fedavg.Client

        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }

        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num
                        } for i in range(args.thread_number)]

    elif args.method == 'fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }

        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num
                        } for i in range(args.thread_number)]

    elif args.method == 'fedmc':
        Server = fedmc.Server
        Client = fedmc.Client
        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }

        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num, "client_infos":client_infos, "alpha":0.1
                        } for i in range(args.thread_number)]

    elif args.method == 'moon':
        Server = moon.Server
        Client = moon.Client
        Model = init_net()
        model_paras = {
            "num_classes": class_num, "KD": True, "projection": True
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num
                        } for i in range(args.thread_number)]

    elif args.method == 'fedbalance':
        Server = fedbalance.Server
        Client = fedbalance.Client

        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }

        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):

            model_paras_local = {
                "new": model_paras,
                "local": {"model": alexnet, "paras": {"num_classes": class_num}
                          }
            }
        elif args.dataset == "cifar100":
            model_paras_local = {
                "new": model_paras,
                "local": {"model": alexnet, "paras": {"num_classes": class_num}
                          }
            }

        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras_local, 'num_classes':class_num, "client_infos":client_infos
                        } for i in range(args.thread_number)]

    elif args.method == 'fedrs':
        Server = fedrs.Server
        Client = fedrs.Client
        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num, "client_infos":client_infos, "alpha":0.1
                        } for i in range(args.thread_number)]

    elif args.method == 'fedrod':
        Server = fedrs.Server
        Client = fedrs.Client

        Model = init_net()
        model_paras = {
            "num_classes": class_num
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':class_num, "client_infos":client_infos, "alpha":0.1
                        } for i in range(args.thread_number)]

    else:
        raise ValueError(
            'Invalid --method chosen! Please choose from availible methods.')

    os.environ["HTTPS_PROXY"] = "http://10.21.0.15:7890"

    wandb.init(
        project="FedTH",
        group=args.method,
        entity="henrytujia",
        job_type="Bash Script")

    wandb.config.update(args)

    # init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))

    # Start server and get initial outputs
    pool = cm.DreamPool(args.thread_number, init_process,
                        (client_info, Client))
    # init server
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}'.format(os.getcwd(),
                                                               time.strftime("%Y%m%d_%H%M%S"), args.method, args.epochs, args.client_number)
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
    server = Server(server_dict, args)
    server_outputs = server.start()
    # Start Federated Training
    # Allow time for threads to start up
    time.sleep(15*(args.client_number/100))
    for r in range(args.comm_round):
        logging.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = pool.map(run_clients, server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        # items = ["local_test_acc"]

        res = np.array([x['results'] for x in client_outputs]).mean()

        # for item, data in zip(items, res):
        #     wandb.log({item: data}, step=r)
        wandb.log({"local_test_acc": res}, step=r)
        server_outputs, acc = server.run(client_outputs)
        wandb.log({'global_test_acc': acc}, step=r)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end-round_start))
    pool.close()
    pool.join()
