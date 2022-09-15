'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML
'''
import re
import torch
import numpy as np
import random
import data_preprocessing.data_loader as dl
import argparse
from models.resnet import resnet56, resnet18
from models.resnet_gradaug import resnet56 as resnet56_gradaug
from models.resnet_gradaug import resnet18 as resnet18_gradaug
from models.resnet_stochdepth import resnet56 as resnet56_stochdepth
from models.resnet_stochdepth import resnet18 as resnet18_stochdepth
from models.resnet_fedalign import resnet56 as resnet56_fedalignw
from models.resnet_fedalign import resnet18 as resnet18_fedalign
from models.resnet_fednonlocal import resnet_nonlocal
from models.resnet_fednonlocal import resnet as resnet_nonlocal_server
from models.resnet_balance import resnet_fedbalance
from models.resnet_balance import resnet_server as resnet_fedbalance_server
from models.resnet_rs import resnet_server as resnet_rs_server
from models.resnet_rs import resnet_client as resnet_rs_client
from models.preresnet import preresnet20 as preresnet
from models.densenet import densenet100bc as densenet
from models.alexnet import alexnet as alexnet
from models.resnet_model import resnet32 as resnet

from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time
# from torch.utils.tensorboard import SummaryWriter
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

# methods
import methods.fedavg as fedavg
import methods.gradaug as gradaug
import methods.fedprox as fedprox
import methods.moon as moon
import methods.stochdepth as stochdepth
import methods.mixup as mixup
import methods.fedalign as fedalign
import methods.fednonlocal as fednonlocal
import methods.fedbalance as fedbalance
import methods.fedrs as fedrs
import data_preprocessing.custom_multiprocess as cm
import sys
sys.path.append('/mnt/data/th')


def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedavg', metavar='N',
                        help='Options are: fedavg, fedprox, moon, mixup, stochdepth, gradaug, fedalign')

    # parser.add_argument('--data_dir', type=str, default='data/cifar100',
    #                     help='data directory: data/cifar100, data/cifar10, or another dataset')

    parser.add_argument('--experi', type=str, default='0',
                        help='the times of experi')

    parser.add_argument('--dataset', type=str,
                        default='cifar10', help="name of dataset")

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--partition_alpha', type=float, default=0.2, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
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
# 0.25
    parser.add_argument('--width', type=float, default=0.25, metavar='WI',
                        help='minimum width for subnet training')

    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for subnet training')

    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets sampled during training')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=1, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=0.1, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--stoch_depth', default=0.5, type=float,
                        help='stochastic depth probability')

    parser.add_argument('--gamma', default=0.0, type=float,
                        help='hyperparameter gamma for mixup')
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
        from FedML.fedml_api.data_preprocessing.cifar10.data_loader import get_client_idxes_dict, get_client_dataloader
    elif args.dataset == "cifar100":
        from FedML.fedml_api.data_preprocessing.cifar100.data_loader import get_client_idxes_dict, get_client_dataloader
    elif args.dataset == "cinic10":
        from FedML.fedml_api.data_preprocessing.cinic10.data_loader import get_client_idxes_dict, get_client_dataloader
    elif args.dataset == "imagenet":
        from FedML.fedml_api.data_preprocessing.ImageNet.data_loader import get_client_idxes_dict, get_client_dataloader
    elif args.dataset == "emnist":
        from FedML.fedml_api.data_preprocessing.emnist.data_loader import get_client_idxes_dict, get_client_dataloader
    elif args.dataset == "svhn":
        from utils.utils import get_client_idxes_dict, get_client_dataloader
    elif "feature" in args.dataset:
        from FedTH.data.digits_feature import get_client_idxes_dict, get_client_dataloader


    dict_client_idexes, class_num, client_infos = get_client_idxes_dict(
        args.datadir, args.partition_method, args.partition_alpha, args.client_number)
    test_dl = get_client_dataloader(
        args.datadir, args.batch_size, dict_client_idexes, client_idx=None, train=False)

    mapping_dict = allocate_clients_to_threads(args)
    # init method and model type
    if args.method == 'fedavg':
        Server = fedavg.Server
        Client = fedavg.Client

        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
            input_nc = 1 if args.dataset == "femnist" else 3
            feature_dim = 576 if args.dataset == "femnist" or args.dataset == "digits+feature" else 784
            blocks = 10 if args.dataset == "cinic10" else 2
            net_mode = 'resnet'
            in_channels = 4
            numclass = 10
            Model = resnet_nonlocal_server
            model_paras = {
                "blocks": blocks, "input_nc": input_nc, "feature_dim": feature_dim, "net_mode": net_mode, "in_channels": in_channels, "numclass": numclass
            }

        elif args.dataset == "cifar100":
            Model = resnet
            numclass = 100
            model_paras = {
            "num_classes": numclass
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': numclass}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':numclass
                        } for i in range(args.thread_number)]
    elif args.method == 'gradaug':
        Server = gradaug.Server
        Client = gradaug.Client
        Model = resnet56_gradaug if 'cifar' in args.data_dir else resnet18_gradaug
        width_range = [args.width, 1.0]
        resolutions = [32, 28, 24, 20] if 'cifar' in args.data_dir else [
            224, 192, 160, 128]
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global,
                       'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    elif args.method == 'fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global,
                       'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method == 'moon':
        Server = moon.Server
        Client = moon.Client
        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
            input_nc = 1 if args.dataset == "femnist" else 3
            feature_dim = 576 if args.dataset == "femnist" or args.dataset == "digits+feature" else 784
            blocks = 10 if args.dataset == "cinic10" else 2
            net_mode = 'resnet'
            in_channels = 4
            numclass = 10
            Model = resnet_nonlocal_server
            model_paras = {
                "blocks": blocks, "input_nc": input_nc, "feature_dim": feature_dim, "net_mode": net_mode, "in_channels": in_channels, "numclass": numclass
            }

        elif args.dataset == "cifar100":
            Model = resnet
            numclass = 100
            model_paras = { 
            "num_classes": numclass,
            "KD":True
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': numclass}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':numclass
                        } for i in range(args.thread_number)]
        # Model = resnet56 if 'cifar' in args.data_dir else resnet18
        # server_dict = {'train_data': train_data_global, 'test_data': test_data_global,
        #                'model_type': Model, 'num_classes': class_num}
        # client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
        #                 'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method == 'stochdepth':
        Server = stochdepth.Server
        Client = stochdepth.Client
        Model = resnet56_stochdepth if 'cifar' in args.data_dir else resnet18_stochdepth
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global,
                       'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method == 'mixup':
        Server = mixup.Server
        Client = mixup.Client
        Model = resnet56 if 'cifar' in args.data_dir else resnet18
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global,
                       'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num} for i in range(args.thread_number)]
    elif args.method == 'fedalign':
        Server = fedalign.Server
        Client = fedalign.Client
        Model = resnet_fedalign  # resnet56_fedalign#resnet_fedalign #
        width_range = [args.width, 1.0]
        resolutions = [32]  # if 'cifar' in args.data_dir else [224]
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'num_classes': class_num}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
                        'width_range': width_range, 'resolutions': resolutions} for i in range(args.thread_number)]
    elif args.method == 'fednonlocal':
        Server = fednonlocal.Server
        Client = fednonlocal.Client
        Model_server = resnet_nonlocal_server
        Model_client = resnet_nonlocal

        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
            input_nc = 1 if args.dataset == "femnist" else 3
            feature_dim = 576 if args.dataset == "femnist" or args.dataset == "digits+feature" else 784
            blocks = 10 if args.dataset == "cinic10" else 2
            net_mode = 'resnet'
            in_channels = 4
            numclass = 10
        model_paras = {
            "blocks": blocks, "input_nc": input_nc, "feature_dim": feature_dim, "net_mode": net_mode, "in_channels": in_channels, "numclass": numclass
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model_server, 'model_paras': model_paras, 'num_classes': numclass}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model_client, 'model_paras': model_paras, 'num_classes':numclass
                        } for i in range(args.thread_number)]
    
    elif args.method == 'fedbalance':
        Server = fedbalance.Server
        Client = fedbalance.Client



        # # Model_server = resnet_fedbalance_server
        # # Model_client = 
        # Model = resnet_fedbalance_server

        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
            input_nc = 1 if args.dataset == "femnist" else 3
            feature_dim = 576 if args.dataset == "femnist" or args.dataset == "digits+feature" else 784
            blocks = 10 if args.dataset == "cinic10" else 2
            net_mode = 'resnet'
            in_channels = 4
            numclass = 10
            Model = resnet_fedbalance_server
            model_paras = {
            "blocks": blocks, "input_nc": input_nc, "feature_dim": feature_dim, "net_mode": net_mode, "in_channels": in_channels, "numclass": numclass
        }

            model_paras_local = {
            "new":model_paras,
            "local":{"model":alexnet,"paras":{"num_classes": numclass}
            }
        }
        elif args.dataset == "cifar100":
            Model = resnet
            numclass = 100
            model_paras = {
            "num_classes": numclass
        }
            model_paras_local = {
            "new":model_paras,
            "local":{"model":preresnet,"paras":{"num_classes": numclass}
            }
        }

        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': numclass}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras_local, 'num_classes':numclass, "client_infos":client_infos
                        } for i in range(args.thread_number)]

    elif args.method == 'fedrs':
        Server = fedrs.Server
        Client = fedrs.Client

        if args.dataset in ("cifar10", "cinic10", "femnist", "svhn", "digits+feature", "office+feature"):
            input_nc = 1 if args.dataset == "femnist" else 3
            feature_dim = 576 if args.dataset == "femnist" or args.dataset == "digits+feature" else 784
            blocks = 10 if args.dataset == "cinic10" else 2
            net_mode = 'resnet'
            in_channels = 4
            numclass = 10
            Model = resnet_nonlocal_server
            model_paras = {
                "blocks": blocks, "input_nc": input_nc, "feature_dim": feature_dim, "net_mode": net_mode, "in_channels": in_channels, "numclass": numclass
            }

        elif args.dataset == "cifar100":
            Model = resnet
            numclass = 100
            model_paras = { 
            "num_classes": numclass
        }
        server_dict = {'train_data': test_dl, 'test_data': test_dl,
                       'model_type': Model, 'model_paras': model_paras, 'num_classes': numclass}
        client_dict = [{'train_data': dict_client_idexes, 'test_data': dict_client_idexes, 'get_dataloader': get_client_dataloader, 'device': i % torch.cuda.device_count(),
                        'client_map': mapping_dict[i], 'model_type': Model, 'model_paras': model_paras, 'num_classes':numclass, "client_infos":client_infos, "alpha":0.1
                        } for i in range(args.thread_number)]

    else:
        raise ValueError(
            'Invalid --method chosen! Please choose from availible methods.')

    os.environ["HTTPS_PROXY"] = "http://10.21.0.15:7890"
   
    wandb.init(
    project="FedTH",
    group = args.method, #+"_preresnet"
    entity = "henrytujia",
    job_type = args.experi)

    wandb.config.update(args)

    # init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))

    # Start server and get initial outputs
    pool = cm.MyPool(args.thread_number, init_process, (client_info, Client))
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
        items = ["before_train_losses", "before_train_accs", "before_test_losses", "before_test_accs",
                 "after_train_losses", "after_train_accs", "after_test_losses", "after_test_accs"]

        res = np.array([x['results'] for x in client_outputs]).mean(axis=1)

        for item, data in zip(items, res):
            wandb.log({item: data},step=r)
            # writer.add_scalar(item, data, r)

        server_outputs, [loss, acc] = server.run(client_outputs)
        wandb.log({'global_test_loss': loss,'global_test_acc': acc},step=r)
        # writer.add_scalar('global_test_loss', loss, r)
        # writer.add_scalar('global_test_acc', acc, r)
        round_end = time.time()
        logging.info('Round {} Time: {}s'.format(r, round_end-round_start))
    pool.close()
    pool.join()
