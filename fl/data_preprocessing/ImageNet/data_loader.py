import logging
import certifi

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


from .datasets import ImageNet
from .datasets import ImageNet_truncated
from .datasets_hdf5 import ImageNet_hdf5
from .datasets_hdf5 import ImageNet_truncated_hdf5


#logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# class Cutout(object):
#     def __init__(self, length):
#         self.length = length

#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)

#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img


def _data_transforms_ImageNet():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image_size = 224
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    return train_transform, valid_transform

def partition_data(net_dataidx_map_origin, partition, n_nets, alpha):
    logging.info("*********partition data***************")

    if partition == "homo":
        total_num = net_dataidx_map_origin[len(net_dataidx_map_origin)-1][-1]
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        K = len(net_dataidx_map_origin)
        N = net_dataidx_map_origin[len(net_dataidx_map_origin)-1][-1]
        logging.info("N = " + str(N))
        net_dataidx_map = {}
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            (begin, end) = net_dataidx_map_origin[k]
            idx_k = np.arange(begin,end)
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

# # for centralized training
# def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
#     return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)


# # for local devices
# def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
#     return get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_ImageNet_truncated(imagenet_dataset_train, imagenet_dataset_test, train_bs,
                                      test_bs, dataidxs=None, net_dataidx_map=None):
    """
        imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
    """
    if type(imagenet_dataset_train) == ImageNet:
        dl_obj = ImageNet_truncated
    elif type(imagenet_dataset_train) == ImageNet_hdf5:
        dl_obj = ImageNet_truncated_hdf5
    else:
        raise NotImplementedError()

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(imagenet_dataset_train, dataidxs, net_dataidx_map, train=True, transform=transform_train,
                      download=False)
    test_ds = dl_obj(imagenet_dataset_test, dataidxs=None, net_dataidx_map=None, train=False, transform=transform_test,
                     download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True,
                        pin_memory=True, num_workers=4)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True,
                        pin_memory=True, num_workers=4)

    return train_dl, test_dl


# def get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs=None):
#     dl_obj = ImageNet

#     transform_train, transform_test = _data_transforms_ImageNet()

#     train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
#     test_ds = dl_obj(datadir, dataidxs=None, train=False, transform=transform_test, download=False)


#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True,
#                         pin_memory=True, num_workers=4)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True,
#                         pin_memory=True, num_workers=4)

#     return train_dl, test_dl


# def get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
#     dl_obj = ImageNet

#     transform_train, transform_test = _data_transforms_ImageNet()

#     train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True,
#                         pin_memory=True, num_workers=4)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True,
#                         pin_memory=True, num_workers=4)

#     return train_dl, test_dl


# def distributed_centralized_ImageNet_loader(dataset, data_dir,
#                         world_size, rank, batch_size):
#     """
#         Used for generating distributed dataloader for 
#         accelerating centralized training 
#     """

#     train_bs=batch_size
#     test_bs=batch_size

#     transform_train, transform_test = _data_transforms_ImageNet()
#     if dataset == 'ILSVRC2012':
#         train_dataset = ImageNet(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=True,
#                                 transform=transform_train) 

#         test_dataset = ImageNet(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=False,
#                                 transform=transform_test) 
#     elif dataset == 'ILSVRC2012_hdf5':
#         train_dataset = ImageNet_hdf5(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=True,
#                                 transform=transform_train) 

#         test_dataset = ImageNet_hdf5(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=False,
#                                 transform=transform_test) 

#     train_sam = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
#     test_sam = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

#     train_dl = data.DataLoader(train_dataset, batch_size=train_bs, sampler=train_sam,
#                         pin_memory=True, num_workers=4)
#     test_dl = data.DataLoader(test_dataset, batch_size=test_bs, sampler=test_sam,
#                         pin_memory=True, num_workers=4)


#     class_num = 1000

#     train_data_num = len(train_dataset)
#     test_data_num = len(test_dataset)

#     return train_data_num, test_data_num, train_dl, test_dl, \
#            None, None, None, class_num


# def load_partition_data_ImageNet(dataset, data_dir,
#                                  partition_method=None, partition_alpha=None, client_number=100, batch_size=10):

#     if dataset == 'ILSVRC2012':
#         train_dataset = ImageNet(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=True)

#         test_dataset = ImageNet(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=False)
#     elif dataset == 'ILSVRC2012_hdf5':
#         train_dataset = ImageNet_hdf5(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=True)

#         test_dataset = ImageNet_hdf5(data_dir=data_dir,
#                                 dataidxs=None,
#                                 train=False)


#     min_size = 0

#     while min_size < 100:

#         net_dataidx_map = train_dataset.get_net_dataidx_map()

#         net_dataidx_map_new = partition_data(net_dataidx_map_origin=net_dataidx_map,partition=partition_method,n_nets=client_number,alpha=partition_alpha)

#         min_size = min([len(idx_j) for idx_j in net_dataidx_map_new.values()])

#     class_num = len(net_dataidx_map)

#     # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
#     train_data_num = len(train_dataset)
#     test_data_num = len(test_dataset)
#     class_num_dict = train_dataset.get_data_local_num_dict()

#     # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)

#     train_data_global, test_data_global = get_dataloader_ImageNet_truncated(train_dataset, test_dataset,
#                                                                             train_bs=batch_size, test_bs=batch_size,
#                                                                             dataidxs=None, net_dataidx_map=None, )

#     logging.info("train_dl_global number = " + str(len(train_data_global)))
#     logging.info("test_dl_global number = " + str(len(test_data_global)))

#     # get local dataset
#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()

#     for client_idx in range(client_number):
#         dataidxs = net_dataidx_map_new[client_idx]
#         data_local_num_dict[client_idx] = len(dataidxs)

#         local_data_num = data_local_num_dict[client_idx]

#         # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

#         # training batch size = 64; algorithms batch size = 32
#         # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#         #                                          dataidxs)
#         train_data_local, test_data_local = get_dataloader_ImageNet_truncated(train_dataset, test_dataset,
#                                                                               train_bs=batch_size, test_bs=batch_size,
#                                                                               dataidxs=dataidxs,
#                                                                               net_dataidx_map=net_dataidx_map_new)

#         # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#         # client_idx, len(train_data_local), len(test_data_local)))
#         train_data_local_dict[client_idx] = train_data_local
#         test_data_local_dict[client_idx] = test_data_local

#     logging.info("data_local_num_dict: %s" % data_local_num_dict)
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def get_client_idxes_dict(data_dir, partition_method, partition_alpha, client_number):

    global train_dataset_global

    train_dataset_global = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                classes=[]) #x for x in range(100)

    global test_dataset_global

    test_dataset_global  = ImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                classes=[]) #x for x in range(100)

    min_size = 0

    while min_size < 100:

        net_dataidx_map = train_dataset_global.get_net_dataidx_map()

        net_dataidx_map_new = partition_data(net_dataidx_map_origin=net_dataidx_map,partition=partition_method,n_nets=client_number,alpha=partition_alpha)

        min_size = min([len(idx_j) for idx_j in net_dataidx_map_new.values()])

    class_num = len(net_dataidx_map)
    
    logging.info("traindata_cls_counts = " + str(class_num))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    logging.info("total data num = " + str(train_data_num))
    return net_dataidx_map_new,class_num


def get_client_dataloader(data_dir, batch_size, net_dataidx_map, val_batchsize = 16, client_idx = None,train = True):

    global train_dataset_global
    global test_dataset_global

    if train:
        dataidxs = net_dataidx_map[client_idx]
        train_idx = dataidxs[:int(len(dataidxs)*0.8)]
        val_idx = dataidxs[int(len(dataidxs)*0.8):]

        train_data_local, _ = get_dataloader_ImageNet_truncated(train_dataset_global, test_dataset_global,
                                                                              train_bs=batch_size, test_bs=batch_size,
                                                                              dataidxs=train_idx,
                                                                              net_dataidx_map=net_dataidx_map)

        val_data_local, _ = get_dataloader_ImageNet_truncated(train_dataset_global, test_dataset_global,
                                                                              train_bs=val_batchsize, test_bs=batch_size,
                                                                              dataidxs=val_idx,
                                                                              net_dataidx_map=net_dataidx_map)

        logging.info("train batch: {},val batch: {}".format(len(train_data_local),len(val_data_local)))
        return train_data_local, val_data_local
    else:
        _, test_data_global = get_dataloader_ImageNet_truncated(train_dataset_global, test_dataset_global,
                                                                            train_bs=batch_size, test_bs=batch_size,
                                                                            dataidxs=None, net_dataidx_map=None, )
                                                                            
        logging.info("all test batch: {}".format(len(test_data_global)))                                      

        return test_data_global

train_dataset_global = None
test_dataset_global = None


def test_run(use_softmax = False):
    import torchvision.models as models
    import torch.nn as nn
    import torch.optim as optim
        #Load Resnet18
    model_ft = models.resnet18()
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 200)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    #Multi GPU
    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0, 1]).cuda()

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

    # train_data_num, test_data_num, train_data_global, test_data_global, \
    # data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = \
    #     load_partition_data_ImageNet("ILSVRC2012", data_dir,
    #                                  partition_method="hetero", partition_alpha=0.2, client_number=client_number,
    #                                  batch_size=64)

    # print(train_data_num, test_data_num, class_num)
    # print(data_local_num_dict)

    # print(train_data_num, test_data_num, class_num)
    # print(data_local_num_dict)

    # i = 0
    model_ft.train()
    for iter in range(60):
        losses = []
        for data, label in train_dl_local:
            data = data.to(device)
            label = label.to(device)

            logs = model_ft(data)
            if use_softmax:
                logs = torch.softmax(logs,1)
            loss = criterion(logs,label)

            loss.backward()
            optimizer_ft.step()
            losses.append(loss.item())
        model_ft.eval()
        correct = 0
        losses_test = []
        with torch.no_grad():
            for data, label in val_dl_local:
                data = data.to(device)
                label = label.to(device)

                logs = model_ft(data)

                # if use_softmax:
                #     logs = torch.softmax(logs,1)
                # loss = criterion(logs,label)
                # losses_test.append(loss.item())
                y_pred = logs.max(1).indices
                correct += (y_pred == label).sum().item()/float(label.size(0))

        print("iter:{},train loss is {:.3f},test acc is {:3f}".format(iter,np.array(losses).mean(),correct))#
        # print(label)

if __name__ == '__main__':
    
    from datasets import ImageNet
    from datasets import ImageNet_truncated
    from datasets_hdf5 import ImageNet_hdf5
    from datasets_hdf5 import ImageNet_truncated_hdf5
    
    data_dir = '/mnt/data/th/FedTH/data/imagenet'
    # data_dir = '/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5'

    client_number = 100

    dict_client_idexes, num_classes = get_client_idxes_dict(data_dir, 'hetero', 0.2, 100)
    test_dl = get_client_dataloader(data_dir, 64, dict_client_idexes,train=False)

    train_dl_local, val_dl_local  = get_client_dataloader(data_dir, 64, dict_client_idexes, client_idx = 0)


    test_run()
    
    print("=============================\n")
    # test_run(True)


    # print("=============================\n")

    # for client_idx in range(client_number):
    #     # i = 0
    #     print(len(train_data_local_dict[client_idx]))
    #     # for batch_idx, (data, target) in enumerate(train_data_local_dict[client_idx]): 
    #     #     print(data)
    #     #     print(label)
    #     #     i += 1
    #     #     if i > 5:
    #     #         break
