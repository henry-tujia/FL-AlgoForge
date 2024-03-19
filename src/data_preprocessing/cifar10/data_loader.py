import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from .datasets import (
    CIFAR10_truncated,
    # DALIDataloader,
    # ExternalSourcePipeline,
    CifarIterator,
)

# logging.basicConfig()


# generate the non-IID distribution for all methods
def read_data_distribution(
    filename="./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt",
):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(
                        tmp[1].strip().replace(",", "")
                    )
    return distribution


def read_net_dataidx_map(
    filename="./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt",
):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    # train_transform = transforms.Compose([transforms.ToPILImage(),
    #                                     transforms.Resize((70, 70)),
    #                                    transforms.RandomCrop((64, 64)),
    #                                    transforms.ToTensor()])

    # valid_transform = transforms.Compose([
    #     transforms.ToPILImage(),transforms.Resize((70, 70)),
    #                                   transforms.CenterCrop((64, 64)),
    #                                   transforms.ToTensor()])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()
    # print(f"{datadir=}")
    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=False, transform=train_transform
    )
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=False, transform=test_transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data_ori(datadir):
    # train_transform, test_transform = _data_transforms_cifar10()
    # print(f"{datadir=}")
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=False)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=False)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    # logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        # logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 64:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif "label" in partition:
        num = eval(partition[5:])
        K = 10
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_nets)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_nets)
                for j in range(n_nets):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_nets)}
            classes_nets = []
            classes_count = [0] * K
            for i in range(n_nets):
                temp = np.random.choice(K, size=num, replace=False, p=None)
                for j in temp:
                    classes_count[j] += 1
                classes_nets.append(temp)
            for index, clss_partition in enumerate(classes_count):
                if clss_partition == 0:
                    continue
                idx_k = np.where(y_train == index)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, clss_partition)
                index_in = 0
                for net_idx in range(n_nets):
                    if index in classes_nets[net_idx]:
                        net_dataidx_map[net_idx] = np.append(
                            net_dataidx_map[net_idx], split[index_in]
                        )
                        index_in += 1
    elif partition == "hetero-fix":
        dataidx_map_file_path = (
            "./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt"
        )
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = (
            "./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt"
        )
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)
    # return get_dataloader_CIFAR10_DALI(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(
    dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
):
    return get_dataloader_test_CIFAR10(
        datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
    )


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
    )

    return train_dl, test_dl


# def get_dataloader_CIFAR10_DALI(datadir, train_bs, test_bs, dataidxs=None):
#     dl_obj = CIFAR10_truncated
#     train_ds = dl_obj(
#         datadir,
#         dataidxs=dataidxs,
#         train=True,
#         download=False,
#     )

#     train_dl = CifarIterator(ds=train_ds, batch_size=train_bs)
#     train_pipe = ExternalSourcePipeline(
#         dl=train_dl, batch_size=train_bs, num_threads=2, device_id=0, train=True
#     )

#     train_pii = DALIDataloader(train_pipe)
    
#     test_ds = dl_obj(datadir, train=False, download=False)
#     test_dl = CifarIterator(ds=test_ds, batch_size=test_bs)
#     test_pipe = ExternalSourcePipeline(
#         dl=test_dl, batch_size=test_bs, num_threads=2, device_id=0, train=False
#     )

#     test_pii = DALIDataloader(pipeline=test_pipe)

#     return train_pii, test_pii


def get_dataloader_test_CIFAR10(
    datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None
):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        download=False,
    )

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
    )

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = (
        partition_data(
            dataset, data_dir, partition_method, client_number, partition_alpha
        )
    )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info(
            "rank = %d, local_sample_number = %d" % (process_id, local_data_num)
        )
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def load_partition_data_cifar10(
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    silo_proc_num=0,
):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = (
        partition_data(
            dataset, data_dir, partition_method, client_number, partition_alpha
        )
    )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, batch_size
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )


def get_client_idxes_dict(data_dir, partition_method, partition_alpha, client_number):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = (
        partition_data("", data_dir, partition_method, client_number, partition_alpha)
    )
    class_num = len(np.unique(y_train))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r])
    #                      for r in range(client_number)])
    # logging.info("total data num = " + str(train_data_num))
    return net_dataidx_map, class_num, traindata_cls_counts


def get_client_dataloader(
    data_dir, batch_size, net_dataidx_map, val_batchsize=16, client_idx=None, train=True
):
    if train:
        dataidxs = net_dataidx_map[client_idx]
        # train_idx = dataidxs[:int(len(dataidxs)*0.9)]
        # val_idx = dataidxs[int(len(dataidxs)*0.9):]
        train_data_local, test_data_local = get_dataloader(
            "", data_dir, batch_size, batch_size, dataidxs
        )
        # val_data_local, test_data_local = get_dataloader("", data_dir, val_batchsize, val_batchsize,
        #                                          val_idx)

        # logging.info("train batch: {0},val batch: {1}".format(             #len(train_data_local), len(test_data_local)))
        return train_data_local, test_data_local
    else:
        train_data_global, test_data_global = get_dataloader(
            "", data_dir, batch_size, batch_size
        )
        return test_data_global
