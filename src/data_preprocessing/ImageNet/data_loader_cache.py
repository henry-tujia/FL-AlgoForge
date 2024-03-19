import logging
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import pathlib

from dataset_folder import ImageNet


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


def _data_transforms_ImageNet():
    train_transform = transforms.Compose(
        [   
            # transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )

    # "test": transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    #     ]
    # ),

    return train_transform, valid_transform


def load_imagenet_data(datadir):
    transform_train, transform_test = _data_transforms_ImageNet()
    root_path = pathlib.Path(datadir)

    train_ds = ImageNet(root_path, "train")
    test_ds = ImageNet(root_path, "test", transform=transform_test)

    # X_train, y_train = train_ds.data, train_ds.target
    # X_test, y_test = test_ds.data, test_ds.target

    return train_ds, test_ds  # X_train, y_train, X_test, y_test


def partition_data(dataset, datadir, partition, n_nets, alpha):
    # logging.info("*********partition data***************")
    train_ds, test_ds = load_imagenet_data(datadir)
    # X_train, y_train, X_test, y_test = load_imagenet_data(datadir)
    X_train, y_train = train_ds.data, train_ds.target
    X_test, y_test = test_ds.data, test_ds.target
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = np.unique(y_train, return_index=False).shape[0]
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
        K = np.unique(y_train, return_index=False).shape[0]
        if num == np.unique(y_train, return_index=False).shape[0]:
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

    return train_ds, test_ds, net_dataidx_map, traindata_cls_counts


def get_dataloader_ImageNet_truncated(
    train_ds,
    test_ds,
    train_bs,
    test_bs,
    dataidxs=None,
):
    """
    imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
    """
    # global train_ds
    # global test_ds
    transform_train, transform_test = _data_transforms_ImageNet()
    # if not train_ds:
    train_ds_client = ImageNet(
        # root=data_dir,
        # tag="train",
        transform=transform_train,
        dataset=train_ds,
        dataidxs=dataidxs,
    )
    # test_ds_client = ImageNet(
    #     root=data_dir, tag="train", transform=transform_test, dataset=test_ds
    # )

    train_dl = data.DataLoader(
        dataset=train_ds_client,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


# dataset, datadir, partition, n_nets, alpha
def get_client_idxes_dict(data_dir, partition_method, partition_alpha, client_number):
    train_ds, test_ds, net_dataidx_map, traindata_cls_counts = partition_data(
        "", data_dir, partition_method, client_number, partition_alpha
    )
    class_num = 200

    return train_ds, test_ds, net_dataidx_map, class_num, traindata_cls_counts


def get_client_dataloader(
    train_ds,
    test_ds,
    data_dir,
    batch_size,
    net_dataidx_map,
    val_batchsize=16,
    client_idx=None,
    train=True,
):
    if train:
        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_ImageNet_truncated(
            # data_dir=data_dir,
            train_ds=train_ds,
            test_ds=test_ds,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=dataidxs,
        )
        return train_data_local, test_data_local
    else:
        # train_data_global, test_data_global = get_dataloader_ImageNet_truncated(
        #     data_dir=data_dir,
        #     train_bs=batch_size,
        #     test_bs=batch_size,
        # )
        return test_ds


if __name__ == "__main__":
    import time
    import functools

    start_time = time.time()
    train_ds, test_ds, net_dataidx_map, class_num, traindata_cls_counts = (
        get_client_idxes_dict(
            pathlib.Path(
                "/vepfs/DI/user/haotan/FL/Multi-FL-Training-main/datasets/tinyimagenet"
            ),
            "hetero",
            0.1,
            100,
        )
    )
    get_client_dataloader = functools.partial(get_client_dataloader, train_ds, test_ds)

    all_time = time.time() - start_time
    for i in range(10):
        train_dl, test_dl = get_client_dataloader(
            # train_ds=train_ds,
            # test_ds=test_ds,
            data_dir=pathlib.Path(
                "/vepfs/DI/user/haotan/FL/Multi-FL-Training-main/datasets/tinyimagenet"
            ),
            batch_size=64,
            net_dataidx_map=net_dataidx_map,
            client_idx=i,
            train=True,
        )

        print(len(next(enumerate(train_dl))))
        # for image,label in train_dl:
        #     print("batch got")
        #     break
    ten_clients_time = time.time() - all_time

    print(f"pure_dataset_time:{all_time}\n{ten_clients_time=}")
