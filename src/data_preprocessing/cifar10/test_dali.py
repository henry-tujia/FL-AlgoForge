import nvidia.dali.types as types
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import numpy as np
import nvidia.dali.ops as ops
import random
import math


class CIFAR10_truncated(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        repeate=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        cifar_dataobj = CIFAR10(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CifarIterator:
    def __init__(self, ds, batch_size) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.length = len(ds)

    def __iter__(self):
        # indexes = list()

        images = []
        labels = []
        for i in range(self.length):
            image, label = self.ds[i]
            images.append(np.ascontiguousarray(image))
            labels.append(np.ascontiguousarray(label))

            if len(images) == self.batch_size:
                yield images, labels
                images.clear()
                labels.clear()

        if len(images) > 0:
            idx_sup = random.choices(
                list(range(self.length)), k=self.batch_size - len(images)
            )
            for index in idx_sup:
                image, label = self.ds[index]
                images.append(np.ascontiguousarray(image))
                labels.append(np.ascontiguousarray(label))
            yield images, labels

    @property
    def size(self):
        return math.ceil(self.length / self.batch_size) * self.batch_size


class ExternalSourcePipeline(Pipeline):
    def __init__(self, dl, batch_size, num_threads, device_id, train=True):
        super(ExternalSourcePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12
        )
        self.train = train
        self.image_input = ops.ExternalSource(device="gpu")
        self.label_input = ops.ExternalSource(device="gpu")
        # self.source = ops.ExternalSource(source=iter(dl), num_outputs=2, device="gpu")
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.length = dl.size
        self.dl = dl
        self.iter = iter(dl)
        if self.train:
            self.flip = ops.Flip(device="gpu", horizontal=1)
        self.transpose = ops.Transpose(device="gpu", perm=[2, 0, 1])
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        self.length = len(ds)
        # self._batch_size = batch_size

    def define_graph(self):
        # images, labels = self.source()
        self.images = self.image_input()
        self.labels = self.label_input()
        images = self.resize(self.images)
        if self.train:
            images = self.flip(images)
        images = self.transpose(images)
        images = self.normalize(images)
        return images, self.labels

    def iter_setup(self):
        try:
            images, labels = next(self.iter)
            # if len(images) < self.batch_size:
            #     # just add last one
            #     tmp_images = images[-1]
            #     tmp_label = labels[-1]
            #     for _ in range(self.batch_size-len(images)):
            #         images.append(tmp_images)
            #         labels.append(tmp_label)
            self.feed_input(self.images, images)
            self.feed_input(self.labels, labels)

        except StopIteration:
            self.iter = iter(self.dl)
            raise StopIteration


class DALIDataloader(DALIGenericIterator):
    def __init__(
        self,
        pipeline,
        output_map=["image", "label"],
        auto_reset=True,
    ):
        # self.size = pipeline.length
        self.batch_size = pipeline.max_batch_size
        self.output_map = output_map
        super().__init__(
            pipelines=pipeline,
            size=pipeline.length,
            auto_reset=auto_reset,
            output_map=output_map,
            # last_batch_policy=LastBatchPolicy.FILL,
            last_batch_padded=True,
            fill_last_batch=True,
        )

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            batch = batch[0]
            return [batch[self.output_map[0]], batch[self.output_map[1]]]
        data = super().__next__()[0]
        return [data[self.output_map[0]], data[self.output_map[1]]]


batch_size = 64
ds = CIFAR10_truncated(
    root="/vepfs/DI/user/haotan/FL/Multi-FL-Training-main/datasets/cifar10",
    dataidxs=list(range(100)),
)
dl = CifarIterator(ds=ds, batch_size=batch_size)

pipe = ExternalSourcePipeline(
    dl=dl, batch_size=batch_size, num_threads=2, device_id=0, train=True
)

pii = DALIDataloader(pipe)

for e in range(10):
    for i, (images, labels) in enumerate(pii):
        print(f"epoch {e}: images size is {images.size(0)}")
    print(i)