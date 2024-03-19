from torchvision.datasets import ImageFolder
import pathlib
from PIL import Image
import torch.utils.data as data
import numpy
import multiprocessing
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import time
import pandas
import functools
# import dask.dataframe as dd


def loader(sample_info,transformer = None):
    sample_info: tuple
    image_path = sample_info[0]
    image = Image.open(image_path)
    # image_array = numpy.array(image).transpose((2, 0, 1))
    time.sleep(0.01)
    if transformer:
        image = transformer(image)
    return [image, sample_info[-1]]  # list(sample_info).extend(image_array)


class ImageNet(data.Dataset):
    def __init__(
        self,
        root: pathlib.Path = "",
        tag: str = "",
        dataset: data.Dataset = None,
        dataidxs=None,
        transform=None,
        target_transform=None,
    ) -> None:
        if isinstance(root, str):
            root = pathlib.Path(root)
        self.root = root
        self.tag = tag
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

        self.loader = functools.partial(loader, transformer = transform)

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if not self.dataset:
            dataset = ImageFolder(self.root / self.tag)
            self.classes = dataset.classes
            self.samples = self.load_images(dataset.samples)
            self.data, self.target = self.split_samples()
        else:
            self.data, self.target = self.dataset.data, self.dataset.target
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]

        # return data, target

    def split_samples(self):
        x, y = [], []
        for item in self.samples:
            x.append(item[0])
            y.append(item[-1])

        return numpy.array(x), numpy.array(y)

    def load_images(self, samples):
        cpu_count = multiprocessing.cpu_count()
        data = []
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=cpu_count) as pool:
            for sample in tqdm(
                pool.imap_unordered(self.loader, samples), total=len(samples)
            ):
                data.append(sample)
        pool.close()
        pool.join()
        # data = process_map(loader, samples, max_workers=cpu_count, chunksize=100)

        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# root_path = pathlib.Path(
#     "/vepfs/DI/user/haotan/FL/Multi-FL-Training-main/datasets/tiny-224"
# )

# for tag in ["train", "val", "test"]:
#     dataset = ImageNet(root=root_path, tag="train")
