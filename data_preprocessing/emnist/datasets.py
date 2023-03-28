import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import EMNIST
import torchvision.transforms as tt
import torch

#logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EMNIST_truncated(data.Dataset):
    def __init__(self, dataidxs=None,dataset = None):
        super(EMNIST_truncated, self).__init__()
        self.dataidxs = dataidxs
        self.dataset = dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        """
            Args:
        root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
            and  ``EMNIST/processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        """

        emnist_dataobj = self.dataset

        data = emnist_dataobj.data #
        target = np.array(emnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = torch.unsqueeze(self.data[index],0), self.target[index]

        return img.float(), target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = EMNIST(root="/mnt/data/th/FedTH/data/emnist", split="balanced", download=True, train=True, 
                transform=tt.Compose([
                    lambda img: tt.functional.rotate(img, -90),
                    lambda img: tt.functional.hflip(img),
                    tt.ToTensor()
                ]))
    train_ds = EMNIST_truncated( dataidxs=None, train=True,dataset= dataset)