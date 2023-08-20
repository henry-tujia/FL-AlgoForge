import logging

import numpy as np
from PIL import Image
from torchvision.datasets import DatasetFolder
import torch.utils.data as data
import six
import lmdb
import os
import pickle

#logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None,dataidxs=None):
        self.db_path = db_path+".lmdb"
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = dict(loads_data(txn.get(b'__keys__')))
        self.dataidxs = dataidxs
        self.get_imgs()
        self.targets = list(self.keys.values())
        self.transform = transform
        self.target_transform = target_transform
        
    def get_imgs(self):
        if self.dataidxs is not None:
            self.imgs = self.dataidxs
        else:
            self.imgs = np.array(list(self.keys.keys()),dtype=int)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            idx = self.imgs[index]
            byteflow = txn.get(u'{}'.format(idx).encode('ascii'))

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        # im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    # def __get_labels__(self):