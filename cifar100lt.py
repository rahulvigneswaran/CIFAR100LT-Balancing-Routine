import os
import random
from re import M
from PIL import Image
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
# import pytorch_lightning as pl

# custom
# from configs.classification.defaults import update_config
# from utils import ClassAwareSampler, DistributedSamplerWrapper

__all__ = ['cifar100lt']

AUG = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
    ]


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, phase, imbalance_ratio):
        train = True if phase == "train" else False
        root = "./"
        super(IMBALANCECIFAR100, self).__init__(root, train, transform=None, target_transform=None, download=True) #FIXME Throws error when download=False
        self.train = train
        self.phase = phase
        self.imb_ratio = self.imb = imbalance_ratio
        self.subset_root = f"datasets/CIFAR100_LT"

        self.prepare_data()

        if self.train:
            self.gen_imbalanced_data()
            self.transform = transforms.Compose(AUG)
        else:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
            if self.phase == "val":
                self.data, self.targets = torch.load(f"{self.subset_root}/val.pth")
            else:
                self.data, self.targets = torch.load(f"{self.subset_root}/test.pth")
        
        self.labels = self.targets
        self.img_num_list = list(np.unique(self.labels, return_counts=True)[1])

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def gen_imbalanced_data(self,):            
        self.data, self.targets = torch.load(f"{self.subset_root}/{self.phase}_imb_{int(1/self.imb_ratio)}.pth")
        
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def prepare_data(self):
        sub_root = f"datasets/CIFAR100_LT"
        links = ["https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/val.pth", "https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/test.pth"]
        md5_checksum = ["5ac4dbf8ecd58ce105b0203fc3f8390e", "7a21cb3e2b27a00d43ab06f760942a64"]
        
        if self.imb == 1/1:
            links.append("https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/train_imb_1.pth")
            md5_checksum.append("312485967a8ea6daa7456489b714d41e")
        elif self.imb == 1/200:
            links.append("https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/train_imb_200.pth")
            md5_checksum.append("596ec1798c132b5bc097275eed9f10cd")
        elif self.imb == 1/100:
            links.append("https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/train_imb_100.pth")
            md5_checksum.append("67dc6f7a622d0a2454c63af039b51d15")
        elif self.imb == 1/50:
            links.append("https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/train_imb_50.pth")
            md5_checksum.append("d76572ab14ff83052480953f636c329e")
        elif self.imb == 1/10:
            links.append("https://github.com/rahulvigneswaran/My_Custom_Datasets/raw/main/CIFAR100_LT/train_imb_10.pth")
            md5_checksum.append("4f94a2c1cbcb2823397be90fc7c251e0")
        else:
            assert False, f"Wrong Imbalance factor {self.imb}. Choose from [1.0, 0.005, 0.01, 0.02, 0.1]"

        for url, md5 in zip(links, md5_checksum):
            torchvision.datasets.utils.download_url(url=url, md5=md5, root=sub_root)