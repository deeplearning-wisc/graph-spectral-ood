import numpy as np
import torch
import torch.utils.data as data
import torchvision

from torchvision import datasets
import torch
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

import os

from random import sample, random
from PIL import ImageFilter

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]




class CorCIFARDataset(data.Dataset):
    def __init__(self, set_name, cortype, data_path):

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        if set_name != 'test':
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        images = np.load(os.path.join(data_path, cortype + '.npy'))
        labels = np.load(os.path.join(data_path, 'labels.npy'))

        self.data = images
        self.label = labels
        self.set_name = set_name

        # self.data = self.data.reshape(5, 10000, self.data.shape[1], self.data.shape[2], self.data.shape[3])
        # self.label = self.label.reshape(5, 10000)

        # train_test_split = 0.7; cutoff_train_test = int(train_test_split*self.data.shape[1])
        # if set_name=='train': # train set in CIFAR-10-C
        #     self.label = self.label[:, :cutoff_train_test]
        #     self.data = self.data[:, :cutoff_train_test]
        # else:
        #     self.label = self.label[:, cutoff_train_test:]
        #     self.data = self.data[:, cutoff_train_test:]

        # self.data = self.data.reshape(-1, 32, 32, 3)
        # self.label = self.label.reshape(-1)

        self.num_class = 10

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)










