import numpy as np
import torch
import torch.utils.data as data
import torchvision

from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample
import random

from PIL import ImageFilter
from PIL import Image

class CorIMAGENETDataset(data.Dataset):
    def __init__(self, set_name, cortype, data_path):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        # t = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

        if set_name != 'test':
            self.transform = None
        else:
            self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        
        '''
        self._image_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        '''

        self.set_name = set_name

        images = np.load(os.path.join(data_path, cortype + '.npy'))
        labels = np.load(os.path.join(data_path, 'labels.npy'))

        self.data = images
        self.label = labels

        self.num_class = 100

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            # img = self._image_transformer(img)

        return img, label

    def __len__(self):
        return len(self.data)










