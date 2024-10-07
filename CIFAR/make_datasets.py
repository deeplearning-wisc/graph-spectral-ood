import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision

import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.lsun_loader as lsun_loader
import utils.svhn_loader as svhn
from utils.tinyimages_80mn_loader import RandomImages
from utils.imagenet_rc_loader import ImageNet

import pathlib
from PIL import ImageFilter
import torchvision.transforms as T
from random import sample

import random
import copy

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

'''
This script makes the datasets used in training. The main function is make_datasets. 
'''


# *** update this before running on your machine ***
cifar10_path = '../data/'
cifar100_path = '../data/'
svhn_path = '../data/'
lsun_c_path = '../data/'
lsun_r_path = '../data/'
isun_path = '../data/'
dtd_path = '../data/'
places_path = '../data/'
tinyimages_300k_path = '../data/'
MNIST_path = '../data/'
FashionMNIST_path = '../data/'
CorMNIST_path = '../data/'
CorCIFAR10_train = '../data/'
CorCIFAR10_test = '../data/'
iNaturalist_path='../data/'
IMAGENET100_train = '../data/'
IMAGENET100_test = '../data/'
CorIMAGENET100_train = '../data/'
CorIMAGENET100_test = '../data/'

class CustomSubset(torch.utils.data.Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        try:
            self.label = dataset.label
        except:
            self.label = dataset.targets

        # self.data = dataset.data
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform is not None:
            x = self.subset_transform(x)
      
        return x, y   
    
    def __len__(self): 
        return len(self.indices)
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SimSiamTransform():
    def __init__(self, image_size=32, mean_std=imagenet_mean_std):
        # image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        # image_size = 32
        p_blur = 0.20 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.2),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        # print(type(x))
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2




def load_tinyimages_300k():
    print('loading TinyImages-300k')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    t = trn.Compose([trn.ToTensor(),
                     trn.ToPILImage(),
                     trn.ToTensor(),
                     trn.Normalize(mean, std)])

    data = RandomImages(root=tinyimages_300k_path, transform=t)

    return data




def load_CIFAR(dataset, classes=[]):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset in ['cifar10']:
        print('loading CIFAR-10')
        train_data = dset.CIFAR10(
            cifar10_path, train=True, transform=None, download=True)
        test_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform, download=True)

    elif dataset in ['cifar100']:
        print('loading CIFAR-100')
        train_data = dset.CIFAR100(
            cifar100_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform, download=True)

    return train_data, test_data

def load_imagenet():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # t = trn.Compose([trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(), normalize])
    test_transform = trn.Compose([trn.Resize([224, 224]), trn.ToTensor(), trn.Normalize(mean, std)])

    train_data = dset.ImageFolder(IMAGENET100_train, transform=None)
    test_data = dset.ImageFolder(IMAGENET100_test, transform=test_transform)

    return train_data, test_data

def load_corimagenet():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # t = trn.Compose([trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(), trn.ToTensor(), normalize])
    test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

    train_data = dset.ImageFolder(CorIMAGENET100_train, transform=None)
    test_data = dset.ImageFolder(CorIMAGENET100_test, transform=test_transform)

    return train_data, test_data


def load_MNIST():

    mean = [0.1307,]
    std = [0.3081,]    

    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    print('loading MNIST')
    train_data = dset.MNIST(MNIST_path, train=True, download=True, transform=train_transform)
    test_data = dset.MNIST(MNIST_path, train=False, download=True, transform=test_transform)

    return train_data, test_data


def load_FashionMNIST():
    
    mean = [0.1307,]
    std = [0.3081,]    

    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    print('loading FashionMNIST')
    train_data = dset.FashionMNIST(FashionMNIST_path, train=True, download=True, transform=train_transform)
    test_data = dset.FashionMNIST(FashionMNIST_path, train=False, download=True, transform=test_transform)

    return train_data, test_data

def load_CorMNIST():

    print('loading CorMNIST')

    from dataloader.cormnistLoader import CorMNISTDataset as Dataset
    train_data = Dataset('train', CorMNIST_path)
    test_data = Dataset('test', CorMNIST_path)

    return train_data, test_data

def load_CorCifar(dataset, cortype):

    print('loading CorCIFAR-10')

    from dataloader.corcifar10Loader import CorCIFARDataset as Dataset

    train_data = Dataset('train', cortype, CorCIFAR10_train)
    test_data = Dataset('test', cortype, CorCIFAR10_test)

    return train_data, test_data


def load_CorImageNet(dataset, cortype):

    print('loading CorIMAGENET-100')
    from dataloader.corimagenetLoader import CorIMAGENETDataset as Dataset

    train_data = Dataset('train', cortype, CorIMAGENET100_train)
    test_data = Dataset('test', cortype, CorIMAGENET100_test)

    return train_data, test_data

def load_SVHN(include_extra=False):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    print('loading SVHN')
    # trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    t = None
    if not include_extra:
        train_data = svhn.SVHN(root=svhn_path, split="train",
                                transform=None)
    else:
        train_data = svhn.SVHN(root=svhn_path, split="train_and_extra",
                               transform=trn.Compose(
                                   [trn.ToTensor(), trn.Normalize(mean, std)]))

    test_data = svhn.SVHN(root=svhn_path, split="test",
                              transform=trn.Compose(
                                  [trn.ToTensor(), trn.Normalize(mean, std)]))

    train_data.targets = train_data.targets.astype('int64')
    test_data.targets = test_data.targets.astype('int64')
    return train_data, test_data



def load_dataset(dataset):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    t = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)])
    if dataset == 'lsun_c':
        print('loading LSUN_C')
        out_data = dset.ImageFolder(root=lsun_c_path, transform=None)

    if dataset == 'lsun_r':
        print('loading LSUN_R')
        out_data = dset.ImageFolder(root=lsun_r_path, transform=None)

    if dataset == 'isun':
        print('loading iSUN')
        out_data = dset.ImageFolder(root=isun_path, transform=None)
    if dataset == 'dtd':
        print('loading DTD')
        out_data = dset.ImageFolder(root=dtd_path, transform=None)
    if dataset == 'places':
        print('loading Places365')
        out_data = dset.ImageFolder(root=places_path, transform=None)

    if dataset == 'iNaturalist':
        print('loading iNaturalist')
        out_data = dset.ImageFolder(root=iNaturalist_path, transform=None)

    return out_data

def load_in_data(in_dset, rng):

    train_data_in_orig, test_in_data = load_CIFAR(in_dset)

    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    train_len = int(0.5 * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]

    train_in_data = CustomSubset(train_data_in_orig, train_idx)
    aux_in_data = CustomSubset(train_data_in_orig, aux_idx)

    return train_in_data, aux_in_data, test_in_data


def load_in_MNIST_data(in_dset, rng):

    train_data_in_orig, test_in_data = load_MNIST()

    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    train_len = int(0.5 * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]

    train_in_data = CustomSubset(train_data_in_orig, train_idx)
    aux_in_data = CustomSubset(train_data_in_orig, aux_idx)

    return train_in_data, aux_in_data, test_in_data


def load_in_mixMNIST_data(in_dset, rng, alpha):
    train_data_in_orig_mnist, test_in_data_mnist = load_MNIST()
    train_data_in_orig_cor, test_in_data_cor = load_CorMNIST()

    test_datasets = [test_in_data_mnist, test_in_data_cor]
    test_in_data = torch.utils.data.ConcatDataset(test_datasets)

    idx = np.array(range(len(train_data_in_orig_mnist)))
    rng.shuffle(idx)
    
    train_len = int(alpha * len(train_data_in_orig_mnist))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]

    train_in_data = CustomSubset(train_data_in_orig_mnist, train_idx)
    aux_in_data_mnist = CustomSubset(train_data_in_orig_mnist, aux_idx)

    aux_in_data_cor = train_data_in_orig_cor
    aux_in_data = aux_in_data_mnist

    return train_in_data, aux_in_data, aux_in_data_cor, test_in_data_mnist, test_in_data_cor


def load_in_mixCifar_data(in_dset, rng, alpha, cortype):

    train_data_in_orig_cifar, test_in_data_cifar = load_CIFAR(in_dset)
    aux_data_cor_orig, test_data_cor = load_CorCifar(in_dset, cortype)
    
    # train_data_in_orig_cifar, test_in_data_cifar = load_imagenet()
    # aux_data_cor_orig, test_data_cor = load_CorImageNet(in_dset, cortype)

    idx = np.array(range(len(train_data_in_orig_cifar)))
    rng.shuffle(idx)

    train_len = int(alpha * len(train_data_in_orig_cifar))
    train_idx = idx[:train_len]
    # train_idx = idx[:]

    aux_idx = idx[int(0.5 * len(train_data_in_orig_cifar)):]
    # aux_idx = idx[:]
    train_in_data = CustomSubset(train_data_in_orig_cifar, train_idx)

    aux_in_data = CustomSubset(train_data_in_orig_cifar, aux_idx)

    idx_cor = np.array(range(len(aux_data_cor_orig)))
    
    rng.shuffle(idx)
    train_len_cor = int(alpha * len(aux_data_cor_orig))
    train_idx_cor = idx_cor[:train_len_cor]
    aux_idx_cor = idx[int(0.5*len(aux_data_cor_orig)):]

    train_in_cor_data = CustomSubset(aux_data_cor_orig, train_idx_cor)
    aux_in_cor_data = CustomSubset(aux_data_cor_orig, aux_idx_cor)

    return train_in_data, aux_in_data, aux_data_cor_orig, test_in_data_cifar, test_data_cor

def load_out_data(aux_out_dset, test_out_dset, in_dset, rng, classes=[]):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    t = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)])
     
    if test_out_dset == 'lsun_c':
        t = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)])
    elif test_out_dset == 'lsun_r' or test_out_dset == 'isun':
        t = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]) 
    elif test_out_dset == 'dtd' or test_out_dset == 'places':
        t = trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]) 
    elif test_out_dset == 'iNaturalist':
        t = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])


    if aux_out_dset == test_out_dset:
        if aux_out_dset == 'tinyimages_300k':
            out_data = load_tinyimages_300k(in_dset)

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.99 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = CustomSubset(out_data, aux_subset_idxs)
            test_out_data = CustomSubset(out_data, test_subset_idxs)

        elif aux_out_dset == 'FashionMNIST':
        
            out_data, _ = load_FashionMNIST()

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.7 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = CustomSubset(out_data, aux_subset_idxs)
            test_out_data = CustomSubset(out_data, test_subset_idxs)

        elif aux_out_dset == 'svhn':

            aux_out_data, test_out_data = load_SVHN()

        else:
            out_data = load_dataset(aux_out_dset)

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.7 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = CustomSubset(out_data, aux_subset_idxs)
            test_out_data = CustomSubset(out_data, test_subset_idxs, subset_transform=t)

    elif aux_out_dset != test_out_dset:
        # load aux data
        if aux_out_dset == 'tinyimages_300k':
            aux_out_data = load_tinyimages_300k()
        elif aux_out_dset == 'svhn':
            aux_out_data, _ = load_SVHN()
        elif aux_out_dset == 'FashionMNIST':
            aux_out_dset, _ == load_FashionMNIST()
        elif aux_out_dset in ['cifar10', 'cifar100']:
            aux_out_data, _ = load_CIFAR(aux_out_dset)
        else:
            aux_out_data = load_dataset(aux_out_dset)

        # load test data
        if test_out_dset == 'svhn':
            _, test_out_data = load_SVHN()
        elif test_out_dset == 'FashionMNIST':
            _, test_out_data = load_FashionMNIST()
        elif test_out_dset in ['cifar10', 'cifar100']:
            _, test_out_data = load_CIFAR(test_out_dset)
        else:
            test_out_data = load_dataset(test_out_dset)

    return aux_out_data, test_out_data





def train_valid_split(test_in_data, test_in_data_cor, aux_in_data, aux_in_data_cor, aux_out_data, rng, pi_1, pi_2):
    

    '''
    Args:
        in_data: data from in-distribution, from test set
        aux_in_data: data from in-distribution component of mixture, not in test set
        aux_out_data: data from auxiliary dataset component of mixture, not in test set

    Returns:
        7 datasets: each dataset split into two, one for training (or testing) and the other for validation
    '''

    aux_in_valid_size_full = int(0.3 * len(aux_in_data))

    valid_in_size = int(0.1 * len(aux_in_data))

    idx_in = np.array(range(len(aux_in_data)))
    rng.shuffle(idx_in)
    
    train_aux_in_idx = idx_in[aux_in_valid_size_full + valid_in_size:]
    valid_in_idx = idx_in[aux_in_valid_size_full: aux_in_valid_size_full + valid_in_size]

    train_aux_in_data_final = CustomSubset(aux_in_data, train_aux_in_idx)
    valid_in_data_final = CustomSubset(aux_in_data, valid_in_idx)
    
    aux_in_cor_valid_size_full = int(0.3 * len(aux_in_data_cor))

    valid_cor_size = int(0.1 * len(aux_in_data_cor))

    idx_cor = np.array(range(len(aux_in_data_cor)))
    rng.shuffle(idx_cor)
    
    train_aux_in_cor_idx = idx_cor[aux_in_cor_valid_size_full + valid_cor_size:]
    train_cor_clean_idx = idx_cor[aux_in_cor_valid_size_full: aux_in_cor_valid_size_full + valid_cor_size]
    

    train_aux_in_data_cor_final = CustomSubset(aux_in_data_cor, train_aux_in_cor_idx)


    train_aux_in_data_mix_list = [train_aux_in_data_final, train_aux_in_data_cor_final]
    train_aux_in_data_mix_final = torch.utils.data.ConcatDataset(train_aux_in_data_mix_list)


    train_data_cor_clean = CustomSubset(aux_in_data_cor, train_cor_clean_idx)


    #create validation dataset for auxiliary dataset componenet of mixture
    
    aux_out_valid_size_full = int(0.3 * len(aux_out_data))

    idx_out = np.array(range(len(aux_out_data)))
    rng.shuffle(idx_out)
    train_aux_out_idx = idx_out[aux_out_valid_size_full:]
    

    train_aux_out_data_final = CustomSubset(aux_out_data, train_aux_out_idx)
    

    valid_total = min(aux_in_valid_size_full, aux_in_cor_valid_size_full, 10*aux_out_valid_size_full)
    

    valid_aux_in_idx = idx_in[:int((1-pi_1-pi_2) * valid_total)]
    valid_aux_in_data_final = CustomSubset(aux_in_data, valid_aux_in_idx)


    valid_aux_in_cor_idx = idx_cor[:int(pi_1 * valid_total)]

    valid_aux_in_data_cor_final = CustomSubset(aux_in_data_cor, valid_aux_in_cor_idx)
    
    valid_aux_out_idx = idx_out[:int(pi_2 * valid_total)]
    valid_aux_out_data_final = CustomSubset(aux_out_data, valid_aux_out_idx)


    valid_aux_data_total = [valid_aux_in_data_final, valid_aux_in_data_cor_final, valid_aux_out_data_final]
    valid_aux_data_final = torch.utils.data.ConcatDataset(valid_aux_data_total)


    return test_in_data, test_in_data_cor, valid_in_data_final, valid_aux_data_final, train_aux_in_data_final, train_aux_in_data_cor_final, train_aux_out_data_final, valid_aux_in_data_final, valid_aux_in_data_cor_final, valid_aux_out_data_final


def make_datasets(in_dset, aux_out_dset, test_out_dset, state, alpha, pi_1, pi_2, cortype):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    if test_out_dset != 'iNaturalist':
        transform = trn.Compose([trn.Resize([32,32]), trn.ToTensor(), trn.Normalize(mean, std)])
    else: 
        transform = trn.Compose([trn.Resize([224, 224]), trn.ToTensor(), trn.Normalize(mean, std)])
    
    if test_out_dset == 'lsun_c' or test_out_dset=='svhn' or test_out_dset=='isun':
        transform_out = trn.Compose([trn.Resize([32, 32]), trn.ToTensor(), trn.Normalize(mean, std)])
    elif test_out_dset == 'lsun_r':
        transform_out = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)])
    elif test_out_dset == 'dtd' or test_out_dset == 'places':
        transform_out = trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]) 
    elif test_out_dset == 'iNaturalist':
        transform_out = trn.Compose([trn.Resize([224, 224]), trn.ToTensor(), trn.Normalize(mean, std)])
        
    # random seed
    rng = np.random.default_rng(state['seed'])

    print('building datasets...')
    train_in_data, aux_in_data, aux_in_data_cor, test_in_data, test_in_data_cor = load_in_mixCifar_data(in_dset, rng, alpha, cortype)
    aux_out_data, test_out_data = load_out_data(aux_out_dset, test_out_dset, in_dset, rng)

    # make validation set from CIFAR test set
    test_in_data, test_in_data_cor, valid_in_data_final, valid_aux_data_final, train_aux_in_data_final, train_aux_in_data_cor_final, \
    train_aux_out_data_final, valid_aux_in_data_final, valid_aux_in_data_cor_final, valid_aux_out_data_final = train_valid_split(
                                                                                                            test_in_data, test_in_data_cor, aux_in_data, 
                                                                                                            aux_in_data_cor, aux_out_data, rng, pi_1, pi_2)

    
    # train_aux_in_data_final = torch.utils.data.ConcatDataset([train_in_data, train_aux_in_data_final])
    
    # print(train_aux_in_data_final.subset_transform)
    # print(train_aux_in_data_cor_final.subset_transform)
    # print(train_aux_out_data_final.subset_transform)
    # print(train_in_data.subset_transform)
    # train_in_data.subset_transform = transform

    valid_in_data_final.subset_transform = transform
    # test_in_data.subset_transform = transform
    # test_in_data_cor.subset_transform = transform
    # test_out_data.subset_transform = transform
    valid_in_data_final.subset_transform = transform
    valid_aux_data_final.subset_transform = transform
    
    valid_aux_in_data_final.subset_transform = transform
    valid_aux_in_data_cor_final.subset_transform = transform
    valid_aux_out_data_final.subset_transform = transform_out
    
    '''
    train_aux_in_data_final.subset_transform = transform
    train_aux_in_data_cor_final.subset_transform = transform
    train_aux_out_data_final.subset_transform = transform_out
    train_in_data.subset_transform = transform
    '''
    train_aux_in_data_final.subset_transform = SimSiamTransform(image_size=32)
    train_aux_in_data_cor_final.subset_transform = SimSiamTransform(image_size=32)
    train_aux_out_data_final.subset_transform = SimSiamTransform(image_size=32)
    train_in_data.subset_transform = SimSiamTransform(image_size=32)  

    train_in_data_noaug = copy.deepcopy(train_in_data)
    train_in_data_noaug.subset_transform = transform 


    # state['prefetch'] = 4
    # create the dataloaders
    train_loader_in = torch.utils.data.DataLoader(
        train_in_data,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True)
    
    train_loader_in_noaug = torch.utils.data.DataLoader(
        train_in_data_noaug,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True)

    # validation for P_0
    valid_loader_in = torch.utils.data.DataLoader(
        valid_in_data_final,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # auxiliary dataset
    # for in-distribution component of mixture
    # drop last batch to eliminate unequal batch size issues
    train_loader_aux_in = torch.utils.data.DataLoader(
        #train_aux_in_data_mix_final,
        train_aux_in_data_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    #for in-distribution cor component of mixture
    #drop last batch to eliminate unequal batch size issues
    train_loader_aux_in_cor = torch.utils.data.DataLoader(
        train_aux_in_data_cor_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)


    #for out-distribution component of mixture
    train_loader_aux_out = torch.utils.data.DataLoader(
        train_aux_out_data_final,
        batch_size=state['batch_size'], shuffle=True,
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    valid_loader_aux = torch.utils.data.DataLoader(
         valid_aux_data_final,
         batch_size=state['batch_size'], shuffle=False,
         num_workers=state['prefetch'], pin_memory=True, drop_last=False)

    valid_loader_aux_in = torch.utils.data.DataLoader(
         valid_aux_in_data_final,
         batch_size=state['batch_size'], shuffle=False,
         num_workers=state['prefetch'], pin_memory=True, drop_last=False)
    
    valid_loader_aux_cor = torch.utils.data.DataLoader(
         valid_aux_in_data_cor_final,
         batch_size=state['batch_size'], shuffle=False,
         num_workers=state['prefetch'], pin_memory=True, drop_last=False)

    valid_loader_aux_out = torch.utils.data.DataLoader(
         valid_aux_out_data_final,
         batch_size=state['batch_size'], shuffle=False,
         num_workers=state['prefetch'], pin_memory=True, drop_last=False)

    test_loader_in = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    test_loader_cor = torch.utils.data.DataLoader(
        test_in_data_cor,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # test loader for ood
    test_loader_out = torch.utils.data.DataLoader(
        test_out_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    return train_loader_in, train_loader_in_noaug, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out, test_loader_in, test_loader_cor, test_loader_out, valid_loader_in, valid_loader_aux, \
           valid_loader_aux_in, valid_loader_aux_cor, valid_loader_aux_out


def make_test_dataset(in_data, test_out_dset, state):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    # load in-distribution data
    if in_data == 'cifar10':
        test_in_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform)
    elif in_data == 'cifar100':
        test_in_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform)
    elif in_data == 'MNIST':
        test_in_data = dset.MNIST(
            MNIST_path, train=False, transform=trn.Compose([trn.ToTensor(), trn.Normalize((0.1307,), (0.3081,))]))


    #load out-distribution
    if test_out_dset == 'svhn':
        test_out_data = svhn.SVHN(root=svhn_path, split="test",
                                  transform=trn.Compose(
                                      [  # trn.Resize(32),
                                          trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
    if test_out_dset == 'FashionMNIST':
        test_out_data = dset.ImageFolder(root=FashionMNIST_path, split="test",
                                  transform=trn.Compose(
                                      [  # trn.Resize(32),
                                          trn.ToTensor(), trn.Normalize((0.1307,), (0.3081,))]), download=True)
    if test_out_dset == 'lsun_c':
        test_out_data = dset.ImageFolder(root=lsun_c_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std), trn.RandomCrop(32, padding=4)]))

    if test_out_dset == 'lsun_r':
        test_out_data = dset.ImageFolder(root=lsun_r_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'isun':
        test_out_data = dset.ImageFolder(root=isun_path,
                                         transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'dtd':
        test_out_data = dset.ImageFolder(root=dtd_path,
                                         transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))

    if test_out_dset == 'places':
        test_out_data = dset.ImageFolder(root=places_path,
                                         transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                                trn.ToTensor(), trn.Normalize(mean, std)]))

    #test data for P_0
    test_loader = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    # test loader for ood
    test_loader_ood = torch.utils.data.DataLoader(
        test_out_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    return test_loader, test_loader_ood
