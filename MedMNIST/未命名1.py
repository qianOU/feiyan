# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:11:07 2020

@author: 28659
"""


import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

FLAG = "pneumoniamnist"
DATAROOT = 'D:\\作业\\非参数统计\\期末\\MedMNIST'

#建立模型存放位置
DIR_PATH = './%s_model_store' % (FLAG)
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)

#数据集基本信息
INFO = {
    "pneumoniamnist": {
        "description": "PneumoniaMNIST: A dataset based on a prior dataset of 5,856 pediatric chest X-ray images. The task is binary-class classification of pneumonia and normal. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-2,916) x (127-2,713). We center-crop the images and resize them into 1 x 28 x 28.",
        "url": "...",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": .81,
            "val": .1,
            "test": .1
        }
    }
        }

# In[get special kind sets]

DATASETS =   np.load(os.path.join(DATAROOT, "{}.npz".format(FLAG)))
LABEL = 1

def split_label(npz_file, label=[0, 1]):
    
    from sklearn.model_selection import train_test_split

    """
    Returns
    -------
    对总数据选择给定的类别数据
    """
    #读入数据划分的比例
    ratio = INFO[FLAG]['n_samples']
    
    types = ['train', 'val', 'test']
    x_all = ['%s_images' % i for i in types]
    y_all = ['%s_labels' % i for i in types]
    x = np.concatenate([npz_file[i] for i in x_all], axis=0)
    y = np.concatenate([npz_file[i] for i in y_all], axis=0)
    
    if isinstance(label, list):
        condition = [True]*len(y)
    else:
        condition = (y == label).reshape(-1)

    x_wanted = x[condition]
    y_wanted = y[condition].reshape(-1)
    
    npz_file = {}
    
    #划分数据集
    obj_counts, h, w = x_wanted.shape
    x_wanted = x_wanted.reshape(obj_counts, -1)
    
    #1 划分trian , test
    xtrain, xtest, ytrain, ytest = train_test_split(x_wanted, y_wanted, test_size=ratio['test'], random_state=0)
    npz_file['test_images'] = xtest.reshape(len(xtest), h, w)
    npz_file['test_labels']  = ytest
    #2. 划分 train, val
    xtrain, xval, ytrian, yval = train_test_split(xtrain, ytrain, test_size=ratio['val'], random_state=0)
    npz_file['train_images'] = xtrain.reshape(len(xtrain), h, w)
    npz_file['train_labels']  = ytrain
    npz_file['val_images'] = xval.reshape(len(xval), h, w)
    npz_file['val_labels']  = yval
    print('label=%s, the size of set: %d\ntrain size: %s, val size:%s, test size:%s' % (label, len(x_wanted), 
                                                                                        len(xtrain), len(xval), len(xtest)))
    return npz_file

DATASETS = split_label(DATASETS, label=LABEL)

# In[Preparing data]
#数据加载器，
class PneumoniaMNIST(Dataset):

    FLAG = ...

    def __init__(self, split='train', transform=None, target_transform=None):
        ''' dataset
        :param split: 'train', 'val' or 'test', select dataset
        :param transform: data transformation
        :param target_transform: target transformation
        :lable : need split type of labels
        '''

        self.npz_file = DATASETS

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = self.npz_file['train_images']
            self.label = self.npz_file['train_labels']
        elif self.split == 'val':
            self.img = self.npz_file['val_images']
            self.label = self.npz_file['val_labels']
        elif self.split == 'test':
            self.img = self.npz_file['test_images']
            self.label = self.npz_file['test_labels']


    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]





task = INFO[FLAG]['task']
n_channels = INFO[FLAG]['n_channels']
n_classes = len(INFO[FLAG]['label'])

batch_size =200



print('==> Preparing data..')
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]) #每个c 数据标准化
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = PneumoniaMNIST(split='train', transform=train_transform)
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PneumoniaMNIST(split='val', transform=val_transform)
val_loader = data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = PneumoniaMNIST(split='test', transform=test_transform)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True)