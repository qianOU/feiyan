# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:11:07 2020

@author: 28659
"""

from settings import *

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
import PIL
import pandas as pd


# In[get special kind sets]



def split_label(npz_file, label=[0, 1], gan=True, gan_model=None):
    
    from sklearn.model_selection import train_test_split
    from skimage import transform

    """
    transform:只对需要在数据集中生成伪造数据的部分进行转换为64*64的
    gan=True :控制是否训练gan
    gan_model:利用gan生成数据，为了判别模型的建立
    gan_model:如果传入list，表示完全用两个生成模型进行分别生成两种类别的，顺序为[0， 1]
    Returns
    -------
    对总数据选择给定的类别数据
    """
    #读入数据划分的比例
    ratio = INFO[FLAG]['n_samples']
    
    types = ['train', 'val', 'test']
    x_all = ['%s_images' % i for i in types]
    y_all = ['%s_labels' % i for i in types]
     
    #处理生成两种类型数据的情况
    if hasattr(gan_model, '__len__'):
        # gan_0, gan_1 = gan_model
        num_label = pd.Series(np.concatenate([npz_file[i] for i in y_all], axis=0).flatten()).value_counts()
        npz_file = dict.fromkeys(x_all+y_all)
        temp_x_list = []
        temp_y_list = []
        for label_i, gen in zip(range(2), gan_model):
            counts = num_label[label_i]
            inp = np.random.randn(counts, latent, 1, 1) #100是生成网络的输入维度
            inp = torch.tensor(inp).type(torch.float).cuda()
            
            with torch.no_grad():
                step = inp.shape[0]//50
                split_list = []
                for i in range(step+1):
                    start = i*50
                    ends = (i+1)*50 if i<step else inp.shape[0]
                    x_gan_1 = inp[start:ends,...]
                    x_gan_1 = gen(x_gan_1).cpu().squeeze().numpy() #[length,  64, 64]
                    split_list.append(x_gan_1)
                inp = np.concatenate(split_list, axis=0)
                temp_x_list.append(inp)
            
            # with torch.no_grad():
            #     # temp_x_list.append(gen(inp).cpu().numpy().reshape(counts, 28, 28))
            #     temp_x_list.append(gen(inp).cpu().squeeze().numpy()) #[counts, 64, 64]
            yy = np.zeros(counts)
            yy.fill(label_i)
            temp_y_list.append(yy.astype(np.int))
        
        x_wanted = np.concatenate(temp_x_list, axis=0)
        y_wanted = np.concatenate(temp_y_list, axis=0)
    
    #处理只需要生成一种类型数据的情况
    else:
        x = np.concatenate([npz_file[i] for i in x_all], axis=0)
        y = np.concatenate([npz_file[i] for i in y_all], axis=0)

        #用于判别生成图片与原图片的情况
        if not gan and (gan_model is not None):
            x = x[(y==label).flatten(), ...]
            x = np.stack([PIL.Image.fromarray(i.astype(np.uint8)).resize([64,64],  resample=2) for i in x], 0) #Image.BILINEAR (2), 将原始28*28的图片转化为64*64
            x = np.stack([i/(i.max()+1) for i in  x], 0) #每一张图片都处理成0-1
            x = transforms.Normalize(mean=[0.5], std=[0.5])(torch.tensor(x)).numpy() #规范化处理0-1
            length = x.shape[0]
            print('label=%s, %s 生成 %d个数据'%(label, type(gan_model).__name__, length))
            x_gan = np.random.randn(length, latent, 1, 1) #100 表示的是生成网络的输入维度
            x_gan = torch.tensor(x_gan).type(torch.float).cuda()
            with torch.no_grad():
                step = x_gan.shape[0]//50
                split_list = []
                for i in range(step+1):
                    start = i*50
                    ends = (i+1)*50 if i<step else x_gan.shape[0]
                    x_gan_1 = x_gan[start:ends,...]
                    x_gan_1 = gan_model(x_gan_1).cpu().squeeze().numpy() #[length,  64, 64]
                    split_list.append(x_gan_1)
                x_gan = np.concatenate(split_list, axis=0)
            y_gan = np.zeros((length, 1)).astype(np.int) #将生成的标签标注为0
            x_wanted = np.concatenate([x, x_gan], axis=0)
            y_wanted = np.concatenate([np.ones_like(y_gan).astype(np.int), y_gan], axis=0) 
        
        #对于用来训练生成模型的数据
        if gan:
            if isinstance(label, list):
                condition = [True]*len(y)
            else:
                condition = (y == label).reshape(-1)
        
            x_wanted = x[condition]
            y_wanted = y[condition].reshape(-1)
            
        if gan_model is None and not gan:
            x_wanted = x
            y_wanted = y
        
        npz_file = {}
        
        if  gan:
            npz_file['train_images'] = x_wanted
            npz_file['train_labels']  = y_wanted
            return npz_file
        
    #划分数据集
    obj_counts, h, w = x_wanted.shape
    x_wanted = x_wanted.reshape(obj_counts, -1) #train_test_spilt 只能处理二维数据
    
    #1 划分trian , test
    xtrain, xtest, ytrain, ytest = train_test_split(x_wanted, y_wanted, test_size=ratio['test'])
    npz_file['test_images'] = xtest.reshape(len(xtest), h, w)
    npz_file['test_labels']  = ytest
    #2. 划分 train, val
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=ratio['val'])
    npz_file['train_images'] = xtrain.reshape(len(xtrain), h, w)
    npz_file['train_labels']  = ytrain
    npz_file['val_images'] = xval.reshape(len(xval), h, w)
    npz_file['val_labels']  = yval
    print("""label=%s, the size of set: %d, \ntrain size: %s label 0/1=%s
          \rval size:%s label 0/1=%s
          \rtest size:%s label 0/1=%s""" % (label, len(x_wanted), 
                             len(xtrain), list(pd.Series(ytrain.flatten()).value_counts().sort_index().values), 
                              len(xval), list(pd.Series(yval.flatten()).value_counts().sort_index().values),
                              len(xtest), list(pd.Series(ytest.flatten()).value_counts().sort_index().values)))
    return npz_file



# In[Preparing data]
#数据加载器，
class PneumoniaMNIST(Dataset):

    FLAG = ...

    def __init__(self, DATASETS, split='train', transform=None, target_transform=None):
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
        
        #将伪造的图片转换为tensor形式
        if self.transform is None:
            for key, value in self.npz_file.items():
                if key.endswith('images'):
                    self.npz_file[key] = torch.FloatTensor(self.npz_file[key]).view(-1, 1, 64, 64)
                
        
        if self.split == 'train':
            self.img = self.npz_file['train_images']
            self.label = self.npz_file['train_labels']
        elif self.split == 'val':
            self.img = self.npz_file['val_images']
            self.label = self.npz_file['val_labels']
        elif self.split == 'test':
            self.img = self.npz_file['test_images']
            self.label = self.npz_file['test_labels']
        #归一化
        # self.img = (self.img/255.0)*2-1


    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        if not isinstance(img,torch.Tensor):
            img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

if __name__ == '__main__':
    path = 'D:\\作业\\非参数统计\\期末\\MedMNIST\\pneumoniamnist.npz'
    DATASETS =np.load(path)
    
    npz_file = split_label(DATASETS, label=1, gan=False, gan_model=None)
    
    #查看原始数据构成
    # DATASETS =np.load(path)
