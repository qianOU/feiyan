from medmnist import environ
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

INFO = "medmnist/medmnist.json"


class MedMNIST(Dataset):

    flag = ...

    def __init__(self, npz_file, split='train', transform=None, target_transform=None):
        ''' dataset
        :param split: 'train', 'val' or 'test', select dataset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        # npz_file = np.load(os.path.join(environ.dataroot,"{}.npz".format(self.flag)))

        # types = ['train', 'val', 'test']
        # x_all = ['%s_images' % i for i in types]
        # y_all = ['%s_labels' % i for i in types]
        # x_wanted = np.concatenate([npz_file[i] for i in x_all], axis=0)
        # y_wanted = np.concatenate([npz_file[i] for i in y_all], axis=0)
        
        # obj_counts, h, w = x_wanted.shape
        # x_wanted = x_wanted.reshape(obj_counts, -1)
        
        # npz_file = {}
        # #1 划分trian , test
        # xtrain, xtest, ytrain, ytest = train_test_split(x_wanted, y_wanted, test_size=.1)
        # npz_file['test_images'] = xtest.reshape(len(xtest), h, w)
        # npz_file['test_labels']  = ytest
        # #2. 划分 train, val
        # xtrain, xval, ytrian, yval = train_test_split(xtrain, ytrain, test_size=.1)
        # npz_file['train_images'] = xtrain.reshape(len(xtrain), h, w)
        # npz_file['train_labels']  = ytrain
        # npz_file['val_images'] = xval.reshape(len(xval), h, w)
        # npz_file['val_labels']  = yval
        # print("""label=%s, the size of set: %d, \ntrain size: %s label 0/1=%s
        #       \rval size:%s label 0/1=%s
        #       \rtest size:%s label 0/1=%s""" % (1, len(x_wanted), 
        #                          len(xtrain), list(pd.Series(ytrain.flatten()).value_counts().sort_index().values), 
        #                           len(xval), list(pd.Series(yval.flatten()).value_counts().sort_index().values),
        #                           len(xtest), list(pd.Series(ytest.flatten()).value_counts().sort_index().values)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

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


class PathMNIST(MedMNIST):
    flag = "pathmnist"


class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNIST_Axial(MedMNIST):
    flag = "organmnist_axial"


class OrganMNIST_Coronal(MedMNIST):
    flag = "organmnist_coronal"


class OrganMNIST_Sagittal(MedMNIST):
    flag = "organmnist_sagittal"
