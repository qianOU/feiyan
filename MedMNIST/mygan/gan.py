from settings import *
from dataloader import *

import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from visdom import Visdom
from datetime import datetime


cuda = True if torch.cuda.is_available() else False


# In[定义 generator and discrimnator]
class Generator_gan(nn.Module):
    def __init__(self, latent_dim = latent, img_shape = (1, 28, 28)):
        super(Generator_gan, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.5))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape = (1, 28, 28)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
# In[split dataset to train val test]
if __name__ == "__main__":
    batch_size =200
    # LABEL = 1 #选择使用的label
    for LABEL in [0]:
        path_ = '%s/gan/label_%s' %(DIR_PATH, LABEL) if isinstance(LABEL, int) else '%s/gan/label_%s' %(DIR_PATH, 'all')
        
        
        if not os.path.exists(path_):
            if not os.path.exists(path_.rsplit('label_')[0]):
                os.mkdir(path_.rsplit('label_')[0])
            os.mkdir(path_)
        
        
        DATASETS =   np.load(os.path.join(DATAROOT, "{}.npz".format(FLAG)))
        DATASETS = split_label(DATASETS, label=LABEL)
        
        
        if not os.path.isdir(DIR_PATH):
            os.mkdir(DIR_PATH)
        
        print('==> Preparing data..')
        train_transform = transforms.Compose([
            transforms.ToTensor(), #将通道数值0-255改为0-1
            # transforms.Normalize(mean=[transform_normal_mean], std=[transform_normal_std]) #每个c 数据标准化
        ])
        
        # val_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[.5], std=[.5])
        # ])
        
        # test_transform = transforms.Compose([
              # transforms.ToTensor(),
        #     transforms.Normalize(mean=[.5], std=[.5])
        # ])
        
        train_dataset = PneumoniaMNIST(DATASETS, split='train', transform=train_transform)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataset = PneumoniaMNIST(DATASETS, split='val', transform=val_transform)
        # val_loader = DataLoader(
        #     dataset=val_dataset, batch_size=batch_size, shuffle=True)
        # test_dataset = PneumoniaMNIST(DATASETS, split='test', transform=test_transform)
        # test_loader = DataLoader(
        #     dataset=test_dataset, batch_size=batch_size, shuffle=True)
        
        # In[配置网络参数 ]
        
        #make dir of image store
        os.makedirs("images", exist_ok=True)
        
        latent_dim = latent #输入给generator 的数据
        total_epochs = 100
        sample_interval = 300
        
        #使用cuda
        cuda = True if torch.cuda.is_available() else False
        
        
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        
        # Initialize generator and discriminator
        generator = Generator_gan()
        discriminator = Discriminator()
        
        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()
        
        # Configure data loader
        
        
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, .99))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, .99))
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # In[train net]
        #同时绘制两条曲线，顺序为y1,y2, x
        viz = Visdom(env='gan_%d' % LABEL)
        
        viz.line([[0, 0]], [0], win='test', opts=dict(title='train loss', legend=['G-loss', 'D-loss']))
        
        for epoch in range(total_epochs):
            for i, (imgs, _) in enumerate(train_loader):
        
                imgs = imgs*2 - 1
                # Adversarial ground truths
                valid = Tensor(imgs.size(0), 1).fill_(1.0).cuda()
                fake = Tensor(imgs.size(0), 1).fill_(0.0).cuda()
        
                # Configure input 
                #全连接层
                real_imgs = imgs.view(imgs.size(0), -1).cuda()
        
                # -----------------
                #  Train Generator_gan
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z =Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        
                # Generate a batch of images
                gen_imgs = generator(z)
                
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()
        
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, total_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )
                
                batches_done = epoch * len(train_loader) + i
                
                viz.line([[g_loss.item(),d_loss.item()]], [batches_done], win='test', update='append')
        
                
                if batches_done % 1 == 0:
                    viz.images(gen_imgs.data[:25]*0.5+0.5, win='x', opts=dict(title='generator'))
                    viz.images(imgs.data[:25]*0.5+0.5, win='y', opts=dict(title='real imgs'))
                    # save_image(gen_imgs.data[:25]*0.5+0.5, "images/gen_%d.png" % batches_done, nrow=5, normalize=True)
                    # save_image(imgs.data[:25]*0.5+0.5, "images/true_%d.png" % batches_done, nrow=5, normalize=True)
                    
        
        #保存模型
        torch.save(generator.state_dict(),  path_+'/%d_generator.pth'% datetime.now().timestamp())
        # 载入模型
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()
