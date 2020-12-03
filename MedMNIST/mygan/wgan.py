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
from torch import autograd

from visdom import Visdom
from datetime import datetime


cuda = True if torch.cuda.is_available() else False



# In[定义 generator and discrimnator]
class Generator_wgan(nn.Module):
    def __init__(self, latent_dim = latent, img_shape = (1, 28, 28)):
        super(Generator_wgan, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 700),
            *block(700, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape = (1, 28, 28)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(.1),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# In[split dataset to train val test]
if __name__ == "__main__":
    batch_size = 200
    
    # LABEL = 1 #选择使用的label 
    for LABEL in [ 1]:
        path_ = '%s/wgan/label_%s' %(DIR_PATH, LABEL) if isinstance(LABEL, int) else '%s/wgan/label_%s' %(DIR_PATH, 'all')
        
        
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[transform_normal_mean], std=[transform_normal_std]) #每个c 数据标准化
        ])
        
        # val_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[.5], std=[.5])
        # ])
        
        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
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
        
        latent_dim = latent#输入给generator 的数据
        total_epochs = 1000
        sample_interval = 300
        clip_value = 0.6 #lower and upper clip value for disc. weights
        n_critic = 5 # number of training steps for discriminator per iter
        lr_g = .0002
        lr_d = .0002
        lr = .0002
        b1 = 0.5
        b2 = .99
        #使用cuda
        cuda = True if torch.cuda.is_available() else False
        
        
        
        
        # Initialize generator and discriminator
        generator = Generator_wgan()
        discriminator = Discriminator()
        
        if cuda:
            generator.cuda()
            discriminator.cuda()
        
        # Configure data loader
        # train_loader = train_loader
        
        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_g)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr_d)
        # optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1,b2))
        # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))
        
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # In[train net]
        #同时绘制两条曲线，顺序为y1,y2, x
        k = 2
        p = 6
        
        
        viz = Visdom(env='wgan_%d' % LABEL)
        
        viz.line([[0, 0]], [0], win='test', opts=dict(title='train loss', legend=['G-loss', 'D-loss']))
        
        
        batches_done = 0
        for epoch in range(total_epochs):
            for i, (imgs, _) in enumerate(train_loader):
                
                # imgs = imgs *2 -1
                # Configure input
                real_imgs = imgs.type(Tensor).cuda()
                real_imgs.requires_grad_(True)
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Sample noise as generator input
                z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        
                # Generate a batch of images
                fake_imgs = generator(z)
        
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity =discriminator(fake_imgs)
        
                # Compute W-div gradient penalty
                real_grad_out = Tensor(real_imgs.size(0), 1).fill_(1.0)
                # real_grad_out.requires_grad_(True)
                real_grad = autograd.grad(
                    real_validity, real_imgs, real_grad_out, create_graph=True,  retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        
                fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake_grad = autograd.grad(
                    fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        
                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
        
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
        
                d_loss.backward()
                optimizer_D.step()
        
                optimizer_G.zero_grad()
        
                # Train the generator every n_critic steps
                if i % n_critic == 0:
        
                    # -----------------
                    #  Train Generator_wgan
                    # -----------------
        
                    # Generate a batch of images
                    fake_imgs = generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)
        
                    g_loss.backward()
                    optimizer_G.step()
        
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, total_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                    )
                    viz.line([[abs(g_loss.item()),abs(d_loss.item())]], [batches_done], win='test', update='append')
                    
                    if batches_done % 1 == 0:
                        viz.images(fake_imgs.data[:25]*0.5+0.5, win='x', opts=dict(title='generator'))
                        viz.images(imgs.data[:25]*0.5+0.5, win='y', opts=dict(title='real imgs'))
                        # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        
                    batches_done += 1
        
        #保存模型
        torch.save(generator.state_dict(),  path_+'/%d_generator.pth'% datetime.now().timestamp())
        # 载入模型
        # model = Generator_wgan()
        # model.load_state_dict(torch.load(path_+'/generator.pth'))
        # model.eval()
        # plt.imshow((model(torch.randn(1, 100).type(torch.float))*0.5+0.5).detach().numpy().reshape(28, 28), cmap='gray')
