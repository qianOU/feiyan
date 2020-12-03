from settings import *
from dataloader import *

import os
import numpy as np
import math
from matplotlib import pyplot as plt
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





class generator_dcwgan(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator_dcwgan, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # self.pool = nn.MaxPool2d()
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.linear = nn.Linear(d * 8, 1)
        self.outlayer = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2) #torch.Size([200, 128, 14, 14])
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) #torch.Size([200, 256, 7, 7])
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        # print(x.shape)
        # x = self.linear(x.reshape(x.shape[0], -1))
        x = self.outlayer(x).squeeze()
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# In[split dataset to train val test]
if __name__ == "__main__":
    batch_size = 60

    # LABEL = 1 #选择使用的label
    for LABEL in [0]:
        path_ = '%s/wgan/label_%s' % (DIR_PATH, LABEL) if isinstance(LABEL, int) else '%s/wgan/label_%s' % (
        DIR_PATH, 'all')

        if not os.path.exists(path_):
            if not os.path.exists(path_.rsplit('label_')[0]):
                os.mkdir(path_.rsplit('label_')[0])
            os.mkdir(path_)

        DATASETS = np.load(os.path.join(DATAROOT, "{}.npz".format(FLAG)))
        DATASETS = split_label(DATASETS, label=LABEL)

        if not os.path.isdir(DIR_PATH):
            os.mkdir(DIR_PATH)

        print('==> Preparing data..')
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])  # 每个c 数据标准化
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

        # make dir of image store
        os.makedirs("images", exist_ok=True)

        latent_dim = latent  # 输入给generator 的数据
        gen_size = gen_size

        total_epochs = 50
        sample_interval = 100
        clip_value = 0.01  # lower and upper clip value for disc. weights
        n_critic = 1  # number of training steps for Discriminator per iter
        lr_g = .00005
        lr_d = .000005
        lr = .0006
        b1 = 0.5
        b2 = .99
        # 使用cuda
        cuda = True if torch.cuda.is_available() else False

        # Initialize generator and Discriminator
        generator = generator_dcwgan(128)
        Discriminator = Discriminator(128)
        generator.weight_init(mean=0.0, std=0.02)
        Discriminator.weight_init(mean=0.0, std=0.02)
        if cuda:
            generator.cuda()
            Discriminator.cuda()

        # Configure data loader
        # train_loader = train_loader

        # Optimizers
        # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_g)
        # optimizer_D = torch.optim.RMSprop(Discriminator.parameters(), lr=lr_d)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(b1,b2))
        optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=lr_d, betas=(b1,b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # In[train net]
        # 同时绘制两条曲线，顺序为y1,y2, x
        k = 2
        p = 6

        viz = Visdom(env='wgan_%d' % LABEL)

        viz.line([[0, 0]], [0], win='test', opts=dict(title='train loss', legend=['G-loss', 'D-loss']))
        batches_done = 0
        
        total_g_loss = []
        total_d_loss = []
        for epoch in range(total_epochs):
            g_loss_list = []
            d_loss_list = []
            if (epoch+1) == 25:
                optimizer_G.param_groups[0]['lr'] /= 10
                optimizer_D.param_groups[0]['lr'] /= 10
                print("learning rate change!")
        
            if (epoch+1) == 35:
                optimizer_G.param_groups[0]['lr'] /= 10
                optimizer_D.param_groups[0]['lr'] /= 10
                print("learning rate change!")
        
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
                z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).view(-1, latent_dim, 1, 1)

                # Generate a batch of images
                fake_imgs = generator(z)

                # Real images
                real_validity = Discriminator(real_imgs)
                # Fake images
                fake_validity = Discriminator(fake_imgs)

                # Compute W-div gradient penalty
                real_grad_out = Tensor(real_imgs.size(0)).fill_(1.0)
                real_grad_out.requires_grad_(False)
                # real_grad_out.requires_grad_(True)
                real_grad = autograd.grad(
                    real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                fake_grad_out = Variable(Tensor(fake_imgs.size(0)).fill_(1.0), requires_grad=False)
                fake_grad = autograd.grad(
                    fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + 2*torch.mean(fake_validity) + div_gp
                
                d_loss.backward()
                optimizer_D.step()
                d_loss_list.append(d_loss.detach().cpu().item())
                optimizer_G.zero_grad()
                # Train the generator every n_critic steps
                if i % n_critic == 5:
                    
                    # -----------------
                    #  Train generator_dcgan
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = generator(z)
                    # Loss measures generator's ability to fool the Discriminator
                    # Train on fake images
                    fake_validity = Discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)*torch.tensor([10]).cuda()
                    g_loss.backward()
                    optimizer_G.step()
                    g_loss_list.append(g_loss.detach().cpu().item())

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, total_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                    )
                    viz.line([[g_loss.item(),d_loss.item()]], [batches_done], win='test', update='append')

                    if batches_done % sample_interval == 0:
                        generator.eval()
                        with torch.no_grad():
                            fake_imgs = generator(z)
                            viz.images(fake_imgs.data[:25] * 0.5 + 0.5, win='x', opts=dict(title='generator'))
                            viz.images(imgs.data[:25] * 0.5 + 0.5, win='y', opts=dict(title='real imgs'))
                            # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                        generator.train()
                    if batches_done % sample_interval == 0:
                        save_image(fake_imgs.data[:25]*0.5+0.5, "images/wgan/%d/gen_%d.png" % (LABEL, batches_done), nrow=5, normalize=True)
                        # save_image(imgs.data[:25]*0.5+0.5, "images/true_%d.png" % batches_done, nrow=5, normalize=True)
                    batches_done += n_critic
            total_d_loss.append(np.mean(d_loss_list))
            total_g_loss.append(np.mean(g_loss_list))
        

        plt.figure()
        plt.plot(range(total_epochs), total_d_loss,color='b', label = 'D-loss')
        plt.plot(range(total_epochs), total_g_loss, color='r', label='G-loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.title('gan for label:%d' %LABEL)
        plt.savefig("images/wgan/%d/training_trace.png" % (LABEL))
        plt.show()
        # 保存模型
        torch.save(generator.state_dict(), path_ + '/%d_generator.pth' % datetime.now().timestamp())
        # 载入模型
        # model = generator_dcgan()
        # model.load_state_dict(torch.load(path_+'/generator.pth'))
        # model.eval()
        # plt.imshow((model(torch.randn(1, 100).type(torch.float))*0.5+0.5).detach().numpy().reshape(28, 28), cmap='gray')
