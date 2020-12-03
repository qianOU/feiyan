# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:22:29 2020

@author: 28659
"""

from test_dcgan import Generator_gan
from test_wgan import generator_dcwgan
from evaluator import getAUC, getACC, save

from settings import *
from dataloader import *
# from models import ResNet18,ResNet50

import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from torch.utils import data
from torch import optim
from torchvision.models import resnet18
import torch.nn as nn
import torch

from matplotlib import pyplot as plt

from models import ResNet50
from visdom import Visdom


ResNet18 = resnet18()
ResNet18.fc = nn.Linear(ResNet18.fc.in_features, 2)
ResNet18.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)


alexnet = torchvision.models.AlexNet()
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 2)
alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
print(alexnet)

cuda = True if torch.cuda.is_available() else False
if cuda:
    torch.cuda.set_device(0)


#label=1 gan生成网络
# =============================================================================
# 注意：我们当初保存的模型参数都是在cuda中的，所以这里一定要将实例化的模型利用.cuda或则
# .to(torch.device('cuda'))保存到cuda中后，才能载入保存的模型参数
# 否则 会报 CUDA error: an illegal memory access was encountered 错误
# =============================================================================
# gan_label_1 = Generator_gan(64).cuda()  #必修先加载入cuda中
# gan_label_1.load_state_dict(torch.load(gan_label_1_path))
# gan_label_1.eval()

# #label=0 gan生成网络
# gan_label_0 = Generator_gan(64).cuda()
# gan_label_0.load_state_dict(torch.load(gan_label_0_path))
# gan_label_0.eval()

# #label=1 wgan生成网络
# wgan_label_1 = generator_dcwgan(128).cuda()
# wgan_label_1.load_state_dict(torch.load(wgan_label_1_path))
# wgan_label_1.eval()

# #label=0 wgan生成网络
# wgan_label_0 = generator_dcwgan(64).cuda()
# wgan_label_0.load_state_dict(torch.load(wgan_label_0_path))
# wgan_label_0.eval()



# def imshow(img, label):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
#     plt.title('label=%d, beyond iswgan, below is gan' % label)
#     plt.show()

# input_ = torch.randn(10, latent, 1, 1).cuda()
# print(input_.shape)
# result = []
# for model in [wgan_label_0, gan_label_0]:
#     result.append(model(input_).detach().cpu())

# imshow(torchvision.utils.make_grid(torch.cat(result, 0), nrow=10), label=0)

# result = []
# for model in [wgan_label_1, gan_label_1]:
#     result.append(model(input_).detach().cpu())

# imshow(torchvision.utils.make_grid(torch.cat(result, 0), nrow=10), label=1)

# In[]


def log(question, path, pattern):
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, str(question))
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, pattern)
    if not os.path.exists(path):
        os.mkdir(path)
    return path
    
def main(question,  gan_model, path=os.path.join(DATAROOT, "{}.npz".format(FLAG)), pattern='gan', total_epochs=8):
    ''' main function
    :param flag: name of subset

    '''
    label = [0, 1] #解决1，2问
    if question == 3:
        label = 0 #解决3问
    elif question == 4:
        label = 1 #解决 4问
    
    if question == 2:
        assert len(gan_model) == 2, '对于问题二需要输入两个生成器，且0类型在前，1类型在后'

    
    #可视化
    viz = Visdom(env='examine_'+pattern + '_' + str(question))
    viz.line([[0, 0]], [0], win='train', opts=dict(title='train loss', legend=['train-loss', 'acc']))
    viz.line([[0, 0 ]], [0], win='val', opts=dict(title='auc', legend=['auc', 'acc']))
    viz.line([[0, 0]], [0], win='test', opts=dict(title='test auc&acc', legend=['auc', 'acc']))
    

    end_epoch = total_epochs #训练论数
    lr = 0.0001
    batch_size = 60
    val_auc_list = []
    dir_path = './%s_model_store' % ('classfier')
    #分类型设置保存路径
    dir_path = log(question, dir_path, pattern)
    
    #保存训练过程csv文件
    global outputdir
    outputdir= './peformance'
    outputdir = log(question, outputdir, pattern)
    
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    print('==> Preparing data..')
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]) #每个c 数据标准化
    ]) if gan_model is  None else None

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ]) if gan_model is  None else None

    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ]) if gan_model is  None else None
    DATASETS =   np.load(path)
    DATASETS = split_label(DATASETS, label=label, gan=False, gan_model=gan_model)
    
    train_dataset = PneumoniaMNIST(DATASETS, split='train', transform=train_transform)
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = PneumoniaMNIST(DATASETS, split='val', transform=val_transform)
    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = PneumoniaMNIST(DATASETS, split='test', transform=test_transform)
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Building model..')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = ResNet18(in_channels=1, num_classes=2).to(device)
    # model = ResNet18.to(device) #使用pytorch自带的resnet构造
    model = alexnet.to(device)
    #defin loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(0, end_epoch):
        train(model, optimizer, criterion, train_loader, device, viz, epoch)
        val(model, val_loader, device, val_auc_list, dir_path, epoch, viz)
        print('%s|%s done!' % (epoch, end_epoch))
    
    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model..')
    restore_model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    # test(model, 'train', train_loader, device, flag, task)
    # test(model, 'val', val_loader, device, flag, task)
    test(model,  test_loader, device,  epoch, viz)


def train(model, optimizer, criterion, train_loader, device, viz, epoch):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda


    '''

    model.train()
    length = len(train_loader)
    # print(dir(model))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        # if task == 'multi-label, binary-class':
        # targets = targets.to(torch.float32).to(device)
        # loss = criterion(outputs, targets)
        # else:
        targets = targets.squeeze().long().to(device)
        loss = criterion(outputs, targets)

        viz.line([[loss.item(),
                  getACC(targets.cpu().numpy(), outputs.detach().cpu().numpy())]], [epoch*length+batch_idx+1], win='train', update='append')
        loss.backward()
        optimizer.step()

        viz.images(inputs.data[:40] * 0.5 + 0.5, win='x', opts=dict(title='train set'))
            # viz.images(imgs.data[:25] * 0.5 + 0.5, win='y', opts=dict(title='real imgs'))
        viz.text(str(targets.squeeze()[:40].cpu().numpy()), win='x2', opts=dict(title='pred'))

def val(model, val_loader, device, val_auc_list, dir_path, epoch, viz):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    length = len(val_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))
            # viz.images(inputs.data[:40] * 0.5 + 0.5, win='x2', opts=dict(title='val set'))

            # if task == 'multi-label, binary-class':
            #     targets = targets.to(torch.float32).to(device)
            #     m = nn.Sigmoid()
            #     outputs = m(outputs).to(device)
            # else:
            targets = targets.squeeze().long().to(device)
            m = nn.Softmax(dim=1) #沿axis=1做计算
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            # viz.line([[getAUC(y_true.cpu().numpy(), y_score.detach().cpu().numpy()), 
            #           getACC(y_true.cpu().numpy(), y_score.detach().cpu().numpy())]], [epoch*length+batch_idx+1], win='val', update='append')
            
        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        viz.line([[getAUC(y_true, y_score), 
                  getACC(y_true, y_score)]], [epoch+1], win='val', update='append')
        
        
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model,  data_loader, device, epoch, viz):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    length = len(data_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long().to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)
            viz.line([[getAUC(y_true.cpu().numpy(), y_score.detach().cpu().numpy()), 
                      getACC(y_true.cpu().numpy(), y_score.detach().cpu().numpy())]], 
                     [epoch*length+1+batch_idx], win='test', update='append')
            y_true = torch.cat((y_true, targets), 0) #2 dim tensor
            y_score = torch.cat((y_score, outputs), 0)# 2 dim tensor
        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        acc = getACC(y_true, y_score)
        
        print('test AUC: %.5f ACC: %.5f' % (auc, acc))

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        outputpath = os.path.join(outputdir, '%s.csv' % (type(model).__name__))
        save(y_true, y_score, outputpath)

#制作gif动态图


if __name__ == '__main__':
    pattern = 'gan'
    question = 0
    
    gan_label_1_path = 'pneumoniamnist_model_store/gan/label_1/1606295785_generator.pth'
    gan_label_0_path = 'pneumoniamnist_model_store/gan/label_0/1606275486_generator.pth'
    wgan_label_1_path = 'pneumoniamnist_model_store/wgan/label_1/1606252977_generator.pth'
    wgan_label_0_path = 'pneumoniamnist_model_store/wgan/label_0/1606269702_generator.pth'

    #确定生成器类型
    if pattern == 'wgan':
        generator_class = generator_dcwgan
        model_pth = [wgan_label_0_path, wgan_label_1_path]
    elif pattern == 'gan':
        generator_class = Generator_gan
        model_pth = [gan_label_0_path, gan_label_1_path]

    gan_model = []
    #实例化model
    if question == 2 or question== 4:
        label_1 =  generator_class(64).cuda()  #必修先加载入cuda中
        label_1.load_state_dict(torch.load(model_pth[1]))
        label_1.eval()
        gan_model.append(label_1)
    
    if question == 2 or question == 3:
    #label=0 gan生成网络
        label_0 = generator_class(64).cuda()
        label_0.load_state_dict(torch.load(model_pth[0]))
        label_0.eval()
        gan_model.append(label_0)
        
    #确定训练的模型
    if len(gan_model) == 2 or not gan_model:
        model = None if len(gan_model)==0  else gan_model
    else:
        model = gan_model[0]

    main(question,  model, path=os.path.join(DATAROOT, "{}.npz".format(FLAG)), pattern=pattern)