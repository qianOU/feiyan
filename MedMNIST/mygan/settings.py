# -*- coding: utf-8 -*-
from torchvision import transforms
FLAG = "pneumoniamnist"
DATAROOT = './'

#建立模型存放位置
DIR_PATH = './%s_model_store' % (FLAG)

#图片大小样式
img_shape = (1, 28, 28)



latent = 100
gen_size = 5

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
            "val": 0.1,
            "test": 0.1
        },
        "num": {
            "train": 4708,
            "val": 524,
            "test": 624
        }
    }
        }
