## 基于pneumoniamnist（肺炎数据集）的GAN与WGAN的生成图片实验
### 使用Alexnet作为判别器，检验图像的生成效果
## 实验结果
#### 1. 对于label=0，即为非肺炎的CT图像有更好的生成效果
#### 2. 生成的图片对于人类来说几乎难以区分，但是对于机器来说还是可以较为轻易的识别出来
## 优化空间
#### 可以尝试使用新的GAN来生成图片
#### 原始噪声输入可以从正态分布，换为均与分布
## 经验（triks)：
#### 1. 对于28\*28的图像GAN与WGAN需要将图片扩大为64*64的大小,再输入判别器中
