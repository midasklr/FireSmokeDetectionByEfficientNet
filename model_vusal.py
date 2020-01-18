#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: model_vusal.py
@Author:kong
@Time: 2020年01月07日19时17分
@Description:
'''

import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from torchvision.datasets import ImageFolder

torch.cuda.set_device(0)  # 设置GPU ID
is_cuda = True
simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),  # H, W, C -> C, W, H 归一化到(0,1)，简单直接除以255
                                       transforms.Normalize([0.485, 0.456, 0.406],  # std
                                                            [0.229, 0.224, 0.225])])

# mean  先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
# 使用 ImageFolder 必须有对应的目录结构
train = ImageFolder("/home/kong/Documents/EfficientNet-PyTorch/cropdata/train", simple_transform)
valid = ImageFolder("/home/kong/Documents/EfficientNet-PyTorch/cropdata/val", simple_transform)
train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=5)
val_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=5)

vgg = models.mobilenet_v2(pretrained=True).cuda()


# 提取不同层输出的 主要代码
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


conv_out = LayerActivations(vgg.features, 18)  # 提出第 一个卷积层的输出
img = next(iter(train_loader))[0]
o = vgg(Variable(img.cuda()))
conv_out.remove()  #
act = conv_out.features  # act 即 第0层输出的特征

# 可视化 输出
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(30):
    ax = fig.add_subplot(12, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(act[0][i].detach().numpy(), cmap="gray")
plt.show()