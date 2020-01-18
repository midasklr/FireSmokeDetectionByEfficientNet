#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: fire_smoke_demo.py
@Author:kong
@Time: 2020年01月02日15时45分
@Description:
'''
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import FireSmokeEfficientNet
import collections
image_dir = './tests/000294.jpg'
model_para = collections.OrderedDict()
model = FireSmokeEfficientNet.from_arch('efficientnet-b0')
# out_channels = model._fc.in_features
model._fc = nn.Linear(1280, 3)
print(model)
modelpara = torch.load('./checkpoint.pth.tar')
# print(modelpara['state_dict'].keys())
for key in modelpara['state_dict'].keys():
    # print(key[7:])
    # newkey = model_para[key.split('.',2)[-1]]
    # print(newkey)
    model_para[key[7:]] =modelpara['state_dict'][key]

# print(model_para.keys())
# 训练模型转换



model.load_state_dict(model_para)

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
image = Image.open(image_dir)
img = tfms(image).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('examples/simple/fire_smoke_map.txt'))
labels_map = [labels_map[str(i)] for i in range(3)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

draw = ImageDraw.Draw(image)
font = ImageFont.truetype('simkai.ttf', 30)
# Print predictions
print('-----')
cout = 0
for idx in torch.topk(outputs, k=2).indices.squeeze(0).tolist():
    cout += 1
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
    position = (10, 30*cout - 20)
    text = '{label:<5} :{p:.2f}%'.format(label=labels_map[idx], p=prob*100)
    draw.text(position, text, font=font, fill="#ff0000", spacing=0, align='left')

image.save('results/result_{}'.format(image_dir.split('/')[-1]))