#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: fire_smoke_detection.py
@Author:kong
@Time: 2020年01月03日10时50分
@Description: efficientnet烟火检测
'''

import json
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from efficientnet_pytorch import FireSmokeEfficientNet
import collections

# from PIL import Image, ImageDraw, ImageFont
image_path = './tests/000127.jpg'
col = 5
row = 4
interCLS = ["smoke","fire"]
model_para = collections.OrderedDict()
model = FireSmokeEfficientNet.from_arch('efficientnet-b0')
# out_channels = model._fc.in_features
model._fc = torch.nn.Linear(1280, 3)
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

# Load ImageNet class names
labels_map = json.load(open('examples/simple/fire_smoke_map.txt'))
labels_map = [labels_map[str(i)] for i in range(3)]

image = Image.open(image_path)
width = image.width
height = image.height
w_len = int(width / col)        ##每个block 长宽:h_len/w_len
h_len = int(height / row)

draw = ImageDraw.Draw(image)
font = ImageFont.truetype("simkai.ttf", 40, encoding="utf-8")#格式，参数分别为 字体文件，文字大小，编码方式

for r in range(row):
    for c in range(col):
        image_tmp = image.crop((c*w_len,r*h_len,(c+1)*w_len,(r+1)*h_len))
        img_tmp = tfms(image_tmp).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            outputs = model(img_tmp)
        print('-----')
        for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))
            # image_tmp.save('{}_{}_{}.jpg'.format(r, c, labels_map[idx]))
            if prob > 0.99 and labels_map[idx] in interCLS:
                draw.line([(c*w_len,r*h_len),((c+1)*w_len, r*h_len),((c+1)*w_len, (r+1)*h_len),(c*w_len,(r+1)*h_len),(c*w_len,r*h_len)],fill = (255,0,0), width = 2)
                draw.text(((c+1)*w_len, r*h_len), labels_map[idx], (255, 0, 0), font=font)  # 写文字，参数为文字添加位置，添加的文字字符串，文字颜色，格式

image.save("results/det_results{}".format(image_path.split('/')[-1]))

