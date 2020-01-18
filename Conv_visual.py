#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: Conv_visual.py
@Author:kong
@Time: 2020年01月07日09时59分
@Description:可视化fire&smoke 模型
'''


import cv2
import numpy as np
import torch
from torch.autograd import Variable
import json
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision import models
from efficientnet_pytorch import FireSmokeEfficientNet
import collections

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def getPretrainedModel():
    model_para = collections.OrderedDict()
    model_tmp = FireSmokeEfficientNet.from_arch('efficientnet-b0')
# out_channels = model._fc.in_features
    model_tmp._fc = torch.nn.Linear(1280, 3)
    modelpara = torch.load('./checkpoint.pth.tar')
    # print(modelpara['state_dict'].keys())
    for key in modelpara['state_dict'].keys():
        model_para[key[7:]] = modelpara['state_dict'][key]
    model_tmp.load_state_dict(model_para)
    return model_tmp


class FeatureVisualization():
    def __init__(self,img_path,selected_layer):
        self.img_path=img_path
        self.image = cv2.imread(img_path)
        self.selected_layer=selected_layer
        self.pretrained_model = getPretrainedModel()

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        print(input.shape)
        # x=input

        x = self.pretrained_model._swish(self.pretrained_model._bn0(self.pretrained_model._conv_stem(input)))
        for index,layer in enumerate(self.pretrained_model._blocks):
            x=layer(x)
            if (index == self.selected_layer):
                return x
        # x = self.pretrained_model._conv_head(x)
        # return x


    def get_single_feature(self):
        features=self.get_feature()
        # print('特征是:',features.shape)

        feature=features[:,6,:,:]
        print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)

        return feature

    def get_all_feature(self):
        features=self.get_feature()
        # print(':',features.shape)

        feature=features[:,:,:,:]
        print(feature.shape)

        feature=feature.view(feature.shape[1],feature.shape[2],feature.shape[3])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_all_feature()
        feature=feature.data.numpy()

        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))

        # to [0,255]
        feature=np.round(feature*255)
        print(feature[0])
        print("image size:",self.image.shape)
        print("feature map size:",feature.shape)
        Nch = feature.shape[0]   #channel num
        for i in range(Nch):
            print('--------------show the {}th channels-----------------'.format(i))
            feature_resize = cv2.resize(feature[i], (self.image.shape[1], self.image.shape[0]))
            feature_resize = cv2.cvtColor(feature_resize, cv2.COLOR_GRAY2BGR)
            feature_resize = cv2.putText(feature_resize,'first_depthwise_conv {}th feature map'.format(i),(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            image_cont = np.concatenate([self.image,feature_resize],axis=0)
            cv2.imwrite('featmap/5rd_depthwise_conv_featmap{}_{}'.format(i,self.img_path.split('/')[-1]),image_cont)

if __name__=='__main__':
    # get class
    myClass=FeatureVisualization('./tests/000294.jpg',1)
    # print (myClass.pretrained_model)
    print("-----------------------------------------------------")
    print(myClass.pretrained_model)

    myClass.save_feature_to_img()
