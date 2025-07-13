# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import ConvNormActivation
from torchvision.models.convnext import LayerNorm2d
from functools import partial

# def get_densenet_model(in_channels=3):
#     model = torchvision.models.densenet169(pretrained=True)
#     model.features[0] = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#     return model

def get_resnet_model():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
    model.fc = nn.Linear(512, 2)
    return model

def get_convnext_model(pretrain):
    norm_layer = partial(LayerNorm2d, eps=1e-6)
    if pretrain:
        model = torchvision.models.convnext_tiny(weights=torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT)
    else:
        model = torchvision.models.convnext_tiny(pretrain=False)
    model.classifier = nn.Sequential(
        norm_layer(768), nn.Flatten(1), nn.Linear(768, 2)
    )
    return model

def get_efficientnet_model():
    model = torchvision.models.efficientnet_b0(pretrained=True)
    return model