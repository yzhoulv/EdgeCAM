from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class InterReguProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=True):
        super(InterReguProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _weight_loss(self, weight, label):

        weight = F.normalize(weight, dim=1)
        weight_label = weight[label]
        temp_res = weight_label.repeat(1, weight.shape[0])
        temp_res = temp_res.reshape(label.shape[0] * weight.shape[0], 512)
        temp_res = torch.sum((temp_res - weight.repeat(label.shape[0], 1)) ** 2, dim=1)
        temp_res = temp_res.reshape(label.shape[0], -1)
        values = torch.where(temp_res == 0, torch.ones(temp_res.shape, device="cuda")*1000, temp_res)
        value, _ = torch.min(values, dim=1)
        return label.shape[0] / (torch.sum(value))

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        weight_loss = self._weight_loss(self.weight, label)
        # weight_loss = 0

        return output, weight_loss
        # return input, 0

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'