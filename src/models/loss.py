from torch import nn
import torch
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, real_label, fake_label):
        loss_disc = (fake_label ** 2 + (real_label - 1) ** 2).mean()
        loss_gen = ((fake_label - 1) ** 2).mean()
        return loss_gen, loss_disc


class VGGLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGLoss, self).__init__()
        self.vgg16 = vgg16(requires_grad)

    def forward(self, input, target):
        v_input = self.vgg16(input)
        v_target = self.vgg16(target)
        loss = 0.0
        for i in range(4):
            loss += F.mse_loss(v_input[i], v_target[i])
        loss /= 4
        return loss


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out