import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class DenseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, bn_size):
        super(DenseLayer, self).__init__()
        planes = out_planes * bn_size
        self.conv1 = conv1x1(in_planes, planes, bias=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, out_planes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return torch.cat([x, out], 1)


class CompressionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(CompressionLayer, self).__init__()
        self.conv1 = conv1x1(in_planes, out_planes)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return out


class DenseCompressionBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, bn_size, num_layers):
        super(DenseCompressionBlock, self).__init__()
        for i in range(num_layers):
            self.add_module("denselayer%d" % (i),
                            DenseLayer(in_planes + i * growth_rate,
                                       growth_rate,
                                       bn_size))
        self.add_module("compression", CompressionLayer(
                                            in_planes + num_layers * growth_rate,
                                            in_planes))

    def forward(self, x):
        identity = x
        for module in self.children():
            x = module(x)
        return x + identity


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_planes, out_planes, upscale_factor):
        super(PixelShuffleUpsampler, self).__init__()
        self.conv = conv3x3(in_planes, (upscale_factor ** 2) * out_planes)
        self.shfl = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shfl(x)
        return self.prelu(x)


class PyramidUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, denseblock_config,
                 bn_size, growth_rate, upscale_factor=2):
        super(PyramidUpBlock, self).__init__()
        for i, num_layers in enumerate(denseblock_config):
            denseblock_params = dict(in_planes=in_planes,
                                     num_layers=num_layers,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate)
            self.add_module("denseblock_%d" % i,
                            DenseCompressionBlock(**denseblock_params))

        self.add_module("final_conv", conv3x3(in_planes, in_planes))
        self.add_module("upsampler",
                        PixelShuffleUpsampler(in_planes, out_planes,
                                              upscale_factor))

    def forward(self, x):
        identity = x
        for name, module in self.named_children():
            if name == "final_conv":
                x = module(x) + identity
            else:
                x = module(x)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownLayer, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, bias=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_planes, out_planes, bias=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.avg = nn.AvgPool2d(2)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class PyramidDownBlock(nn.Module):
    def __init__(self, in_planes, downplanes_config):
        super(PyramidDownBlock, self).__init__()
        for i, out_planes in enumerate(downplanes_config):
            self.add_module("downlayer_%d" % i, DownLayer(in_planes, out_planes))
            in_planes = out_planes

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

