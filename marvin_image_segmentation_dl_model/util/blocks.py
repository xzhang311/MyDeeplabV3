#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import kaiming_normal

class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, baseplanes, stride=1, rate=1, downsample=None):
        super(BottleneckBlock, self).__init__()

        kernel_size = 3
        kernel_size_effective = kernel_size + (kernel_size-1)*(rate-1)
        pad_total = kernel_size_effective - 1
        self.pad_size = pad_total // 2
        self.inplanes = inplanes
        self.baseplanes = baseplanes

        self.conv1 = nn.Conv2d(self.inplanes, self.baseplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.baseplanes)

        self.conv2 = nn.Conv2d(self.baseplanes, self.baseplanes, kernel_size=kernel_size, stride=stride, padding=self.pad_size, bias=False, dilation=rate)
        self.bn2 = nn.BatchNorm2d(self.baseplanes)

        self.conv3 = nn.Conv2d(self.baseplanes, self.baseplanes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.baseplanes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AtrousResNetBlock(nn.Module):
    def __init__(self, baseplanes, inplanes, num_units, stride_first_block=1, base_rate = 1, multi_grid = None):
        super(AtrousResNetBlock, self).__init__()

        self.expansion = 4
        self.baseplanes = baseplanes
        self.inplanes = inplanes
        self.outplanes = self.baseplanes * self.expansion
        self.num_units = num_units
        self.base_rate = base_rate
        self.multi_grid = multi_grid
        self.stride_first_block = stride_first_block
        self.block = self._make_block()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self):
        BottleneckBlocks = []

        _inplanes = self.inplanes

        # First BottleneckBlock block
        if self.multi_grid is not None:
            dilate_rate = self.multi_grid[0] * self.base_rate
        else:
            dilate_rate = 1

        downsample = nn.Sequential(
            nn.Conv2d(_inplanes, self.baseplanes * self.expansion, kernel_size=1, stride=self.stride_first_block, bias=False),
            nn.BatchNorm2d(self.baseplanes * self.expansion)
        )
        BottleneckBlocks.append(BottleneckBlock(_inplanes, self.baseplanes, rate=dilate_rate, stride=self.stride_first_block, downsample=downsample))

        # Rest BottleneckBlock block
        _inplanes = self.baseplanes * self.expansion
        for i in range(1, self.num_units):
            if self.multi_grid is not None:
                dilate_rate = self.multi_grid[i] * self.base_rate
            else:
                dilate_rate = 1

            if _inplanes != self.baseplanes * self.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(_inplanes, self.baseplanes * self.expansion, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.baseplanes * self.expansion)
                )
            else:
                downsample = None

            BottleneckBlocks.append(BottleneckBlock(_inplanes, self.baseplanes, rate=dilate_rate, downsample=downsample))
            _inplanes = self.baseplanes * self.expansion

        return nn.Sequential(*BottleneckBlocks)

    def forward(self, x):
        out = self.block(x)
        return out

def conv_bn_relu(inplanes, outplanes, kernel_size=3, stride=1, rate=1):
    kernel_size = kernel_size
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_size = pad_total // 2

    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=pad_size, dilation=rate, bias=False),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    )

def conv_relu(inplanes, outplanes, kernel_size=3, stride=1, rate=1):
    kernel_size = kernel_size
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_size = pad_total // 2

    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=pad_size, dilation=rate, bias=True),
        nn.ReLU(inplace=True)
    )

def deconv_bn_relu(inplanes, outplanes, kernel_size = 4, stride = 2):
    return nn.Sequential(
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    )
