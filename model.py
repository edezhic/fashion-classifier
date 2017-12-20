import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fractions import gcd
import numpy as np
from batchrenorm import BatchReNorm1d, BatchReNorm2d

class CoolNameNet(nn.Module):
    def __init__(self, input_channels, output_classes, width=2, depth=3, density=2, shortcuts=False):
        super(CoolNameNet, self).__init__()

        self.planes = width * 32

        self.input_conv = nn.Conv2d(input_channels, self.planes, kernel_size=3, bias=False)

        # CNN with Dense connectivity and SqeezeAndExcite blocks
        dense_blocks = []
        for i in range(depth):
            groups = min(32, 2**(i+1))
            dense_blocks.append(DenseBlock(self.planes, density, groups, add_shortcut=shortcuts))

        self.network = nn.Sequential(*dense_blocks)

        self.bn = BatchReNorm1d(self.planes)
        self.output_classes = output_classes
        self.output_linear = nn.Linear(self.planes, self.output_classes, bias=False)

    def forward(self, x):
        y = self.input_conv(x)
        y = self.network(y)
        y = F.adaptive_avg_pool2d(y, 1) + F.adaptive_max_pool2d(y, 1)
        y = y.view(y.size(0), self.planes)
        y = self.bn(y)
        y = self.output_linear(y)
        y = y.view(y.size(0), self.output_classes)
        return y

    def total_parameters_number(self):
        total_params = 0
        for key, module in self._modules.items():
            total_params += sum([np.prod(p.size()) for p in module.parameters()])
        return total_params

class DenseBlock(nn.Module):
    def __init__(self, planes, density, groups, add_shortcut):
        super(DenseBlock, self).__init__()
        self.inner_blocks = [None] * density
        self.add_shortcut = add_shortcut
        total_planes = planes

        for i in range(density):
            output_planes = max(planes // 2**i, 32)
            self.inner_blocks[i] = BasicBlock(total_planes, output_planes, groups)
            total_planes += output_planes

        self.bottleneck = nn.Sequential(
            BasicBlock(total_planes, planes, groups),
            nn.Dropout2d(p=0.05),
            BasicBlock(planes, planes, groups, stride=2),
            SqeezeAndExcite(planes)
        )

        if self.add_shortcut:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(planes).cuda(),
                nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False).cuda(),
            )

    def forward(self, x):
        y = x
        # Pass all concatenated planes from previous blocks to each subsequent
        for block in self.inner_blocks:
            y = torch.cat((y, block(y)), 1)

        y = self.bottleneck(y)

        if self.add_shortcut:
            return y + self.shortcut(x)
        else:
            return y

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups, stride=1):
        super(BasicBlock, self).__init__()
        groups = check_groups(in_planes, out_planes, groups)
        self.bn = nn.BatchNorm2d(in_planes).cuda()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False).cuda()

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))

class SqeezeAndExcite(nn.Module):
    def __init__(self, planes, reduction = 8):
        super(SqeezeAndExcite, self).__init__()
        self.sqex = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False).cuda(),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False).cuda(),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.sqex(y)
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

def check_groups(in_planes, out_planes, groups):
    if in_planes % groups != 0 or out_planes % groups != 0:
        return gcd(in_planes, out_planes)
    else:
        return groups
