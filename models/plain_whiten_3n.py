"""
plain_whiten_3n for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math
import extension as my

__all__ = ['PLAIN_whiten_3n','plain_whiten_3n20', 'plain_whiten_3n32', 'plain_whiten_3n44', 'plain_whiten_3n56', 'plain_whiten_3n110', 'plain_whiten_3n164',
           'plain_whiten_3n1001', 'plain_whiten_3n1202']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)

    def forward(self, x):
        x = self.relu(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x





class PLAIN_whiten_3n(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PLAIN_whiten_3n, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = my.Norm(16)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(self.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def plain_whiten_3n20(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [3, 3, 3], **kwargs)
    return model


def plain_whiten_3n32(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [5, 5, 5], **kwargs)
    return model


def plain_whiten_3n44(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [7, 7, 7], **kwargs)
    return model


def plain_whiten_3n56(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [9, 9, 9], **kwargs)
    return model


def plain_whiten_3n110(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [18, 18, 18], **kwargs)
    return model


def plain_whiten_3n164(**kwargs):
    model = PLAIN_whiten_3n(Bottleneck, [18, 18, 18], **kwargs)
    return model


def plain_whiten_3n1001(**kwargs):
    model = PLAIN_whiten_3n(Bottleneck, [111, 111, 111], **kwargs)
    return model


def plain_whiten_3n1202(**kwargs):
    model = PLAIN_whiten_3n(BasicBlock, [200, 200, 200], **kwargs)
    return model



if __name__ == '__main__':
    net = preact_plain_whiten_3n110()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())
