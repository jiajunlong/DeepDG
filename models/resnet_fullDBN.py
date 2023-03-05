"""
resnet_fullDBN for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math
import extension as my

__all__ = ['ResNet_fullDBN', 'PreAct_ResNet_fullDBN', 'resnet_fullDBN20', 'resnet_fullDBN32', 'resnet_fullDBN44', 'resnet_fullDBN56', 'resnet_fullDBN110', 'resnet_fullDBN164',
           'resnet_fullDBN1001', 'resnet_fullDBN1202', 'preact_resnet_fullDBN20', 'preact_resnet_fullDBN110', 'preact_resnet_fullDBN164', 'preact_resnet_fullDBN1001']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = my.Norm(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)
        self.shortcut = shortcut

    def forward(self, x):
        x = self.relu(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = my.Norm(planes)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = my.Norm(planes * 4)

        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual
        return x


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(PreActBasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1 = my.Norm(inplanes)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        residual = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += residual
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(PreActBottleneck, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1 = my.Norm(inplanes)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
       # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = my.Norm(planes)
        self.relu2 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes)
        self.bn3 = my.Norm(planes)
        self.relu3 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(x))
        if self.shortcut is not None:
            residual = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out += residual
        return out


class ResNet_fullDBN(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_fullDBN, self).__init__()
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
        shortcut = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, shortcut)]
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


class PreAct_ResNet_fullDBN(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_fullDBN, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        #self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.bn = my.Norm(64 * block.expansion)
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
        shortcut = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = [block(self.inplanes, planes, stride, shortcut)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_fullDBN20(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet_fullDBN32(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet_fullDBN44(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet_fullDBN56(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet_fullDBN110(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet_fullDBN164(**kwargs):
    model = ResNet_fullDBN(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet_fullDBN1001(**kwargs):
    model = ResNet_fullDBN(Bottleneck, [111, 111, 111], **kwargs)
    return model


def resnet_fullDBN1202(**kwargs):
    model = ResNet_fullDBN(BasicBlock, [200, 200, 200], **kwargs)
    return model


def preact_resnet_fullDBN20(**kwargs):
    model = PreAct_ResNet_fullDBN(PreActBasicBlock, [3, 3, 3], **kwargs)
    return model


def preact_resnet_fullDBN110(**kwargs):
    model = PreAct_ResNet_fullDBN(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet_fullDBN164(**kwargs):
    model = PreAct_ResNet_fullDBN(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet_fullDBN1001(**kwargs):
    model = PreAct_ResNet_fullDBN(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = preact_resnet_fullDBN110()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())
