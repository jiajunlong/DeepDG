import argparse

import torch
import torch.nn as nn
import extension.normalization as myNorm

__all__ = [
    'vgg',
]


class VGG(nn.Module):

    def __init__(self, features, replace_norm, init_weights=True):
        super(VGG, self).__init__()
        self.replace_norm = replace_norm
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.avgpool = nn.AvgPool2d(2, 2)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, num_classes),
        # )
        self.in_features = 512 * 7 * 7
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # def _replace_norm(self):
    #     for i in self.replace_norm['layer']:
    #         layer = 'layer' + str(i)
    #         for j in range(self.layers[int(i)-1]):
    #             for k in self.replace_norm['norm_num']:
    #                 num = str(j)
    #                 bn = 'bn' + str(k)
    #                 pre_block = getattr(getattr(self, layer), num)
    #                 setattr(pre_block, bn, myNorm.Norm(getattr(pre_block, bn).num_features))


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, myNorm.Norm(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vgg(args, **kwargs):
    model = VGG(make_layers(cfg['E']), args.replace_norm, **kwargs)
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='test normalization')
    myNorm.add_arguments(parse)
    args = parse.parse_args()
    args.norm = 'BN'
    myNorm.setting(args)
    x = torch.randn(4, 3, 224, 224)
    print(vgg(args)(x))
