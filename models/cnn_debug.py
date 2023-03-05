import torch.nn as nn
import extension as my

__all__ = ['cnn_debug_d1w16', 'cnn_debug_d2w16', 'cnn_debug_d3w16', 'cnn_debug_d4w16','cnn_debug_d5w16', 'cnn_debug_d2w32', 'cnn_debug_d3w32', 'cnn_debug_d4w32', 'cnn_debug_d2w64', 'cnn_debug_d3w64', 'cnn_debug_d4w64']


class Cnn_debug(nn.Module):
    def __init__(self, num_classes=10, depth=4, width=16, **kwargs):
        super(Cnn_debug, self).__init__()
        layers = [nn.Conv2d(3, width, kernel_size=3, stride=1, padding=0, bias=True), my.Norm(width), nn.ReLU(True)]
        for index in range(depth - 1):
            layers.append(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=0, bias=True))
            layers.append(my.Norm(width))
            layers.append(nn.ReLU(True))
        feature_size=32-2*depth
        layers.append(nn.AvgPool2d(feature_size, stride=1))
        layers.append(my.View(width))
        layers.append(nn.Linear(width,num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cnn_debug_d1w16(**kwargs):
    return Cnn_debug(depth=1, width=16, **kwargs)


def cnn_debug_d2w16(**kwargs):
    return Cnn_debug(depth=2, width=16, **kwargs)


def cnn_debug_d3w16(**kwargs):
    return Cnn_debug(depth=3, width=16, **kwargs)


def cnn_debug_d4w16(**kwargs):
    return Cnn_debug(depth=4, width=16, **kwargs)


def cnn_debug_d5w16(**kwargs):
    return Cnn_debug(depth=5, width=16, **kwargs)


def cnn_debug_d2w32(**kwargs):
    return Cnn_debug(depth=2, width=32, **kwargs)


def cnn_debug_d3w32(**kwargs):
    return Cnn_debug(depth=3, width=32, **kwargs)


def cnn_debug_d4w32(**kwargs):
    return Cnn_debug(depth=4, width=32, **kwargs)


def cnn_debug_d2w64(**kwargs):
    return Cnn_debug(depth=2, width=64, **kwargs)


def cnn_debug_d3w64(**kwargs):
    return Cnn_debug(depth=3, width=64, **kwargs)


def cnn_debug_d4w64(**kwargs):
    return Cnn_debug(depth=4, width=64 **kwargs)
