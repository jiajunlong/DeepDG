import torch.nn as nn
import extension as my

__all__ = ['simple']


class Simple(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple, self).__init__()
        self.net = my.layers.NamedSequential(
            conv1=my.NormConv(3, 32, 5, stride=1, padding=2, bias=False, adjustScale=False),
            pool1=nn.MaxPool2d(3, 2, 1),

            norm2=my.Norm(32),
            relu2=nn.ReLU(True),
            conv2=my.NormConv(32, 32, 5, stride=1, padding=2, bias=False),
            pool2=nn.MaxPool2d(3, 2, 1),

            norm3=my.Norm(32),
            relu3=nn.ReLU(True),
            conv3=nn.Conv2d(32, 64, 5, 1, 2, bias=False),

            norm4=my.Norm(64),
            relu4=nn.ReLU(True),
            pool3=nn.AvgPool2d(3, 2, 1),
            view=my.View(64 * 4 * 4),
            #fc=nn.Linear(64 * 4 * 4, num_classes)
            fc=my.normalization.normalization.Pearson_Linear(64 * 4 * 4, num_classes)
            )

    def forward(self, x):
        return self.net(x)


def simple(**kwargs):
    return Simple(**kwargs)
