import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GroupNorm_NLP(nn.Module):
    def __init__(self, num_features, num_groups=4, num_channels=0, dim=3, eps=1e-5, frozen=False, affine=True, *args, **kwargs):
        super(GroupNorm_NLP, self).__init__()
        self.num_features = num_features
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups>num_features:
            self.num_groups=num_features
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.shape = [1] * 3
        self.shape[2] = num_features

        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == 3 and size[2] == self.num_features
        x = input.reshape(size[0]*size[1]*self.num_groups, size[2] // self.num_groups)

        mean = x.mean(-1, keepdim=True)
        xc = x - mean
        std = xc.std(-1, keepdim=True) + self.eps
        xn = xc/std

        output = xn.reshape_as(input)
        #output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, groups={num_groups}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    groupNumber=2
    cn = GroupNorm_NLP(16,num_groups=groupNumber, affine=True)
    print(cn)
    x = torch.randn(4, 2, 16)
    #cn.eval()
    y = cn(x)
    zy = y.view(y.size(0)*y.size(1)*groupNumber, y.size(2)//groupNumber)
    print(zy.std(-1))
    print(zy.mean(-1))
