import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


__all__ = ['BatchNormDebug']

class BatchNormDebug(nn.Module):
    def __init__(self, num_features, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        """"""
        super(BatchNormDebug, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            #self.weight = Parameter(torch.Tensor(*self.shape))
            #self.bias = Parameter(torch.Tensor(*self.shape))
            self.weight = Parameter(torch.Tensor(self.num_features))
            self.bias = Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(50400))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        assert input.dim() == self.dim and input.size(1) == self.num_features
        # always use the training mode of BN
        output = F.batch_norm(input, self.running_mean, self.running_var, training=self.training, momentum=self.momentum,
                              eps=self.eps)
        output = output.view_as(input)
        if not self.training:
            print('-------test------calculate mean and variance---')
            x = output.transpose(0,1).contiguous().view(self.num_features,-1)
            self.NormDistribution_mean=x.data.mean(-1, keepdim=True)
            #xc = x-self.debug_mean
            self.NormDistribtion_std = x.data.std(dim=-1)
            self.Dis_mean = self.NormDistribution_mean.norm(p='fro')
            std_1=self.NormDistribtion_std.clone().fill_(1)
            self.Dis_std = torch.norm(self.NormDistribtion_std-std_1, p='fro')
            print('---BNDebug: Dis_mean:', self.Dis_mean, '--Dis_std:', self.Dis_std)
        if self.affine:
            output = output * self.weight.view(*self.shape) + self.bias.view(*self.shape)
        return output

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    GBN = BatchNormDebug(4, momentum=0.1,eps=0, affine=True)
    print(GBN)
    # print(GBN.weight)
    # print(GBN.bias)
    x = torch.randn(20, 4, 1, 1) * 2 + 1
    print('x mean = {}, var = {}'.format(x.mean(), x.var()))
    y = GBN(x)
    GBN.eval()
    print('y size = {}, mean = {}, var = {}'.format(y.size(), y.mean(0), y.var(0)))
    y_2=GBN(x)
    print('y2 size = {}, mean = {}, var = {}'.format(y_2.size(), y_2.mean(0), y_2.var(0)))
    #print(GBN.running_mean, GBN.running_var)
