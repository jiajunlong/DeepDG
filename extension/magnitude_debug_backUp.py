import matplotlib.pyplot as plt
import numpy as np
import torch


class MagnitudeDebug(object):
    _saved = []

    def __init__(self, name="", step=1, forward=True):
        self.name = name
        self.values = []
        name += '_f' if forward else '_b'
        self._saved.append((name, self.values))
        self.it = 0
        self.step = step
        self.forward = forward
        assert self.step > 0

    def __call__(self, m, inputs, outputs):
        self.it += 1
        if self.it % self.step != 0:
            return
        #import pdb
        #pdb.set_trace()
        x = inputs[0] if self.forward else outputs[0].size(0) * outputs[0] # note that the gradients are averaged by batch size in PyTorch
        # x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        if x is None:
            return


        print(self.name, x.size())
        # self.values.append(x.abs().max().item())
        # x = x.abs()
        # print('{}: max = {}, mean = {}'.format(self.name, x.max(), x.mean()))

        xm = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        N_examples = xm.size(1)

        xx = xm.matmul(xm.t()) / N_examples

        # print(xx.size())
        eig, _ = xx.eig()
        #print('{}: eigenvalue max = {}, mean = {}, min = {}'.format(self.name, eig.max(), eig.mean(), eig.min()))
        self.values.append(eig.max().item())
        #print(self._saved)

    @staticmethod
    def reset():
        MagnitudeDebug._saved.clear()

    @staticmethod
    def get():
        return MagnitudeDebug._saved

    @staticmethod
    def display():
        print('\n-------------Magnitude Debug Display---------------------------')
        print(MagnitudeDebug._saved.__len__())
        for name, values in MagnitudeDebug._saved:
            print(name)
            plt.plot(values, label=name)
            plt.title(name)
            plt.show()
        return

    @staticmethod
    def save_as_csv(filename=None):
        if filename is None:
            print('No filename, can not save')
            return
        header = []
        values = []
        for name, value in MagnitudeDebug._saved:
            if not value:
                continue
            header.append(name)
            values.append(value)
        values = np.array(values, dtype=np.float).T
        np.savetxt(filename, values, fmt='%.8f', delimiter=',', header=','.join(header))
        return
