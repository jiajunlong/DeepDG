import matplotlib.pyplot as plt
import numpy as np
import torch
import extension as ext
import argparse
import math

class MagnitudeDebug(object):
    _saved = []
    _isOn = True # the switch in case to close the debug, \eg, during test
    def __init__(self, cfg: argparse.Namespace, name="", step=1, forward=True):
        self.name = name
        self.values_ME = []
        self.values_Norm = []
        self.values_CN = []
        self.values_CN_50 = []
        self.values_CN_80 = []
        name += '_f' if forward else '_b'
        self._saved.append((name + '_ME', self.values_ME))
        self._saved.append((name + '_Norm', self.values_Norm))
        self._saved.append((name + '_CN', self.values_CN))
        self._saved.append((name + '_CN_50', self.values_CN_50))
        self._saved.append((name + '_CN_80', self.values_CN_80))
        self.it = 0
        self.eps = 1e-7 # for numerical stability
        self.step = step
        self.forward = forward
        self.vis = ext.visualization_logScale.setting(cfg, name, {
            "ME": "maximum_eig",
            "Norm": "F2Norm",
            "CN": "condition_number",
            "CN_50": "condition_number_50",
            "CN_80": "condition_number_80",
        })
        assert self.step > 0

    def __call__(self, m, inputs, outputs):
        if not self._isOn:
            return
        self.it += 1
        if self.it % self.step != 0:
            return
        #import pdb
        #pdb.set_trace()
        x = inputs[0] if self.forward else outputs[0].size(0) * outputs[0] # note that the gradients are averaged by batch size in PyTorch
        # x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        if x is None:
            return


        ## For conditioning analysis

        #print(self.name, x.size())
        # self.values.append(x.abs().max().item())
        # x = x.abs()
        # print('{}: max = {}, mean = {}'.format(self.name, x.max(), x.mean()))

        xm = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        N_examples = xm.size(1)


        x_norm = xm.norm().item()/math.sqrt(N_examples)

        self.values_Norm.append(x_norm)
        self.vis.add_value("Norm", x_norm)


        xx = xm.matmul(xm.t()) / N_examples
        # print(xx.size())
        eig, _ = xx.symeig()
        eig_max=eig.max().item()
        self.values_ME.append(eig_max)
        self.vis.add_value("ME", eig_max)

        index_e_50 = int(eig.numel() * 0.5)
        index_e_80 = eig.numel() - round(eig.numel() * 0.8)

        cn = eig_max / (eig.abs().min().item() + self.eps)
        cn_50 = eig_max / (abs(eig[index_e_50].item()) + self.eps)
        cn_80 = eig_max / (abs(eig[index_e_80].item()) + self.eps)

        #print('{}: eigenvalue max = {}, mean = {}, min = {}'.format(self.name, eig.max(), eig.mean(), eig.min()))
        self.values_CN.append(cn)
        self.values_CN_50.append(cn_50)
        self.values_CN_80.append(cn_80)
        self.vis.add_value("CN", cn)
        self.vis.add_value("CN_50", cn_50)
        self.vis.add_value("CN_80", cn_80)
        #print(self._saved)

    @staticmethod
    def reset():
        MagnitudeDebug._saved.clear()

    @staticmethod
    def get():
        return MagnitudeDebug._saved

    @staticmethod
    def closeDebug():
        MagnitudeDebug._isOn = False
        return

    @staticmethod
    def openDebug():
        MagnitudeDebug._isOn = True
        return

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
        #import pdb
        #pdb.set_trace()
        values = np.array(values, dtype=np.float).T
        np.savetxt(filename, values, fmt='%.8f', delimiter=',', header=','.join(header))
        return
