# coding=utf-8
import torch
from network import img_network
from network import networks


def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    else:
        net = networks.get_network(args)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total
