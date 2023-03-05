from network.resnet import resnet18
from network.resnet import resnet34
from network.resnet import resnet50
from network.resnet import resnet101
from network.resnet import resnet152
from network.vgg import vgg


def get_network(args):
    if args.net not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(args.net))
    return globals()[args.net](args)



