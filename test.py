import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock


class MyResNet(ResNet):
    def __init__(self, replace_norm, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)
        self.replace_norm = replace_norm
        self._replace_norm()

    def _replace_norm(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for replace_layer in self.replace_norm:
                    if name == replace_layer["name"]:
                        if replace_layer["type"] == "group_norm":
                            num_groups = replace_layer["num_groups"]
                            new_norm = nn.GroupNorm(num_groups, module.num_features, affine=True)
                        elif replace_layer["type"] == "instance_norm":
                            new_norm = nn.InstanceNorm2d(module.num_features, affine=True)
                        else:
                            raise ValueError(f"Unknown normalization type: {replace_layer['type']}")
                        setattr(self, name, new_norm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-layers", nargs="+", type=str, default=[], help="Replace batch norm layers")
    args = parser.parse_args()

    replace_norm = []
    for replace_layer_str in args.replace_layers:
        replace_layer = replace_layer_str.split(":")
        assert len(replace_layer) == 2, "replace-layers should be in the format 'layer_name:norm_type'"
        replace_norm.append({"name": replace_layer[0], "type": replace_layer[1]})

    # create model with replaced normalization layers
    model = MyResNet(replace_norm, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
    print(model)

