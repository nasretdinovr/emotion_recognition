import torch
import torch.nn as nn
import models.resnet as resnet
import torch.nn.functional as F
from pytorch.common.losses import *
from collections import OrderedDict


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Tanh10(nn.Module):
    def forward(self, x):
        x = nn.Tanh()(x.mul(0.05))
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=[], pretrain_weight=None):
        super(ResNet, self).__init__()
        sample_size = data_size['width']
        sample_duration = data_size['depth']

        if depth == 34:
            pretrained_net = resnet.resnet34(sample_size=sample_size, sample_duration=sample_duration)
        elif depth == 50:
            pretrained_net = resnet.resnet50(sample_size=sample_size, sample_duration=sample_duration)
        elif depth == 101:
            pretrained_net = resnet.resnet101(sample_size=sample_size, sample_duration=sample_duration)
        else:
            pretrained_net = resnet.resnet18(sample_size=sample_size, sample_duration=sample_duration)
        num_ftrs = 9*pretrained_net.fc.in_features

        if not pretrain_weight is None:
            try:
                state_dict = torch.load(pretrain_weight)['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                pretrained_net.load_state_dict(new_state_dict)
            except:
                pass

        modules = nn.Sequential()
        modules.add_module('Flatten', Flatten())
        modules.add_module('pr0', nn.ReLU())
        modules.add_module('fc1', nn.Linear(num_ftrs, 1024, bias=True))
        modules.add_module('pr1', nn.ReLU())
        #modules.add_module('dp1', nn.Dropout())
        #modules.add_module('fc2', NormalizedLinear(1024, num_classes))
        modules.add_module('fc2', nn.Linear(1024, num_classes, bias=True))
        #modules.add_module('dp2', nn.Dropout())

        # init by xavier
        modules.apply(weights_init)

        pretrained_net.fc = modules
        self.net = FeatureExtractor(pretrained_net, emb_name)

    def forward(self, data):
        # tensor size: [batch_size, 3, depth, crop_size, crop_size]
        output = self.net(data)
        return output

