import torch
import torch.nn as nn
__all__ = ['vgg19']

import math

import torch.nn as nn
import torch.nn.init as init

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, num_classes, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}




def vgg19(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
    elif dataset == 'cifar10':
        num_classes = num_classes or 10

    """VGG 19-layer model (configuration "E")"""
    return VGG(num_classes, make_layers(cfg['D']))

# class VGG(nn.Module):
#     def __init__(self, vgg_name, num_classes=10):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, num_classes)
#         self.regime = [
#             {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
#              'weight_decay': 1e-4, 'momentum': 0.9},
#             {'epoch': 81, 'lr': 1e-2},
#             {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
#             {'epoch': 164, 'lr': 1e-4}
#         ]
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)
#
#
#
#
# def vgg11(**kwargs):
#     num_classes, dataset = map(
#         kwargs.get, ['num_classes', 'dataset'])
#     if dataset == 'imagenet':
#         num_classes = num_classes or 1000
#     elif dataset == 'cifar10':
#         num_classes = num_classes or 10
#
#     return VGG('VGG11', num_classes)
#
# def vgg13(**kwargs):
#     num_classes, dataset = map(
#         kwargs.get, ['num_classes', 'dataset'])
#     if dataset == 'imagenet':
#         num_classes = num_classes or 1000
#     elif dataset == 'cifar10':
#         num_classes = num_classes or 10
#
#     return VGG('VGG13', num_classes)
#
# def vgg16(**kwargs):
#     num_classes, dataset = map(
#         kwargs.get, ['num_classes', 'dataset'])
#     if dataset == 'imagenet':
#         num_classes = num_classes or 1000
#     elif dataset == 'cifar10':
#         num_classes = num_classes or 10
#
#     return VGG('VGG16', num_classes)
#
# def vgg19(**kwargs):
#
#     num_classes, dataset = map(
#         kwargs.get, ['num_classes', 'dataset'])
#     if dataset == 'imagenet':
#         num_classes = num_classes or 1000
#     elif dataset == 'cifar10':
#         num_classes = num_classes or 10
#
#     return VGG('VGG19', num_classes)

