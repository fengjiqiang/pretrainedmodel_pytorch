import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import act_fn, print_values


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


"""************* Original PNN Implementation ****************"""

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes),  #TODO paper does not use it!
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            self.noise = (2 * self.noise - 1) * self.level
        y = torch.add(x, self.noise)
        return self.layers(y)   #input, perturb, relu, batchnorm, conv1x1


class NoiseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),  #TODO paper does not use it!
            NoiseLayer(planes, planes, level),  #perturb, relu, conv1x1
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nfilters, nclasses, pool, level, first_filter_size=3):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        if first_filter_size == 7:
            pool = 1
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        elif first_filter_size == 3:
            pool = 4
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
            )
        elif first_filter_size == 0:
            print('\n\nThe original noiseresnet18 model does not support noise masks in the first layer, '
                  'use perturb_resnet18 model, or set first_filter_size to 3 or 7\n\n')
            return

        self.pre_layers[0].weight.requires_grad = False # (5) Felix added this, first layer rand conv
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, filter_size=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8



def noiseresnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=7,
                  pool_type=None, input_size=None, scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False,
                  debug=False, noise_type='uniform', train_masks=False, mix_maps=None):
    return NoiseResNet(NoiseBasicBlock, [2, 2, 2, 2], nfilters=nfilters, pool=avgpool, nclasses=nclasses,
                       level=level, first_filter_size=first_filter_size)
