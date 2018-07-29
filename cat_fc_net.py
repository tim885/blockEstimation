# script for test of network with parallel FC layers

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def main():
    model = models.__dict__['resnet18'](pretrained=True)
    # remove FC layer and store in nn.Sequential container
    model = nn.Sequential(*list(model.children())[:-1])


if __name__ == '__main__':
    main()



