
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.models as models

from .unet import UNet


class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=64, bilinear=True)
        # self.resnet = models.resnet50(num_classes=2)
        # self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
       
        self.vgg16_bn = models.vgg16_bn(num_classes=2)
        self.vgg16_bn.features[0] = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 

    def forward(self, x):
        y = self.UNet(x)
        # y = self.resnet(y)
        y = self.vgg16_bn(y)
        return y

if __name__ == "__main__":
    net = Mynet()
    print(net)
