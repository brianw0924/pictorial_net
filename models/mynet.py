
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.models as models

from .unet import UNet


class Gaze_Net(nn.Module):
    def __init__(self):
        super(Gaze_Net, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=1, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

class Eye_Localization_Net(nn.Module):
    def __init__(self):
        super(Eye_Localization_Net, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=1, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=6, bias=True)

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

class Lid_2point(nn.Module):
    def __init__(self):
        super(Lid_2point, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=1, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

if __name__ == "__main__":
    net = Gaze_Net()
    print(net)
