
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.models as models

from .unet import UNet, UNet_3output

class Gaze_Net(nn.Module):
    '''
    Input shape: (C,H,W) == (3, 144, 192)
    Output: (yaw, pitch)
    '''
    def __init__(self):
        super(Gaze_Net, self).__init__()
        self.UNet = UNet(in_channels=3, out_channels=3, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        # self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

class Center_Net(nn.Module):
    def __init__(self):
        super(Center_Net, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=1, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=6, bias=True)

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

class Landmark_Net(nn.Module):
    def __init__(self):
        super(Landmark_Net, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=1, num=16, bilinear=True)
        
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        self.vgg16_bn.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=68, bias=True)
        '''
        Pupil: 8*2
        Iris : 8*2
        Lid  : 34*2
        '''    

    def forward(self, x):
        y = self.UNet(x)
        y = self.vgg16_bn(y)
        return y

class Seg_Net(nn.Module):
    '''
    output: 4 classes
    0: background
    1: lid
    2: iris
    3: pupil
    '''
    def __init__(self):
        super(Seg_Net, self).__init__()
        self.UNet = UNet(in_channels=1, out_channels=2, num=16, bilinear=True)
        # self.UNet = UNet_3output(in_channels=1, out_channels=2, num=16, bilinear=True)


    def forward(self, x):
        return self.UNet(x)

if __name__ == "__main__":
    net = Gaze_Net()
    print(net)
