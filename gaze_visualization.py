import os
import glob
import numpy as np
import cv2
import time
import json
import importlib
import logging
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from utils import Set_seed
from models.mynet import Mynet 

def inference_and_visualization(model,video_dir,args):

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144,192)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    color = (0, 255, 0) # arrow color
    thickness = 4       # arrow thickness
    s = (72,96)         # start point of the arrow
    scaling = 150       # scaling the arrow length

    fps = 24            # output video fps
    size = (192,144)    # output video size in (w, h)

    for file in tqdm(os.listdir(video_dir)):

        video = cv2.VideoCapture(os.path.join(video_dir,file))
        if not video.isOpened():
            print(f'Failed to capture video {file}.')
            continue
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        Writer = cv2.VideoWriter(os.path.join(args.outdir,file) ,fourcc,fps,size)
        success = True
        while success:
            success, frame = video.read()
            if not success:
                break
            frame = frame[:,:frame.shape[1]//2,:] # only use right eye

            '''
            frame cropping need to do it by yourself currently ...
            '''
            frame = frame[140:284,120:312,:] # cropping
            image = torch.unsqueeze(tfm(frame),dim=0)

            output = model(image.cuda())
            x = torch.sin(output[:,1]*np.pi/180)
            y = torch.sin(output[:,0]*np.pi/180)
            t = (int(s[0]+scaling*x.item()),int(s[1]+scaling*y.item()))
            save_image = cv2.arrowedLine(frame, s, t,color, thickness)
            Writer.write(save_image)



def main():

    '''
    video_dir
            |___video1.mp4
            |___video2.mp4
            ...
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--video_dir', type=str, default="./test_video")
    args = parser.parse_args()

    Set_seed(args.seed)

    # model & loss function
    model = Mynet()
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.to(device)

    inference_and_visualization(model,args.video_dir,args)

                
if __name__ == '__main__':
    main()


