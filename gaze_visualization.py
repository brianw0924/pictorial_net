import os
import glob
import numpy as np
import cv2
import datetime
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

from utils import Set_seed, pitch_error, yaw_error
from models.mynet import Mynet 


'''

0. Remember opencv is xy-indexing
x: dim=1
y: dim=0
'''

def Detect_eye(frame, size, threshold=20000):
    '''
    Find the closest cropped area to target area
    Make sure (area - target area) < threshold
    '''
    eye_cascade = cv2.CascadeClassifier("./others/haarcascade_eye.xml")
    eye_cascade.load("./others/haarcascade_eye.xml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray) # will return list of [ex, ey, ew, eh]
    if len(eyes) == 0: return None # didn't find any box
    target_area = (float(size[0]) * float(size[1]))
    candidates = list(filter(lambda e: abs(e[0]*e[1] - target_area) < threshold, eyes))
    if len(candidates) == 0: return None # didn't find any box that meets the requirement

    return min(candidates, key=lambda e: abs(e[0]*e[1] - target_area)) # return the closest one

def Cropping(video_dir, file, size):
    '''
    Will return when find the first cropping box
    '''
    target_area = float(size[0]) * float(size[1])
    video = cv2.VideoCapture(os.path.join(video_dir,file))
    center_x, center_y = None, None # center of the cropping box
    if not video.isOpened():
        tqdm.write(f'Failed to capture video {file}.')
        return center_x, center_y
        
    success = True
    candidates = []
    f_shape = None
    while success:
        success, frame = video.read()
        if not success:
            break
        f_shape = frame.shape
        frame = frame[:,:frame.shape[1]//2,:] # only use right eye
        eye = Detect_eye(frame, size) # box: [ex, xy, ew, eh]
        if eye is not None:
        #     candidates.append(eye)
    # if len(candidates) == 0: return center_x, center_y
    # box = min(candidates, key=lambda e: abs(e[0]*e[1] - target_area))
    
            ex, ey, ew, eh = eye
            center_x, center_y = ex+ew//2, ey+eh//2
            if center_x < size[0]//2: center_x = size[0]//2
            elif center_x > f_shape[1] - size[0]//2: center_x = f_shape[1] - size[0]//2
            if center_y < size[1]//2: center_y = size[1]//2
            elif center_y > f_shape[0] - size[1]//2: center_y = f_shape[0] - size[1]//2
            break

    return center_x, center_y
    
def Inference_and_visualization(model, args, device):

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144,192)), # depends on your model input size
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # VIDEO PARAMETER
    fps = 24            # output video fps
    size = (384,288)    # output video frame size in (w, h)


    # ARROW PARAMETER
    color = (0, 255, 0)              # arrow color
    thickness = 4                    # arrow thickness
    scaling = 150                    # scaling the arrow length
    s = (size[0]//2, size[1]//2)     # arrow starting point

    for file in tqdm(os.listdir(args.video_dir)):
        starttime = datetime.datetime.now()
        tqdm.write(f'Inferencing: {file}')

        # Crop eye
        center_x, center_y = Cropping(args.video_dir, file, size)
        if center_x is None or center_y is None:
            print("Can't find satisfied cropping box.")
            continue
        tqdm.write(f"center x: {center_x}, center y: {center_y}")  

        # Run inference
        video = cv2.VideoCapture(os.path.join(args.video_dir,file))
        if not video.isOpened():
            print(f'Failed to capture video {file}.')
            continue

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        Writer = cv2.VideoWriter(os.path.join(args.outdir,file) ,fourcc,fps,size)
        success = True
        count = 0
        
        with open(os.path.join("./test_csv",f"{file.split('.')[0]}.csv"), 'w') as f:
            f.write("frame, pitch, yaw\n")
            while success:
                success, frame = video.read()
                if not success:
                    break
                count+=1
                frame = frame[:,:frame.shape[1]//2,:] # only use right eye
                frame = frame[center_y-size[1]//2:center_y+size[1]//2, center_x-size[0]//2:center_x+size[0]//2, :]
                image = torch.unsqueeze(tfm(frame),dim=0) # (bs, C, H, W)
                gaze_output= model(image.to(device))
                f.write(f'{count}, {gaze_output[:,0].item()}, {gaze_output[:,1].item()}\n')

                x = torch.sin(gaze_output[:,1]*np.pi/180)
                y = torch.sin(gaze_output[:,0]*np.pi/180)
                t = (int(s[0]+scaling*x.item()),int(s[1]+scaling*y.item())) # arrow ending point
                save_image = cv2.arrowedLine(frame, s, t,color, thickness)
                Writer.write(save_image)

        endtime = datetime.datetime.now()
        tqdm.write(f'Spent time: {(endtime - starttime).seconds} | Frame count: {count}')





def main():

    '''
    video_dir
            |___video1.mp4
            |___video2.mp4
            ...
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
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
    # model.eval() # 開 eval 在影像模糊的時候有可能爛掉
    Inference_and_visualization(model,args, device)

                
if __name__ == '__main__':
    main()


