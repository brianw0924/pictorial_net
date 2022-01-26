import os
import glob
import numpy as np
import cv2
import datetime
import time
import json
import logging
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models as models
from torchvision.utils import save_image
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from utils import Set_seed, pitch_error, yaw_error
from models.mynet import Gaze_Net


'''
0. Remember opencv is xy-indexing
x: dim=1, i.e. column
y: dim=0, i.e. row
'''

def parser_args():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--gaze_model_path', type=str, default="./result/Neurobit/UNet16_vgg16bn_NOpretrain_aug_mixed/model_state.pth")
    parser.add_argument('--valid_model_path', type=str, default="./result/detect_eye_open/vgg19_bn_weighted06/model_state.pth")
    parser.add_argument('--video_dir', type=str, default="./test_video")        # input
    parser.add_argument('--output_video', type=str, default='./test_result')    # output video
    parser.add_argument('--output_csv', type=str, default='./test_csv')         # output csv

    # parameter
    parser.add_argument('--Lefteye_ROI', type=tuple, default=(0, 210, 800, 1080))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--Righteye_ROI', type=tuple, default=(0, 210, 160, 440))     # (x, y) <=> (dim=1, dim=0) <=> (w, h)
    parser.add_argument('--fps', type=int, default=210)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--threshold', type=float, default=0.5) # threshold for eye validity
    args = parser.parse_args()

    if not os.path.exists(args.output_video):
        os.mkdir(args.output_video)
    if not os.path.exists(args.output_csv):
        os.mkdir(args.output_csv)

    return args

def Inference_and_visualization(args, gaze_model, valid_model, device):
    sigmoid = nn.Sigmoid()

    tfm = transforms.Compose([
        transforms.Resize((144,192)), # depends on your model input size
        transforms.ToTensor(),
    ])

    # Bounding box
    L_top, L_bot, L_left, L_right = args.Lefteye_ROI[0], args.Lefteye_ROI[1], args.Lefteye_ROI[2], args.Lefteye_ROI[3]
    R_top, R_bot, R_left, R_right = args.Righteye_ROI[0], args.Righteye_ROI[1], args.Righteye_ROI[2], args.Righteye_ROI[3]
    
    assert(L_top<=L_bot and L_left<=L_right and R_top<=R_bot and R_left<=R_right)
    assert((L_bot-L_top) == (R_bot-R_top))
    assert((L_right-L_left) == (R_right-R_left))
    
    # Video size
    crop_size = (L_right-L_left, L_bot-L_top) # (w, h)
    merge_size = (crop_size[0]*2, crop_size[1])

    # Arrow parameter
    color = (0, 255, 0)                     # arrow color
    thickness = 4                           # arrow thickness
    scaling = 150                           # scaling the arrow length
    s = (crop_size[0]//2, crop_size[1]//2)  # arrow starting point

    # Iterate thru videos
    for file in tqdm(sorted(os.listdir(args.video_dir))):
        starttime = datetime.datetime.now()
        tqdm.write(f'Inferencing: {file}')

        # Check if video is broken
        video = cv2.VideoCapture(os.path.join(args.video_dir,file))
        if not video.isOpened():
            print(f'Failed to capture video {file}.')
            continue
        
        # Video parameter
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # LeftEye_Writer  = cv2.VideoWriter(os.path.join(args.output_video,f"{file.split('.')[0]}_left.mp4") , fourcc, args.fps, crop_size)
        # RightEye_Writer = cv2.VideoWriter(os.path.join(args.output_video,f"{file.split('.')[0]}_right.mp4") , fourcc, args.fps, crop_size)
        Writer = cv2.VideoWriter(os.path.join(args.output_video,file) , fourcc, args.fps, merge_size)

        # Inference
        success = True
        count = 0
        with open(os.path.join(args.output_csv,f"{file.split('.')[0]}_left.csv"), 'w') as lf:
            with open(os.path.join(args.output_csv,f"{file.split('.')[0]}_right.csv"), 'w') as rf:
                lf.write("yaw,pitch,valid\n")
                rf.write("yaw,pitch,valid\n")
                while success:
                    success, frame = video.read()
                    if not success:
                        break
                    count+=1
                    left_eye, right_eye = frame[L_top:L_bot, L_left:L_right, :], frame[R_top:R_bot, R_left:R_right, :]
                    
                    # Model input
                    images = torch.cat([
                            torch.unsqueeze(tfm(Image.fromarray(left_eye).convert('RGB')),dim=0),
                            torch.unsqueeze(tfm(Image.fromarray(right_eye).convert('RGB')),dim=0)
                        ], dim=0
                    )

                    # Model inference
                    with torch.no_grad():
                        pred = gaze_model(images.to(device))
                        pred_valid = sigmoid(valid_model(images.to(device)))

                    pred_valid[pred_valid>args.threshold]  = 1
                    pred_valid[pred_valid<=args.threshold] = 0

                    lf.write(f"{pred[0,0]},{pred[0,1]},{pred_valid[0]}\n")
                    rf.write(f"{pred[1,0]},{pred[1,1]},{pred_valid[1]}\n")

                    # Project to x-y
                    lx = - torch.sin(pred[0,0]*np.pi/180)
                    ly = - torch.sin(pred[0,1]*np.pi/180)
                    rx = - torch.sin(pred[1,0]*np.pi/180)
                    ry = - torch.sin(pred[1,1]*np.pi/180)
                    lt = (int(s[0] + scaling * lx.item()), int(s[1] + scaling * ly.item())) # arrow ending point
                    rt = (int(s[0] + scaling * rx.item()), int(s[1] + scaling * ry.item())) # arrow ending point
                    
                    # Draw arrow
                    left_eye  = cv2.arrowedLine(left_eye,  s, lt, color, thickness)
                    right_eye = cv2.arrowedLine(right_eye, s, rt, color, thickness)

                    # Merge two eyes to one video
                    two_eye = np.concatenate((right_eye, left_eye), axis=1)

                    # LeftEye_Writer.write(left_eye)
                    # RightEye_Writer.write(right_eye)
                    Writer.write(two_eye)

            endtime = datetime.datetime.now()
            tqdm.write(f'Spent time: {(endtime - starttime).seconds} | Frame count: {count}')





def main():

    '''
    video_dir/
        ├─video1.mp4
        ├─video2.mp4
        ├─ ...
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'device: {device}')

    # arguments
    args = parser_args()

    # random seed
    Set_seed(args.seed)

    # Gaze model
    gaze_model = Gaze_Net()
    gaze_model.load_state_dict(torch.load(args.gaze_model_path))
    gaze_model.to(device)
    gaze_model.eval()

    # Valid model
    valid_model = models.vgg19_bn()
    valid_model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
    valid_model.load_state_dict(torch.load(args.valid_model_path))
    valid_model.to(device)
    valid_model.eval()

    # inference
    Inference_and_visualization(args, gaze_model, valid_model, device)

                
if __name__ == '__main__':
    main()


