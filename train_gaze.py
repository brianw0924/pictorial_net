import os
import time
import json
import importlib
import argparse
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import cv2
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
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from utils import Set_seed, vec2angle, pitch_error, yaw_error, update_log_gaze
from dataloader import Gaze_loader
from models.mynet import Gaze_Net

'''
[notes]

* Don't use SGDM, output will get nan
* Before Training, Remember to confirm:
    0. model ?
    1. args.out_dir ?
    2. args.dataset ?
    3. args.cross_subject ?


args.data_dir/
   ├─image/
   |   ├─0000000.png
   |   ├─0000001.png
   |   ...
   └─gaze/
       └─gaze.txt

'''

def parse_args():
    parser = argparse.ArgumentParser()
    ''' Paths '''
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--data_dir', type=str, default="/home/brianw0924/Desktop/Neurobit/dataset_nocrop")
    parser.add_argument('--dataset', type=str, default="Neurobit", choices=["Neurobit", "TEyeD"]) # if you set Neurobit, will do random cropping
    parser.add_argument('--out_dir', type=str, default='./result')
    
    ''' Load pretrain '''
    # parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default='pretrain.pth?dl=1')
    
    ''' How to split dataset '''
    parser.add_argument('--cross_subject', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.8)
    
    ''' paramters '''
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=1024*100)
    parser.add_argument('--val_steps', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=128)
    parser.add_argument('--earlystop', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.3)

    args = parser.parse_args()

    assert os.path.exists(args.data_dir)

    return args

def validation(args, step, model, val_loader, device, log):
    model.eval()

    val_pitch_error = 0
    val_yaw_error   = 0
    acc = 0

    count = 0
    for images, gazes in val_loader:
        count+=gazes.shape[0]

        # save_image(images,"preview.png", nrow=8)
        # print(vec2angle(gazes))
        # print(images.shape)
        # break
    
        with torch.no_grad():
            pred = model(images.to(device))

        val_pitch_error  += (pitch_error(pred, vec2angle(gazes).to(device)).item())
        val_yaw_error    += (yaw_error(pred, vec2angle(gazes).to(device)).item())

    val_pitch_error /= count
    val_yaw_error   /= count

    tqdm.write( 'Validation  | '
                'Step {} | '
                'Pitch error (MAE) {:.2f} | '
                'Yaw error (MAE) {:2f}'.format(
                    str(step).zfill(6),
                    val_pitch_error,
                    val_yaw_error,
                ))
    update_log_gaze(args, log, "Validation", step, val_pitch_error, val_yaw_error)

    return val_pitch_error, val_yaw_error


def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device, log):
    
    model.train()

    train_pitch_error   = 0
    train_yaw_error     = 0

    train_iterator = iter(train_loader)
    count = 0
    patience = 0
    error = 1e10
    for step in tqdm(range(args.total_steps)):
        try:
            images, gazes = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, gazes = next(train_iterator)    

        # save_image(images,"preview.png", nrow=8)
        # print(vec2angle(gazes))
        # print(images.shape)
        # break

        count+= images.shape[0]

        pred = model(images.to(device))
        loss = criterion(pred, vec2angle(gazes).to(device)) / args.accumulate
        loss.backward()

        train_pitch_error   += pitch_error(pred, vec2angle(gazes).to(device)).item()
        train_yaw_error     += yaw_error(pred, vec2angle(gazes).to(device)).item() 

        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:
            
            # TRAINING RECORD
            train_pitch_error   /= count
            train_yaw_error     /= count

            tqdm.write( 'Train       | '
                        'Step {} | '
                        'Pitch error (MAE) {:2f} | '
                        'Yaw error (MAE) {:2f} | '.format(
                            str(step+1).zfill(6),
                            train_pitch_error,
                            train_yaw_error,
                        ))
    
            update_log_gaze(args, log, "Train", step+1, train_pitch_error, train_yaw_error)

            #VALIDATION
            val_pitch_error, val_yaw_error = validation(
                args, step+1, model, val_loader, device, log
            )


            # SAVE MODEL
            if (val_pitch_error + val_yaw_error) / 2 < error:
                tqdm.write(f"Save model at step {str(step+1).zfill(6)}")
                torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_state.pth'))
                error = (val_pitch_error + val_yaw_error)/2
                patience = 0
            else:
                patience+=1

            train_pitch_error   = 0
            train_yaw_error     = 0
            count               = 0

            model.train()

            if patience == args.earlystop: break


def main():

    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # TRAIN LOG
    log = {
        "Train":{},
        "Validation":{}
    }

    # ARGUMENTS
    args = parse_args()
    print(args)

    # RANDOM SEED
    Set_seed(args.seed)

    # OUTPUT DIRECTORY
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # RECORD ARGS
    args_json = os.path.join(args.out_dir, 'args.json')
    with open(args_json, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data_dir (TEyeD)
    print("Preparing data ...")
    train_loader, val_loader = Gaze_loader(args)

    # MODEL & LOSS
    model = Gaze_Net()

    # LOAD PRE-TRAIN MODEL
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))

    model.to(device)
    # print(model)

    # LOSS FUNCTION
    criterion = nn.MSELoss()

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    # SCHEDULER
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.total_steps-args.warmup_steps)/args.accumulate)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_steps//args.accumulate, after_scheduler=scheduler)
    optimizer.zero_grad()
    optimizer.step()

    # TRAINING
    train(args, model, optimizer, scheduler_warmup, criterion, train_loader, val_loader, device, log)

if __name__ == '__main__':
    main()
