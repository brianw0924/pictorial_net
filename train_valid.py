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

from utils import Set_seed, vec2angle, pitch_error, yaw_error, update_log_valid
from dataloader import Valid_loader
# from models.mynet import Valid_Net

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
    parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    # parser.add_argument('--data_dir', type=str, default="/home/brianw0924/Desktop/Neurobit/dataset")
    parser.add_argument('--dataset', type=str, default="Neurobit", choices=["Neurobit", "TEyeD"]) # if you set Neurobit, will do random cropping
    parser.add_argument('--out_dir', type=str, default='./result/detect_eye_open/vgg19_bn_weighted02')

    ''' Load pretrain '''
    # parser.add_argument('--pretrain', type=str, default='./result/gaze/UNet16_vgg16bn_pretrain/model_state.pth')
    parser.add_argument('--pretrain', type=str, default=None)

    ''' How to split dataset '''
    parser.add_argument('--cross_subject', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.9)

    ''' paramters '''
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=1024*10)
    parser.add_argument('--val_steps', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=128)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    assert os.path.exists(args.data_dir)

    return args

def validation(args, step, model, val_loader, device, log):
    
    model.eval()

    sigmoid = nn.Sigmoid()

    sensitivity, specificity = 0, 0
    open_count, closed_count = 0, 0
    for images, valid in val_loader:
        open_count += (valid==1).float().sum()
        closed_count += (valid==0).float().sum()

        # tqdm.write(f"{open_count}, {closed_count}")
        with torch.no_grad():
            pred = sigmoid(torch.squeeze(model(images.to(device)),dim=1))

        pred[pred>args.threshold] = 1
        pred[pred<=args.threshold] = 0

        sensitivity += (pred[valid.to(device)==1] == valid[valid==1].to(device)).float().sum()
        specificity += (pred[valid.to(device)==0] == valid[valid==0].to(device)).float().sum()
    
    sensitivity = round((sensitivity/open_count).item(), 3)
    specificity = round((specificity/closed_count).item(), 3)

    # 仿 F-score
    val_score = (1+args.beta**2) * sensitivity * specificity / (args.beta**2 * sensitivity + specificity)

    # Save log
    update_log_valid(args, log, "Validation", step, sensitivity, specificity, val_score)

    # Print log
    tqdm.write( 'Validation  | '
                'Step {} | '
                'Sensitivity {} | '
                'Specificity {} | '
                'Score {}'.format(
                    str(step).zfill(6),
                    sensitivity,
                    specificity,
                    val_score
                ))

    return val_score


def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device, log):
    
    model.train()

    sigmoid = nn.Sigmoid()

    train_iterator = iter(train_loader)
    patience = 0
    best_score = 0
    sensitivity, specificity = 0, 0
    open_count, closed_count = 0, 0
    for step in tqdm(range(args.total_steps)):
        try:
            images, valid = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, valid = next(train_iterator)    

        # save_image(images,"preview.png", nrow=8)
        # print(images.shape)
        # break


        open_count += (valid==1).float().sum()
        closed_count += (valid==0).float().sum()

        # tqdm.write(f"{open_count}, {closed_count}")

        pred = torch.squeeze(model(images.to(device)),dim=1)
        loss = criterion(pred, valid.to(device)) / args.accumulate
        loss.backward()

        pred = sigmoid(pred)
        pred[pred>args.threshold] = 1
        pred[pred<=args.threshold] = 0

        sensitivity += (pred[valid.to(device)==1] == valid[valid==1].to(device)).float().sum()
        specificity += (pred[valid.to(device)==0] == valid[valid==0].to(device)).float().sum()

        # tqdm.write(f"{(pred[valid.to(device)==1] == valid[valid==1].to(device)).shape}")


        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:

            sensitivity = round((sensitivity/open_count).item(), 3)
            specificity = round((specificity/closed_count).item(), 3)

            # 仿 F-score
            train_score = (1+args.beta**2) * sensitivity * specificity / (args.beta**2 * sensitivity + specificity)
            
            # Save log
            update_log_valid(args, log, "Train", step+1, sensitivity, specificity, train_score)

            # Print log
            tqdm.write( 'Train       | '
                        'Step {} | '
                        'Sensitivity {} | '
                        'Specificity {} | '
                        'Score {}'.format(
                            str(step+1).zfill(6),
                            sensitivity,
                            specificity,
                            train_score
                        ))
    

            # Validation
            val_score = validation(
                args, step+1, model, val_loader, device, log
            )

            # Save model
            if val_score > best_score:
                tqdm.write(f"Save model at step {str(step+1).zfill(6)}")
                torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_state.pth'))
                best_score = val_score
                patience = 0
            else:
                patience+=1

            sensitivity, specificity = 0, 0
            open_count, closed_count = 0, 0
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
    train_loader, val_loader = Valid_loader(args)

    # MODEL & LOSS
    model = models.vgg19_bn(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)

    # LOAD PRE-TRAIN MODEL
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))

    model.to(device)
    # print(model)

    # LOSS FUNCTION
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2]).to(device))

    # OPTIMIZER
    optimizer = torch.optim.AdamW(
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
