import os
import time
import json
import importlib
import argparse
import numpy as np
import random
import math
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
from PIL import Image
from torchvision.utils import save_image
from collections import OrderedDict
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from utils import Set_seed, vec2angle, pitch_error, yaw_error, update_log_lm
from dataloader_v2 import Center_loader
from models.mynet import Eye_Localization_Net
from models.pictorial_net import Model

'''
[notes]

Suggestion: 
* Don't use SGDM, pred will get nan
* Before Training, Remember to confirm:
    0. selecting correct model ?
    1. args.out_dir ?
    2. args.cross_target ?

'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--dataset', type=str, default="TEyeD")
    parser.add_argument('--out_dir', type=str, default='./result/landmark/UNet16_vggbn16_1')
    parser.add_argument('--cross_target', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=1024*2000)
    parser.add_argument('--val_steps', type=int, default=1024*10)
    parser.add_argument('--warmup_steps', type=int, default=1024*10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.3)

    args = parser.parse_args()

    assert os.path.exists(args.data_dir)

    return args

def validation(args, step, model, val_loader, device, log):
    criterion = nn.MSELoss(reduction='sum')
    model.eval()

    val_loss = 0
    val_pupil_center_loss = 0
    val_iris_center_loss = 0
    val_lid_center_loss = 0

    count = 0 # count how many sample used (because last iteration might not be full batch)
    for images, pupil_center, iris_center, lid_center in val_loader:
        count+=images.shape[0]
        with torch.no_grad():
            pred = model(images.to(device)) # pupil center, iris center, lid center

        pupil_center_loss = criterion(pred[:,:2],pupil_center.to(device))
        iris_center_loss  = criterion(pred[:,2:4],iris_center.to(device))
        lid_center_loss   = criterion(pred[:,4:],lid_center.to(device))
        loss              = (pupil_center_loss + iris_center_loss + lid_center_loss)

        val_loss               += loss.item()
        val_pupil_center_loss  += pupil_center_loss.item()
        val_iris_center_loss   += iris_center_loss.item()
        val_lid_center_loss    += lid_center_loss.item()

    val_loss               /= count
    val_pupil_center_loss  /= count
    val_iris_center_loss   /= count
    val_lid_center_loss    /= count
    
    tqdm.write( 'Validation  | '
                'Step {} | '
                'Loss {:.4f} | '
                'Pupil center (RMSE) {:.2f} | '
                'Iris center (RMSE) {:2f} | '
                'Lid center (RMSE) {:2f}'.format(
                    str(step).zfill(6),
                    val_loss,
                    math.sqrt(val_pupil_center_loss),
                    math.sqrt(val_iris_center_loss),
                    math.sqrt(val_lid_center_loss)
                ))

    update_log_lm(args, log, "Validation", step, val_loss, val_pupil_center_loss, val_iris_center_loss, val_lid_center_loss)

    return val_loss, val_pupil_center_loss, val_iris_center_loss, val_lid_center_loss


def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device, log):
    model.train()

    train_loss = 0
    train_pupil_center_loss = 0
    train_iris_center_loss = 0 
    train_lid_center_loss = 0

    train_iterator = iter(train_loader)
    error = 1e10
    count = 0
    for step in tqdm(range(args.total_steps)):

        try:
            images, pupil_center, iris_center, lid_center = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, pupil_center, iris_center, lid_center = next(train_iterator)

        save_image(images, "preview.png", nrow=4)

        count+= images.shape[0]
        pred = model(images.to(device)) # pupil, iris, lid center

        pupil_center_loss   = criterion(pred[:,:2],pupil_center.to(device))
        iris_center_loss    = criterion(pred[:,2:4],iris_center.to(device))
        lid_center_loss     = criterion(pred[:,4:],lid_center.to(device))
        loss                = (pupil_center_loss + iris_center_loss + lid_center_loss) / args.accumulate
        loss.backward()

        train_loss              += loss.item()
        train_pupil_center_loss += pupil_center_loss.item()
        train_iris_center_loss  += iris_center_loss.item()
        train_lid_center_loss   += lid_center_loss.item()

        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:
            
            # TRAINING RECORD
            train_loss              /=  (args.val_steps / args.accumulate)
            train_pupil_center_loss /=  args.val_steps
            train_iris_center_loss  /=  args.val_steps
            train_lid_center_loss   /=  args.val_steps



    
            tqdm.write( 'Train       | '
                        'Step {} | '
                        'Loss {:.4f} | '
                        'Pupil center (RMSE) {:.2f} | '
                        'Iris center (RMSE) {:2f} | '
                        'Lid center (RMSE) {:2f}'.format(
                            str(step).zfill(6),
                            train_loss,
                            math.sqrt(train_pupil_center_loss),
                            math.sqrt(train_iris_center_loss),
                            math.sqrt(train_lid_center_loss)
                        ))
    
            update_log_lm(args, log, "Train", step+1, train_loss, train_pupil_center_loss, train_iris_center_loss, train_lid_center_loss)

            # VALIDATION
            val_loss, val_pupil_center_loss, val_iris_center_loss, val_lid_center_loss = validation(
                args, step+1, model, val_loader, device, log
            )

            # SAVE MODEL
            if val_loss < error:
                tqdm.write(f"Save model at step {str(step+1).zfill(6)}")
                state = OrderedDict([
                    ('state_dict', model.state_dict()),
                    # ('optimizer', optimizer.state_dict()),
                ])
                torch.save(state, os.path.join(args.out_dir, f'model_state.pth'))
                error = val_loss

            train_loss              = 0
            train_pupil_center_loss = 0
            train_iris_center_loss  = 0 
            train_lid_center_loss   = 0
            count                   = 0

            model.train()


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

    # RANDOM SEED
    Set_seed(args.seed)

    # pred DIRECTORY
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # RECORD ARGS
    args_json = os.path.join(args.out_dir, 'args.json')
    with open(args_json, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data_dir (TEyeD)
    print("Preparing data ...")
    train_loader, val_loader = Center_loader(args)

    # MODEL & LOSS
    model = Eye_Localization_Net()
    model.to(device)
    # print(model)
    criterion = nn.MSELoss()

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # SCHEDULER
    scheduler = CosineAnnealingLR(optimizer, T_max=(args.total_steps-args.warmup_steps)/args.accumulate)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_steps//args.accumulate, after_scheduler=scheduler)
    optimizer.zero_grad()
    optimizer.step()

    # TRAINING
    train(args, model, optimizer, scheduler_warmup, criterion, train_loader, val_loader, device, log)

if __name__ == '__main__':
    main()
