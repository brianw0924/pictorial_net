import os
import time
import json
import importlib
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
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

from utils import Set_seed, vec2angle, pitch_error, yaw_error, plot_curve, update_log
from dataloader_v2 import get_loader_TEyeD
from models.mynet import Mynet 
from models.pictorial_net import Model

'''
[notes]

* Don't use SGDM, output will get nan
* Before Training, Remember to confirm:
    0. model ?
    1. args.outdir ?
    2. args.cross_target ?

'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--outdir', type=str, default='./result/cross_target/UNet16_vgg16bn_3')
    parser.add_argument('--cross_target', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=32*20000)
    parser.add_argument('--val_steps', type=int, default=32*1000)
    parser.add_argument('--warmup_steps', type=int, default=32*200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

def validation(args, step, model, criterion, val_loader, device, log):
    model.eval()


    val_loss = 0
    val_pitch_error = 0
    val_yaw_error = 0

    count = 0
    for images, gazes in val_loader:
        count+=gazes.shape[0]
        with torch.no_grad():
            gaze_out = model(images.to(device))

        loss = criterion(gaze_out, vec2angle(gazes).to(device))

        val_loss+=(loss.item())
        val_pitch_error+=(pitch_error(gaze_out, vec2angle(gazes).to(device)).item())
        val_yaw_error+=(yaw_error(gaze_out, vec2angle(gazes).to(device)).item())

    val_loss/= len(val_loader)
    val_pitch_error /= count
    val_yaw_error/= count
    
    tqdm.write( 'Validation  | '
                'Step {} | '
                'Loss {:.4f} | '
                'Pitch error (MAE) {:.2f} | '
                'Yaw error (MAE) {:2f}'.format(
                    str(step).zfill(6),
                    val_loss,
                    val_pitch_error,
                    val_yaw_error
                ))

    update_log(args, log, "Validation", step, val_loss, val_pitch_error, val_yaw_error)

    return val_loss, val_pitch_error, val_yaw_error


def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device, log):
    model.train()

    plot_train_loss, plot_train_pitch_error, plot_train_yaw_error = [], [], []
    plot_val_loss, plot_val_pitch_error, plot_val_yaw_error = [], [], []

    train_loss = 0
    train_pitch_error = 0
    train_yaw_error = 0
    count = 0
    error = 1e10

    train_iterator = iter(train_loader)

    for step in tqdm(range(args.total_steps)):

        try:
            images, gazes = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, gazes = next(train_iterator)
        
        # save_image(images,"preview.png",nrow=4)

        count+= images.shape[0]

        gaze_out = model(images.to(device))
        loss = criterion(gaze_out, vec2angle(gazes).to(device))
        loss = loss / args.accumulate
        loss.backward()

        train_loss+= loss.item()
        train_pitch_error+= pitch_error(gaze_out, vec2angle(gazes).to(device)).item()
        train_yaw_error+= yaw_error(gaze_out, vec2angle(gazes).to(device)).item()

        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:
            
            # TRAINING RECORD
            train_loss /= len(train_loader)
            train_pitch_error /= count
            train_yaw_error /= count

            tqdm.write( 'Train       | '
                        'Step {} | '
                        'Loss {:.4f} | '
                        'Pitch error (MAE) {:2f} | '
                        'Yaw error (MAE) {:2f} | '
                        'lr: {}'.format(
                            str(step+1).zfill(6),
                            train_loss,
                            train_pitch_error,
                            train_yaw_error,
                            optimizer.param_groups[0]["lr"]
                        ))
    
            update_log(args, log, "Train", step+1, train_loss, train_pitch_error, train_yaw_error)

            #VALIDATION
            val_loss, val_pitch_error, val_yaw_error = validation(
                args, step+1, model, criterion, val_loader, device, log
            )

            # SAVE MODEL
            if val_loss < error:
                tqdm.write(f"Save model at step {str(step+1).zfill(6)}")
                state = OrderedDict([
                    ('state_dict', model.state_dict()),
                    # ('optimizer', optimizer.state_dict()),
                ])
                torch.save(state, os.path.join(args.outdir, f'model_state.pth'))
                error = val_loss

            # PLOTTING
            plot_train_loss.append(train_loss)
            plot_train_pitch_error.append(train_pitch_error)
            plot_train_yaw_error.append(train_yaw_error)

            plot_val_loss.append(val_loss)
            plot_val_pitch_error.append(val_pitch_error)
            plot_val_yaw_error.append(val_yaw_error)

            plot_curve(args, plot_train_loss, plot_train_pitch_error, plot_train_yaw_error, plot_val_loss, plot_val_pitch_error, plot_val_yaw_error)
            
            train_loss = 0
            train_pitch_error = 0
            train_yaw_error = 0
            count = 0


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

    # OUTPUT DIRECTORY
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # RECORD ARGS
    args_json = os.path.join(args.outdir, 'args.json')
    with open(args_json, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # DATASET (TEyeD)
    print("Preparing data ...")
    train_loader, val_loader = get_loader_TEyeD(args)

    # MODEL & LOSS
    model = Mynet()
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
    # validation(args, 0, model, criterion, val_loader, device, log) # first validationing before training
    train(args, model, optimizer, scheduler_warmup, criterion, train_loader, val_loader, device, log)

if __name__ == '__main__':
    main()
