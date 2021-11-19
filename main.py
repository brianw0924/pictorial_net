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
from collections import OrderedDict
from tqdm import tqdm

from utils import Set_seed, vec2angle, pitch_loss, yaw_loss, plot_curve, update_log
from dataloader import TEyeDDataset
from dataloader_v2 import get_loader_TEyeD
from models.mynet import Mynet 
from models.pictorial_net import Model

'''
Before Training:

Remember to confirm:
0. model ?
1. args.outdir ?
2. args.cross_target ?
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--outdir', type=str, default='./result/cross_target/UNet16_linear1024')
    parser.add_argument('--cross_target', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--milestones', type=list, default=[5,15])
    parser.add_argument('--lr_decay', type=float, default=0.1)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

def train(args, epoch, model, optimizer, criterion, train_loader, device, log):
    model.train()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    count = 0
    for step, (images, gazes) in enumerate(tqdm(train_loader)):
        count+=gazes.shape[0]

        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, vec2angle(gazes).to(device))

        loss.backward()
        optimizer.step()

        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))
        
        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())

    loss_avg = sum(loss_list) / count
    pitch_loss_avg = sum(pitch_loss_list) / count
    yaw_loss_avg = sum(yaw_loss_list) / count

    tqdm.write( 'Train | '
                'Epoch {} | '
                'Loss {:.4f} | '
                'Pitch error (MAE) {:2f} | '
                'Yaw error (MAE) {:2f}'.format(
                    epoch,
                    loss_avg,
                    pitch_loss_avg,
                    yaw_loss_avg
                ))
    
    update_log(args, log, "Train", epoch, loss_avg, pitch_loss_avg, yaw_loss_avg)

    return loss_avg, pitch_loss_avg, yaw_loss_avg

def validation(args, epoch, model, criterion, val_loader, device, log):
    model.eval()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    count = 0
    for step, (images, gazes) in enumerate(tqdm(val_loader)):
        count+=gazes.shape[0]
        with torch.no_grad():
            outputs = model(images.to(device))

        loss = criterion(outputs, vec2angle(gazes).to(device))
        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))

        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())


    loss_avg = sum(loss_list) / count
    pitch_loss_avg = sum(pitch_loss_list) / count
    yaw_loss_avg = sum(yaw_loss_list) / count
    
    tqdm.write( 'Validation  | '
                'Epoch {} | '
                'Loss {:.4f} | '
                'Pitch error (MAE) {:.2f} | '
                'Yaw error (MAE) {:2f}'.format(
                    epoch,
                    loss_avg,
                    pitch_loss_avg,
                    yaw_loss_avg
                ))

    update_log(args, log, "Validation", epoch, loss_avg, pitch_loss_avg, yaw_loss_avg)

    return loss_avg, pitch_loss_avg, yaw_loss_avg

def main():

    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

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
    # torch.save(model.state_dict(), os.path.join(args.outdir, f'model_state.pth'))
    criterion = nn.MSELoss(reduction='sum')

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay)

    # PLOTTING
    plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss = [], [], []
    plot_val_loss, plot_val_pitch_loss, plot_val_yaw_loss = [], [], []

    # TRAINING
    validation(args, 0, model, criterion, val_loader, device, log) # first validationing before training

    error = 1e10
    for epoch in range(1, args.epochs + 1):

        # Training
        train_loss, train_pitch_loss, train_yaw_loss = train(
            args, epoch, model, optimizer, criterion, train_loader, device, log
        )
        scheduler.step()

        # Validation
        val_loss, val_pitch_loss, val_yaw_loss = validation(
            args, epoch, model, criterion, val_loader, device, log
        )

        # Append loss for plotting
        plot_train_loss.append(train_loss)
        plot_train_pitch_loss.append(train_pitch_loss)
        plot_train_yaw_loss.append(train_yaw_loss)

        plot_val_loss.append(val_loss)
        plot_val_pitch_loss.append(val_pitch_loss)
        plot_val_yaw_loss.append(val_yaw_loss)

        # Save state
        if val_loss < error:
            tqdm.write(f"Save model at epoch {epoch}")
            state = OrderedDict([
                ('state_dict', model.state_dict()),
                # ('optimizer', optimizer.state_dict()),
            ])
            torch.save(state, os.path.join(args.outdir, f'model_state.pth'))
            error = val_loss

        # Plot
        plot_curve(args, plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss, plot_val_loss, plot_val_pitch_loss, plot_val_yaw_loss)



if __name__ == '__main__':
    main()
