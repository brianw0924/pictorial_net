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

from utils import Set_seed, vec2angle, pitch_loss, yaw_loss, plot_curve
from dataloader import TEyeDDataset
from dataloader import get_loader_TEyeD
from models.mynet import Mynet 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/processed_data")
    parser.add_argument('--outdir', type=str, default='./result')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--milestones', type=list, default=[6,15])
    parser.add_argument('--lr_decay', type=float, default=0.1)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

def train_dynamic_loading(args, epoch, model, optimizer, criterion, dataset_dir, device):

    tqdm.write('Train {}'.format(epoch))

    all_path = glob.glob(os.path.join(dataset_dir,"*"))[:-1]

    model.train()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    for num_file, file in enumerate(tqdm(all_path)):
        train_dataset = TEyeDDataset(file)
        # tqdm.write(f'Training on : {file} | Image amount: {len(train_dataset)}')
        if len(train_dataset) == 0:
            continue

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loss_list_inner = []
        pitch_loss_list_inner = []
        yaw_loss_list_inner = []

        # Inner loop #

        for step, (images, gazes) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, vec2angle(gazes).to(device))
            loss.backward()
            optimizer.step()

            p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
            y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))
        
            
            loss_list_inner.append(loss.item())
            pitch_loss_list_inner.append(p_loss.item())
            yaw_loss_list_inner.append(y_loss.item())

        # Inner loop #

        loss_avg = sum(loss_list_inner) / len(loss_list_inner)
        p_loss_avg = sum(pitch_loss_list_inner) / len(pitch_loss_list_inner)
        y_loss_avg = sum(yaw_loss_list_inner) / len(yaw_loss_list_inner)

        loss_list.append(loss_avg)
        pitch_loss_list.append(p_loss_avg)
        yaw_loss_list.append(y_loss_avg)

        tqdm.write('Train | Epoch {} Step {}/{} | '
                    'Loss {:.4f} | '
                    'Pitch loss {:2f} | '
                    'Yaw loss {:2f}'.format(
                        epoch,
                        num_file+1,
                        len(all_path),
                        loss_avg,
                        p_loss_avg,
                        y_loss_avg
                    ))
        
        del train_loader, train_dataset, loss_list_inner

    loss_avg = sum(loss_list) / len(loss_list)
    pitch_loss_avg = sum(pitch_loss_list) / len(pitch_loss_list)
    yaw_loss_avg = sum(yaw_loss_list) / len(yaw_loss_list)

    return loss_avg, pitch_loss_avg, yaw_loss_avg


def train(epoch, model, optimizer, criterion, train_loader, device):
    model.train()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    for step, (images, gazes) in enumerate(tqdm(train_loader)):
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

    loss_avg = sum(loss_list) / len(loss_list)
    pitch_loss_avg = sum(pitch_loss_list) / len(pitch_loss_list)
    yaw_loss_avg = sum(yaw_loss_list) / len(yaw_loss_list)

    tqdm.write( 'Train | '
                'Epoch {} | '
                'Loss {:.4f} | '
                'Pitch loss {:2f} | '
                'Yaw loss {:2f}'.format(
                    epoch,
                    loss_avg,
                    pitch_loss_avg,
                    yaw_loss_avg
                ))

    return loss_avg, pitch_loss_avg, yaw_loss_avg

def test(epoch, model, criterion, test_loader, device):
    model.eval()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    for step, (images, gazes) in enumerate(test_loader):

        with torch.no_grad():
            outputs = model(images.to(device))

        loss = criterion(outputs, vec2angle(gazes).to(device))
        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))

        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())


    loss_avg = sum(loss_list) / len(loss_list)
    pitch_loss_avg = sum(pitch_loss_list) / len(pitch_loss_list)
    yaw_loss_avg = sum(yaw_loss_list) / len(yaw_loss_list)
    
    tqdm.write( 'Test | '
                'Epoch {} | '
                'Loss {:.4f} | '
                'Pitch loss {:.2f} | '
                'Yaw loss {:2f}'.format(
                    epoch,
                    loss_avg,
                    pitch_loss_avg,
                    yaw_loss_avg
                ))

    return loss_avg, pitch_loss_avg, yaw_loss_avg

def main():

    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

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
    train_loader, test_loader = get_loader_TEyeD(
        args.dataset, args.batch_size, args.num_workers, True)

    # MODEL & LOSS
    model = Mynet()
    model.to(device)
    # print(model)
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay)

    # PLOTTING
    plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss = [], [], []
    plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss = [], [], []

    # TRAINING
    test(0, model, criterion, test_loader, device) # first testing before training
    for epoch in range(1, args.epochs + 1):

        # Training
        train_loss, train_pitch_loss, train_yaw_loss = train(epoch, model, optimizer, criterion, train_loader, device)
        # train_loss, train_pitch_loss, train_yaw_loss = train_dynamic_loading(
        #     args, epoch, model, optimizer, criterion, args.dataset, device
        # )
        scheduler.step()

        # Validation
        test_loss, test_pitch_loss, test_yaw_loss = test(
            epoch, model, criterion, test_loader, device
        )

        # Append loss for plotting
        plot_train_loss.append(train_loss)
        plot_train_pitch_loss.append(train_pitch_loss)
        plot_train_yaw_loss.append(train_yaw_loss)

        plot_test_loss.append(test_loss)
        plot_test_pitch_loss.append(test_pitch_loss)
        plot_test_yaw_loss.append(test_yaw_loss)

        # Save state
        state = OrderedDict([
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
        ])
        torch.save(state, os.path.join(outdir, f'model_state_epoch{epoch}.pth'))

        # Plot
        plot_curve(args, plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss, plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss)



if __name__ == '__main__':
    main()
