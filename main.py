#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.models as models

from dataloader import TEyeDDataset
from dataloader import get_loader, get_loader_TEyeD

torch.backends.cudnn.benchmark = True


logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

global_step = 0

def Set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_curve(plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss, plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss):
    x = [i+1 for i in range(len(plot_train_loss))]
    plt.plot(x,plot_train_loss,color='navy',label='train(MSE)')
    plt.plot(x,plot_train_pitch_loss,color='darkgreen',label='train_pitch(RMSE)')
    plt.plot(x,plot_train_yaw_loss,color='darkred',label='train_yaw(RMSE)')
    plt.plot(x,plot_test_loss,ls=':',color='navy',label='test(MSE)')
    plt.plot(x,plot_test_pitch_loss,ls=':',color='darkgreen',label='test_pitch(RMSE)')
    plt.plot(x,plot_test_yaw_loss,ls=':',color='darkred',label='test_yaw(RMSE)')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Loss_Curve.png")
    plt.clf()    


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', type=str, required=True, choices=['lenet', 'resnet_preact', 'pictorial_net'])
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/processed_data")
    parser.add_argument('--outdir', type=str, default='./result')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--image_size', type=tuple, default=None)

    # optimizer
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)
    args.milestones = json.loads(args.milestones)

    return args
        

'''
from user's view
x: right
y: down
z: point to user from camera
'''

def vec2angle(gaze):
    # gaze: (bs,x,y,z)
    x, y, z = gaze[:,0], gaze[:, 1], gaze[:, 2]
    pitch = torch.atan(y/z) * 180 / np.pi
    yaw = torch.atan(x/z) * 180 / np.pi
    pitch = torch.unsqueeze(pitch,dim=1)
    yaw = torch.unsqueeze(yaw,dim=1)
    return (torch.cat((pitch,yaw),dim=1))

def pitch_loss(preds, labels):
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'
    return torch.sqrt(criterion(preds[:,0], labels[:,0]))

def yaw_loss(preds, labels):
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'
    return torch.sqrt(criterion(preds[:,1], labels[:,1]))

def train_dynamic_loading(args, epoch, model, optimizer, criterion, dataset_dir, device):
    global global_step

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
            global_step += 1

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

        tqdm.write('Epoch {} Step {}/{} | '
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
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    start = time.time()
    for step, (images, gazes) in enumerate(train_loader):
        global_step += 1

        optimizer.zero_grad()
        outputs, gazemap = model(images.to(device))
        loss = criterion(outputs, vec2angle(gazes).to(device))
        loss.backward()
        optimizer.step()
        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))
        
        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())


        if step % 1000 == 0:
            logger.info('Epoch {} Step {}/{} | '
                        'Loss {:.4f} | '
                        'Pitch loss {:2f} | '
                        'Yaw loss {:2f}'
                        .format(
                            epoch,
                            step,
                            len(train_loader),
                            loss.item(),
                            
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    loss_avg = sum(loss_list) / len(loss_list)
    pitch_loss_avg = sum(pitch_loss_list) / len(pitch_loss_list)
    yaw_loss_avg = sum(yaw_loss_list) / len(yaw_loss_list)

    return loss_avg, pitch_loss_avg, yaw_loss_avg

def test(epoch, model, criterion, test_loader, device):
    tqdm.write('Test {}'.format(epoch))

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
    
    tqdm.write('Epoch {} | Loss {:.4f} | Pitch loss {:.2f} | Yaw loss {:2f}'.format(
        epoch, loss_avg, pitch_loss_avg, yaw_loss_avg))


    return loss_avg, pitch_loss_avg, yaw_loss_avg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=2))


    # set random seed
    Set_seed(args.seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    print("Preparing data ...")

    # TEyeD dataset
    train_loader, test_loader = get_loader_TEyeD(
        args.dataset, args.batch_size, args.num_workers, True)

    # model & loss function
    module = importlib.import_module('models.{}'.format(args.arch))
    model = module.Model()
    model.to(device)
    print(model)
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'

    # # SGD
    # optimizer
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.base_lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    #     nesterov=args.nesterov)

    # Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.milestones, gamma=args.lr_decay)

    # run test before start training
    test(0, model, criterion, test_loader, device)

    plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss = [], [], []
    plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss = [], [], []

    for epoch in range(1, args.epochs + 1):

        # Training
        train_loss, train_pitch_loss, train_yaw_loss = train(epoch, model, optimizer, criterion, train_loader, config, device)
        # train_loss, train_pitch_loss, train_yaw_loss = train_dynamic_loading(
        #     args, epoch, model, optimizer, criterion, args.dataset, device
        # )
        # scheduler.step()

        # Testing / Validation
        test_loss, test_pitch_loss, test_yaw_loss = test(
            epoch, model, criterion, test_loader, device
        )

        plot_train_loss.append(train_loss)
        plot_train_pitch_loss.append(train_pitch_loss)
        plot_train_yaw_loss.append(train_yaw_loss)

        plot_test_loss.append(test_loss)
        plot_test_pitch_loss.append(test_pitch_loss)
        plot_test_yaw_loss.append(test_yaw_loss)

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
        ])
        model_path = os.path.join(outdir, f'model_state_epoch{epoch}.pth')
        torch.save(state, model_path)

        plot_curve(plot_train_loss, plot_train_pitch_loss, plot_train_yaw_loss, plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss)




if __name__ == '__main__':
    main()
