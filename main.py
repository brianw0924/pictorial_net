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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms as transforms
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader, get_loader_TEyeD

torch.backends.cudnn.benchmark = True


logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


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
    parser.add_argument('--test_id', type=int, required=True)
    parser.add_argument('--outdir', type=str, default='./result')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--image_size', type=tuple, default=None)

    # optimizer
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_images', action='store_true')
    parser.add_argument('--tensorboard_parameters', action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False
        args.tensorboard_images = False
        args.tensorboard_parameters = False

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
    return criterion(preds[:,0], labels[:,0])

def yaw_loss(preds, labels):
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'
    return criterion(preds[:,1], labels[:,1])


def train(epoch, model, optimizer, criterion, train_loader, config, writer, device):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_list = []

    start = time.time()
    for step, (images, gazes) in enumerate(train_loader):
        global_step += 1

        if config['tensorboard_images'] and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        optimizer.zero_grad()
        outputs, gazemap = model(images.to(device))
        loss = criterion(outputs, vec2angle(gazes).to(device))
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())


        if config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss.item(), global_step)

        if step % 1000 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f}'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss.item(),
                        ))

    elapsed = time.time() - start
    # logger.info('Elapsed {:.2f}'.format(elapsed))

    loss_avg = sum(loss_list) / len(loss_list)

    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    print(f'Train: {loss_avg}')

    return loss_avg

def test(epoch, model, criterion, test_loader, config, writer, device):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []
    start = time.time()
    for step, (images, gazes) in enumerate(test_loader):
        
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)


        with torch.no_grad():
            outputs, gazemap = model(images.to(device))
        loss = criterion(outputs, vec2angle(gazes).to(device))
        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))

        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())


    loss_avg = sum(loss_list) / len(loss_list)
    pitch_loss_avg = sum(pitch_loss_list) / len(pitch_loss_list)
    yaw_loss_avg = sum(yaw_loss_list) / len(yaw_loss_list)
    
    logger.info('Epoch {} | Loss {:.4f} | Pitch loss {:.2f} | Yaw loss {:2f}'.format(
        epoch, loss_avg, pitch_loss_avg, yaw_loss_avg))

    elapsed = time.time() - start
    # logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_avg, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if config['tensorboard_parameters']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)


    return loss_avg, pitch_loss_avg, yaw_loss_avg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=2))

    # TensorBoard SummaryWriter
    writer = SummaryWriter() if args.tensorboard else None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    config = {
        'tensorboard': args.tensorboard,
        'tensorboard_images': args.tensorboard_images,
        'tensorboard_parameters': args.tensorboard_parameters,
    }

    # run test before start training
    test(0, model, criterion, test_loader, config, writer, device)

    plot_train_loss, plot_test_loss, plot_test_pitch_loss, plot_test_yaw_loss = [], [], [], []

    for epoch in range(1, args.epochs + 1):

        # Training
        train_loss = train(epoch, model, optimizer, criterion, train_loader, config, writer, device)
        # scheduler.step()

        # Testing / Validation
        test_loss, test_pitch_loss, test_yaw_loss = test(
            epoch, model, criterion, test_loader, config, writer, device)

        plot_train_loss.append(train_loss)
        plot_test_loss.append(train_loss)
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

    if args.tensorboard:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)

    print(len(plot_train_loss))
    print(plot_train_loss)
    print(len(plot_test_loss))
    print(plot_test_loss)
    x = [i+1 for i in range(len(plot_train_loss))]
    plt.plot(x,plot_train_loss,color='black',label='train')
    plt.plot(x,plot_test_loss,color='blue',label='test')
    plt.plot(x,plot_test_loss,color='green',label='test_pitch')
    plt.plot(x,plot_test_loss,color='red',label='test_yaw')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Loss_Curve")
    plt.clf()


if __name__ == '__main__':
    main()
