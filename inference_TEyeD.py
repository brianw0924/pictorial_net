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

from utils import Set_seed, vec2angle, pitch_loss, yaw_loss
from dataloader_v2 import get_loader_TEyeD
from models.mynet import Mynet 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--model_dir', type=str, default="./result")
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

def inference(args, epoch, model, criterion, test_loader, device):
    model.eval()

    loss_list = []
    pitch_loss_list = []
    yaw_loss_list = []

    num = 0
    for step, (images, gazes) in enumerate(tqdm(test_loader)):
        num+=gazes.shape[0]
        with torch.no_grad():
            outputs = model(images.to(device))

        loss = criterion(outputs, vec2angle(gazes).to(device))
        p_loss = pitch_loss(outputs, vec2angle(gazes).to(device))
        y_loss = yaw_loss(outputs, vec2angle(gazes).to(device))

        loss_list.append(loss.item())
        pitch_loss_list.append(p_loss.item())
        yaw_loss_list.append(y_loss.item())


    loss_avg = sum(loss_list) / num
    pitch_loss_avg = sum(pitch_loss_list) / num
    yaw_loss_avg = sum(yaw_loss_list) / num
    
    tqdm.write( 'Test  | '
                'Epoch {} | '
                'Loss {:.4f} | '
                'Pitch loss {:.2f} | '
                'Yaw loss {:2f}'.format(
                    epoch,
                    loss_avg,
                    pitch_loss_avg,
                    yaw_loss_avg
                ))

    p_distribution = [0,0,0,0,0,0]
    y_distribution = [0,0,0,0,0,0]

    for i in range(6):
        p_distribution[i]+= (torch.tensor(pitch_loss_list)<i).sum().item()
        y_distribution[i]+= (torch.tensor(yaw_loss_list)<i).sum().item()

    p_distribution = [ round(i/len(test_loader)*100) for i in p_distribution]
    y_distribution = [ round(i/len(test_loader)*100) for i in y_distribution]
    return p_distribution, y_distribution


def main():

    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    
    # ARGUMENTS
    args = parse_args()

    # RANDOM SEED
    Set_seed(args.seed)

    # DATASET (TEyeD)
    print("Preparing data ...")
    _, test_loader = get_loader_TEyeD(args)

    # MODEL & LOSS
    model = Mynet()
    criterion = nn.MSELoss(reduction='sum') # actually, default is already 'mean'

    # LOAD MODEL
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "UNet16_linear1024", "model_state.pth"))['state_dict'])
    model.to(device)

    # TESTING
    P, Y = inference(args, 0, model, criterion, test_loader, device)
    print("           pitch  |  yaw")
    for i in range(6):
        print(f'<={i} degree: {P[i]:3}%  |  {Y[i]}%')



if __name__ == '__main__':
    main()
