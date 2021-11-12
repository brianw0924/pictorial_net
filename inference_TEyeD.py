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
from dataloader import TEyeDDataset
from dataloader import get_loader_TEyeD
from models.mynet import Mynet 
from main import test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/brianw0924/hdd/processed_data")
    parser.add_argument('--model_dir', type=str, default="./result")
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()

    assert os.path.exists(args.dataset)

    return args

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
    _, test_loader = get_loader_TEyeD(
        args.dataset, args.batch_size, args.num_workers, True)

    # MODEL & LOSS
    model = Mynet()
    criterion = nn.MSELoss(reduction='mean') # actually, default is already 'mean'

    # LOAD MODEL
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "UNet", "model_state_epoch20.pth"))['state_dict'])
    model.to(device)

    # TESTING
    test(0, model, criterion, test_loader, device)

if __name__ == '__main__':
    main()
