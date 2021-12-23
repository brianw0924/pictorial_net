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

from utils import Set_seed, update_log_iris_lm
from dataloader_v2 import Iris_landmark_loader
from models.mynet import Iris_landmark_Net

'''
[notes]

Suggestion: 
* Don't use SGDM, pred will get nan
* Before Training, Remember to confirm:
    0. selecting correct model ?
    1. args.out_dir ?
    2. args.cross_target ?

'''
def preview(images, iris_lm):
    ims = []
    for b in range(images.shape[0]):
        p = iris_lm[b].cpu().detach().numpy()
        p = [int(i) for i in p]
        image = torch.cat((images[b], images[b], images[b]), dim=0)
        image = np.uint8(np.asarray(transforms.functional.to_pil_image(image)))

        for i in range(8):
            cv2.circle(image, p[i*2:i*2+2], radius=1, color=(0, 0, 255), thickness=-1)
        image = transforms.functional.to_tensor(image)
        image = torch.unsqueeze(image, dim=0)
        ims.append(image)
    ims = torch.cat(ims,dim=0)
    save_image(ims, "preview.png", nrow=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/brianw0924/hdd/TEyeD")
    parser.add_argument('--dataset', type=str, default="TEyeD")
    parser.add_argument('--out_dir', type=str, default='./result/landmark/UNet16_vggbn16_0')
    parser.add_argument('--cross_target', action="store_true")
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--total_steps', type=int, default=1024*500)
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

    # Preview the first batch
    for images, iris_lm in val_loader:
        with torch.no_grad():
            pred = model(images.to(device))
        preview(images, pred)
        break

    val_loss = 0
    count = 0 # count how many sample used (because last iteration might not be full batch)
    for images, iris_lm in val_loader:
        count+=images.shape[0]
        with torch.no_grad():
            pred = model(images.to(device)) # pupil center, iris center, lid center
        loss = criterion(pred, iris_lm.to(device))
        val_loss += loss.item()
    val_loss = math.sqrt(val_loss/count)
    
    tqdm.write( 'Validation  | '
                'Step {} | '
                'Iris center (RMSE) {:2f} | '.format(
                    str(step).zfill(6),
                    val_loss,
                ))

    update_log_iris_lm(args, log, "Validation", step, val_loss, val_loss)

    return val_loss


def train(args, model, optimizer, scheduler, criterion, train_loader, val_loader, device, log):
    model.train()

    train_loss = 0

    train_iterator = iter(train_loader)
    error = 1e10
    count = 0
    for step in tqdm(range(args.total_steps)):

        try:
            images, iris_lm = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, iris_lm = next(train_iterator)

        count+= images.shape[0]
        pred = model(images.to(device)) # pupil, iris, lid center
        loss = criterion(pred, iris_lm.to(device))
        loss.backward()
        
        train_loss += loss.item()

        if (step+1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if (step+1) % args.val_steps == 0:
            
            # TRAINING RECORD
            train_loss = math.sqrt(train_loss / (args.val_steps / args.accumulate))

            tqdm.write( 'Train       | '
                        'Step {} | '
                        'Iris center (RMSE) {:2f}'.format(
                            str(step).zfill(6),
                            train_loss,
                        ))
    
            update_log_iris_lm(args, log, "Train", step+1, train_loss, train_loss)

            # VALIDATION
            val_loss = validation(
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
    train_loader, val_loader = Iris_landmark_loader(args)

    # MODEL & LOSS
    model = Iris_landmark_Net()
    model.to(device)
    # print(model)
    criterion = nn.MSELoss()

    # OPTIMIZER
    optimizer = torch.optim.AdamW(
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
