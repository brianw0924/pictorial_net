import torch
import torch.nn as nn
import numpy as np
import random
import math
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def Set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def vec2angle(gaze):
    '''
    gaze shape : (batchsize, x, y, z) or (batchsize, yaw, pitch)
    
    * from your viewpoint *
    x+: right
    y+: down
    z+: point to you
    pitch+: up
    yaw+: left(your pov) ; right (patient's pov)
    '''
    if gaze[0].shape == torch.Size([2]): return gaze # label is already [yaw, pitch]

    x, y, z = gaze[:,0], gaze[:, 1], gaze[:, 2]
    pitch = - torch.atan(y/z) * 180 / np.pi
    yaw = - torch.atan(x/z) * 180 / np.pi
    pitch = torch.unsqueeze(pitch,dim=1)
    yaw = torch.unsqueeze(yaw,dim=1)
    return (torch.cat((yaw,pitch),dim=1))

def pitch_error(preds, labels):
    '''
    Mean Absolute Error
    '''
    criterion = nn.L1Loss(reduction='sum')
    return criterion(preds[:,1], labels[:,1])

def yaw_error(preds, labels):
    '''
    Mean Absolute Error
    '''
    criterion = nn.L1Loss(reduction='sum')
    return criterion(preds[:,0], labels[:,0])

def update_log_gaze(args, log: dict, train_or_val: str, step, pitch_error, yaw_error):
    log[train_or_val][step] = f'Pitch error {round(pitch_error,3)} | Yaw error {round(yaw_error,3)}'
    with open(os.path.join(args.out_dir, 'training_log.json'),'w') as f:
        json.dump(log, f, indent=2)

def update_log_valid(args, log: dict, train_or_val: str, step, sensitivity, specificity, score ):
    log[train_or_val][step] = f'Sensitivity {sensitivity} | Specificity {specificity} | Score: {score}'
    with open(os.path.join(args.out_dir, 'training_log.json'),'w') as f:
        json.dump(log, f, indent=2)

