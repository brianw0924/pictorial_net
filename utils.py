import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt


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
    gaze shape : (batchsize, x, y, z)
    
    * from your viewpoint *
    x+: right
    y+: down
    z+: point to you
    pitch+: z->y
    yaw+: z->x
    '''
    x, y, z = gaze[:,0], gaze[:, 1], gaze[:, 2]
    pitch = torch.atan(y/z) * 180 / np.pi
    yaw = torch.atan(x/z) * 180 / np.pi
    pitch = torch.unsqueeze(pitch,dim=1)
    yaw = torch.unsqueeze(yaw,dim=1)
    return (torch.cat((pitch,yaw),dim=1))

def pitch_error(preds, labels):
    '''
    Mean Absolute Error
    '''
    criterion = nn.L1Loss(reduction='sum')
    return criterion(preds[:,0], labels[:,0])

def yaw_error(preds, labels):
    '''
    Mean Absolute Error
    '''
    criterion = nn.L1Loss(reduction='sum')
    return criterion(preds[:,1], labels[:,1])

def plot_curve(args, plot_train_loss, plot_train_pitch_error, plot_train_yaw_error, plot_val_loss, plot_val_pitch_error, plot_val_yaw_error):
    x = [i+1 for i in range(len(plot_train_loss))]
    plt.plot(x,plot_train_loss,color='navy',label='train(MSE)')
    plt.plot(x,plot_train_pitch_error,color='darkgreen',label='train_pitch(MAE)')
    plt.plot(x,plot_train_yaw_error,color='darkred',label='train_yaw(MAE)')
    plt.plot(x,plot_val_loss,ls=':',color='navy',label='val(MSE)')
    plt.plot(x,plot_val_pitch_error,ls=':',color='darkgreen',label='val_pitch(MAE)')
    plt.plot(x,plot_val_yaw_error,ls=':',color='darkred',label='val_yaw(MAE)')
    plt.legend(loc='upper left')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.outdir,"Loss_Curve.png"))
    plt.clf()   

def update_log(args, log: dict, train_or_val: str, epoch, loss, pitch_error, yaw_error):
    log[train_or_val][epoch] = f'Loss {round(loss,3)} | Pitch error {round(pitch_error,3)} | Yaw error {round(yaw_error,3)}'
    with open(os.path.join(args.outdir, 'training_log.json'),'w') as f:
        json.dump(log, f, indent=2)
