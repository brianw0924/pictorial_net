# coding: utf-8

import os
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

def tfm(image):
    a = random.randint(0,200)
    b = random.randint(0,200)
    c = random.randint(0,200)
    d = random.randint(0,200)
    padding = (a,b,c,d)
    transform = transforms.Compose([
        transforms.Pad(padding, fill=0,padding_mode="constant"),
        transforms.Resize((144,192)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    return transform(image)

class TEyeDDataset(data.Dataset):
    '''
    dataset_dir
        |-image
        |-gaze
        |-landmark
        |-segmentation
    '''
    def __init__(self, dataset_dir):
        
        self.image_path = sorted(glob.glob(os.path.join(dataset_dir,"image","*")))
        self.gaze_path = os.path.join(dataset_dir,"gaze","gaze.txt")
        self.landmark_path = os.path.join(dataset_dir,"landmark","landmark.txt")
        self.gaze = []
        self.landmark = []
        with open(self.gaze_path) as f:
            for line in f.readlines():
                line = line.split(';')
                self.gaze.append(torch.tensor([float(g) for g in line[1:-1]]))
        
        with open(self.landmark_path) as f:
            for line in f.readlines():
                line = line.split(';')
                self.landmark.append(torch.tensor([float(l) for l in line[2:-1]]))
        
        assert(len(self.gaze) == len(self.landmark) == len(self.image_path))

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.gaze[index]

    def __len__(self):
        return len(self.image_path)

def get_loader_TEyeD(dataset_dir, batch_size, num_workers, use_gpu):

    # train: 0.7 ; val: 0.3
    ratio = 0.7

    train_dataset = TEyeDDataset(dataset_dir)
    split_len = int(len(train_dataset) * ratio)
    train_dataset, val_dataset = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
    print(f'train data len: {len(train_dataset)} | validation data len: {len(val_dataset)}')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, val_loader
