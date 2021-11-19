# coding: utf-8

import os
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


tfm = transforms.Compose([
    transforms.Resize((144,192)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

class TEyeDDataset_mixed(data.Dataset):
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
        self.landmark_path = os.path.join(dataset_dir,"landmark","iris_landmark.txt")
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
        print(len(self.gaze))
        print(len(self.landmark))
        print(len(self.image_path))
        assert(len(self.gaze) == len(self.landmark) == len(self.image_path))

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.gaze[index]

    def __len__(self):
        return len(self.image_path)

class TEyeDDataset_cross(data.Dataset):
    '''
    dataset_dir
        |-image
        |-gaze
        |-landmark
        |-segmentation
    '''
    def __init__(self, dataset_dir, ratio, val):
        self.image_path = sorted(glob.glob(os.path.join(dataset_dir,"image","*")))
        self.gaze_path = os.path.join(dataset_dir,"gaze","gaze.txt")
        self.landmark_path = os.path.join(dataset_dir,"landmark","iris_landmark.txt")
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

        split = int(len(self.image_path)*ratio)
        if val:
            self.image_path = self.image_path[split:]
            self.gaze = self.gaze[split:]
            self.landmark = self.landmark[split:]
        else:
            self.image_path = self.image_path[:split]
            self.gaze = self.gaze[:split]
            self.landmark = self.landmark[:split]

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.gaze[index]

    def __len__(self):
        return len(self.image_path)


def get_loader_TEyeD(args):

    if args.cross_target:
        train_dataset = TEyeDDataset_cross(dataset_dir=args.dataset, ratio=args.ratio, val=False)
        val_dataset  = TEyeDDataset_cross(dataset_dir=args.dataset, ratio=args.ratio, val=True)
    else:
        train_dataset = TEyeDDataset_mixed(dataset_dir=args.dataset)
        split_len = int(len(train_dataset) * args.ratio)
        train_dataset, val_dataset = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
    
    print(f'train data len: {len(train_dataset)} | validation data len: {len(val_dataset)}')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
