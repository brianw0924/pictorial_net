import os
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

'''
dataset_dir
    |
    |-image
    |   |-0000000.png
    |   |-0000001.png
    |   ...
    |
    |-gaze
    |   |-gaze.txt
    |
    |-landmark
    |   |-pupil_landmark.txt
    |   |-iris_landmark.txt
    |   |-lid_landmark.txt
    |
    |-segmentation
        |-0000000.png
        |-0000001.png
        ...

'''

tfm = transforms.Compose([
    transforms.Resize((144,192)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=0.5, std=0.5)
])

class Gaze_dataset(data.Dataset):

    def __init__(self, dataset_dir, ratio=None, val=True):
        self.image_path = sorted(glob.glob(os.path.join(dataset_dir,"image","*")))
        self.gaze_path = os.path.join(dataset_dir,"gaze","gaze.txt")
        self.gaze = []

        # gaze
        with open(self.gaze_path) as f:
            next(f)
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(',')
                g = torch.tensor([float(g) for g in line])
                self.gaze.append(g)

        # if cross-subject, split to train set & validation set
        if ratio:
            split = int(len(self.image_path)*ratio)
            if val:
                self.image_path = self.image_path[split:]
                self.gaze = self.gaze[split:]
            else:
                self.image_path = self.image_path[:split]
                self.gaze = self.gaze[:split]

        assert(len(self.gaze) == len(self.image_path))

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.gaze[index]

    def __len__(self):
        return len(self.image_path)

class Center_dataset(data.Dataset):

    def __init__(self, dataset_dir, ratio=None, val=True):
        self.image_path = sorted(glob.glob(os.path.join(dataset_dir,"image","*")))
        self.pupil_path = os.path.join(dataset_dir,"landmark","pupil_landmark.txt")
        self.iris_path = os.path.join(dataset_dir,"landmark","iris_landmark.txt")
        self.lid_path = os.path.join(dataset_dir,"landmark","lid_landmark.txt")
        self.pupil_lm = []
        self.iris_lm = []
        self.lid_lm = []

        with open(self.pupil_path) as f:
            next(f)
            for line in f.readlines():
                line = line.strip().split(',')
                px = torch.tensor([float(x) for x in line[::2]]).mean() / 2 # /2 because I resize to (144, 192)
                py = torch.tensor([float(y) for y in line[1::2]]).mean() / 2
                self.pupil_lm.append(torch.tensor((px,py)))

        with open(self.iris_path) as f:
            next(f)
            for line in f.readlines():
                line = line.strip().split(',')
                ix = torch.tensor([float(x) for x in line[::2]]).mean() / 2
                iy = torch.tensor([float(y) for y in line[1::2]]).mean() / 2
                self.iris_lm.append(torch.tensor((ix,iy)))
            
        with open(self.lid_path) as f:
            next(f)
            for line in f.readlines():
                line = line.strip().split(',')
                lx = torch.tensor([float(x) for x in line[::2]]).mean() / 2
                ly = torch.tensor([float(y) for y in line[1::2]]).mean() / 2
                self.lid_lm.append(torch.tensor((lx,ly)))

        assert(len(self.pupil_lm) == len(self.image_path) == len(self.iris_lm) == len(self.lid_lm))

        # if cross-subject, split to train set & validation set
        if ratio:
            split = int(len(self.image_path)*ratio)
            if val:
                self.image_path = self.image_path[split:]
                self.pupil_lm = self.pupil_lm[split:]
                self.iris_lm = self.iris_lm[split:]
                self.lid_lm = self.lid_lm[split:]
            else:
                self.image_path = self.image_path[:split]
                self.pupil_lm = self.pupil_lm[:split]
                self.iris_lm = self.iris_lm[:split]
                self.lid_lm = self.lid_lm[:split]

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.pupil_lm[index], self.iris_lm[index], self.lid_lm[index]

    def __len__(self):
        return len(self.image_path)

class Lid_2point_dataset(data.Dataset):

    def __init__(self, dataset_dir, ratio=None, val=True):
        self.image_path = sorted(glob.glob(os.path.join(dataset_dir,"image","*")))
        self.lid_path = os.path.join(dataset_dir,"landmark","lid_landmark.txt")
        self.lid_2point = []
            
        with open(self.lid_path) as f:
            next(f)
            for line in f.readlines():
                line = line.strip().split(',')
                lx1, ly1 = float(line[0])/2, float(line[1])/2
                lx2, ly2 = float(line[-2])/2, float(line[-1])/2
                self.lid_lm.append(torch.tensor((lx1, ly1, lx2, ly2)))

        assert(len(self.image_path) == len(self.lid_lm))

        # if cross-subject, split to train set & validation set
        if ratio:
            split = int(len(self.image_path)*ratio)
            if val:
                self.image_path = self.image_path[split:]
                self.lid_lm = self.lid_lm[split:]
            else:
                self.image_path = self.image_path[:split]
                self.lid_lm = self.lid_lm[:split]

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        return tfm(image),  self.lid_lm[index]

    def __len__(self):
        return len(self.image_path)



def Gaze_loader(args):

    if args.cross_target:
        train_dataset = Gaze_dataset(dataset_dir=args.data_dir, ratio=args.ratio, val=False)
        val_dataset   = Gaze_dataset(dataset_dir=args.data_dir, ratio=args.ratio, val=True)
    else:
        train_dataset = Gaze_dataset(dataset_dir=args.data_dir, ratio=None)
        split_len = int(len(train_dataset) * args.ratio)
        train_dataset, val_dataset = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
    
    print(f'train data len: {len(train_dataset)} | validation data len: {len(val_dataset)}')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader

def Center_loader(args):

    if args.cross_target:
        train_dataset = Center_dataset(dataset_dir=args.data_dir, ratio=args.ratio, val=False)
        val_dataset  = Center_dataset(dataset_dir=args.data_dir, ratio=args.ratio, val=True)
    else:
        train_dataset = Center_dataset(dataset_dir=args.data_dir, ratio=None)
        split_len = int(len(train_dataset) * args.ratio)
        train_dataset, val_dataset = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
    
    print(f'train data len: {len(train_dataset)} | validation data len: {len(val_dataset)}')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader