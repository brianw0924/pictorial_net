import os
import numpy as np
import glob
import random
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

'''
data_dir/
        |
        ├─image
        |   ├─0000000.png
        |   ├─0000001.png
        |   ...
        |
        ├─gaze
        |   └─gaze.txt
        |
        ├─landmark
        |   ├─pupil_landmark.txt
        |   ├─iris_landmark.txt
        |   └─lid_landmark.txt
        |
        └─segmentation_dataset
            ├─0000000.png
            ├─0000001.png
            ...
'''

class Gaze_dataset(data.Dataset):

    def __init__(self, dataset, data_dir, ratio=None, val=False):
        self.tfm = transforms.Compose([
                    transforms.Resize((144,192)),
                    transforms.ToTensor(),
                ])

        self.dataset = dataset
        self.val = val
        self.image_path = sorted(glob.glob(os.path.join(data_dir,"image","*")))
        self.gaze_path = os.path.join(data_dir,"gaze","gaze.txt")
        self.gaze = []

        # gaze
        with open(self.gaze_path) as f:
            next(f)
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(',')
                g = torch.tensor([float(g) for g in line])
                self.gaze.append(g)

        # remove invalid data (closed eye)
        self.image_path = [p for i,p in enumerate(self.image_path) if self.gaze[i][0] != -1]
        self.gaze = [g for g in self.gaze if g[0]!=-1]

        # if cross-subject, split to train set & validation set
        if ratio:
            split = int(len(self.image_path)*ratio)
            if self.val:
                self.image_path = self.image_path[split:]
                self.gaze = self.gaze[split:]
            else:
                self.image_path = self.image_path[:split]
                self.gaze = self.gaze[:split]

        assert(len(self.gaze) == len(self.image_path))

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        if self.dataset == "Neurobit":
            if self.val:
                image = image.crop((120, 50, 520, 350)) # (left, top, right ,bot)
            else:
                vertical = random.randint(-50,50)
                horizontal = random.randint(-70,70)
                image = image.crop((120+horizontal, 50+vertical, 520+horizontal, 350+vertical))
        
        return self.tfm(image),  self.gaze[index]

    def __len__(self):
        return len(self.image_path)

class Valid_dataset(data.Dataset):

    def __init__(self, dataset, data_dir, ratio=None, val=False):
        self.tfm = transforms.Compose([
                    transforms.Resize((144,192)),
                    transforms.ToTensor(),
                ])
        self.dataset = dataset
        self.val = val
        self.image_path = sorted(glob.glob(os.path.join(data_dir,"image","*")))
        self.gaze_path = os.path.join(data_dir,"gaze","gaze.txt")
        self.gaze = []
        self.valid = []

        # gaze
        with open(self.gaze_path) as f:
            next(f)
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(',')
                if float(line[0]) == -1:
                    self.valid.append(0.)
                else:
                    self.valid.append(1.)
        
        if ratio:
            split = int(len(self.image_path)*ratio)
            if self.val:
                self.image_path = self.image_path[split:]
                self.valid = self.valid[split:]
            else:
                self.image_path = self.image_path[:split]
                self.valid = self.valid[:split]

        assert(len(self.valid) == len(self.image_path))

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])

        if self.dataset == "Neurobit":
            if self.val:
                image = image.crop((120, 50, 520, 350)) # (left, top, right ,bot)
            else:
                vertical = random.randint(-50,50)
                horizontal = random.randint(-70,70)
                image = image.crop((120+horizontal, 50+vertical, 520+horizontal, 350+vertical))
        
        return self.tfm(image),  self.valid[index]

    def __len__(self):
        return len(self.image_path)

def Gaze_loader(args):

    if args.cross_subject:
        train_dataset = Gaze_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=args.ratio, val=False)
        val_dataset   = Gaze_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=args.ratio, val=True)
    else:
        train_dataset = Gaze_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=None, val=False)
        val_dataset = Gaze_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=None, val=True)
        split_len = int(len(train_dataset) * args.ratio)
        train_dataset, _ = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
        _, val_dataset = data.random_split(val_dataset, [split_len, len(val_dataset) - split_len])

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

def Valid_loader(args):

    if args.cross_subject:
        train_dataset = Valid_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=args.ratio, val=False)
        val_dataset   = Valid_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=args.ratio, val=True)
    else:
        train_dataset = Valid_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=None, val=False)
        val_dataset = Valid_dataset(dataset=args.dataset, data_dir=args.data_dir, ratio=None, val=True)
        split_len = int(len(train_dataset) * args.ratio)
        train_dataset, _ = data.random_split(train_dataset, [split_len, len(train_dataset) - split_len])
        _, val_dataset = data.random_split(val_dataset, [split_len, len(val_dataset) - split_len])

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