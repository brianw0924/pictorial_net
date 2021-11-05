# coding: utf-8

import os
import numpy as np
import glob
import torch
import torch.utils.data
import torchvision.transforms as transforms

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((144,192)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])



class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, '{}.npz'.format(subject_id))
        with np.load(path) as fin:
            self.images = fin['image']
            self.poses = fin['pose']
            self.gazes = fin['gaze']
        self.length = len(self.images)
        assert(len(self.images) == len(self.poses) == len(self.gazes))
        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.poses = torch.from_numpy(self.poses)
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        return self.images[index].float(), self.poses[index].float(), self.gazes[index].float()

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

class TEyeDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        with np.load(dataset_dir) as f:
            print(dataset_dir)
            self.images = f['image']
            self.gazes = f['gaze'][:,1:].astype(np.float32)
            valid = (self.gazes[:,0] != -1)
            self.gazes = self.gazes[valid]
            self.images = self.images[valid]
        self.length = len(self.images)
        assert(len(self.images) == len(self.gazes))
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        return tfm(self.images[index]),  self.gazes[index]

    def __len__(self):
        return self.length

def get_loader_TEyeD(dataset_dir, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    all_path = glob.glob(os.path.join(dataset_dir,"*"))
    train_dataset = torch.utils.data.ConcatDataset([
        TEyeDDataset(path) for path in all_path[1:25]
    ])
    test_dataset = TEyeDDataset(all_path[0])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)
    subject_ids = ['p{:02}'.format(index) for index in range(15)]
    test_subject_id = subject_ids[test_subject_id]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id != test_subject_id
    ])
    test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)


    # assert len(train_dataset) == 42000
    # assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
