from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

def get_data_folder():
    """
    Return server-dependent path to store the data.
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = '/home/castrechini/projects/bac_project/dataset/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class UnlabeledMammoDataset(Dataset):
    """Unlabeled Mammography Dataset."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_mammo_dataloaders(batch_size=64, num_workers=8, is_instance=False):
    """
    Mammography data loaders.
    """
    data_folder = get_data_folder()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Labeled dataset
    train_labeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/train'), transform=transform_train)
    val_labeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
    
    train_loader_labeled = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if is_instance:
        train_unlabeled = UnlabeledMammoDataset(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)
        n_data = len(train_unlabeled)
    else:
        train_unlabeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)
        n_data = len(train_unlabeled)

    train_loader_unlabeled = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if is_instance:
        return train_loader_labeled, val_loader, train_loader_unlabeled, n_data
    else:
        return train_loader_labeled, val_loader, train_loader_unlabeled

class MammoInstanceSample(Dataset):
    """
    Mammography Dataset with instance sampling for contrastive learning.
    """
    def __init__(self, root, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
        self.root = root
        self.transform = transform
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        self.num_samples = len(self.image_paths)

        self.cls_positive = [[i] for i in range(self.num_samples)]
        self.cls_negative = [[j for j in range(self.num_samples) if j != i] for i in range(self.num_samples)]

        if 0 < percent < 1:
            for i in range(self.num_samples):
                n = int(len(self.cls_negative[i]) * percent)
                self.cls_negative[i] = np.random.permutation(self.cls_negative[i])[:n]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if not self.is_sample:
            return image, index
        else:
            pos_idx = index
            replace = True if self.k > len(self.cls_negative[index]) else False
            neg_idx = np.random.choice(self.cls_negative[index], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

            return image, index, sample_idx

def get_mammo_dataloaders_sample(batch_size=64, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    Mammography data loaders with instance sampling for contrastive learning.
    """
    data_folder = get_data_folder()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = MammoInstanceSample(root=os.path.join(data_folder, 'processed_hbb_5000'),
                                    transform=transform_train, k=k, mode=mode, is_sample=is_sample, percent=percent)
    n_data = len(train_set)
    train_loader_unlabeled = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_set = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader_unlabeled, val_loader, n_data
