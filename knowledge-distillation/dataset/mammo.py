import os
import socket
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")
    
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = '/home/castrechini/projects/bac_project/src/data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    print(f"Data folder: {data_folder}")
    return data_folder


class MammogramDataset(Dataset):
    """Mammogram Dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print(f"Loading data from {csv_file}")
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 4]

        if self.transform:
            image = self.transform(image)

        return image, label


class MammogramDatasetInstance(MammogramDataset):
    """Mammogram Dataset with Instance Index."""
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label, idx


def get_mammogram_dataloaders_small(batch_size=32, num_workers=8, is_instance=False):
    """
    Load the small dataset with true labels and return train, validation, and test dataloaders.
    """
    data_folder = get_data_folder()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if is_instance:
        train_set = MammogramDatasetInstance(csv_file=os.path.join(data_folder, 'train.csv'),
                                             root_dir=data_folder,
                                             transform=transform)
        n_data = len(train_set)
    else:
        train_set = MammogramDataset(csv_file=os.path.join(data_folder, 'train.csv'),
                                     root_dir=data_folder,
                                     transform=transform)
        n_data = len(train_set)

    valid_set = MammogramDataset(csv_file=os.path.join(data_folder, 'valid.csv'),
                                 root_dir=data_folder,
                                 transform=test_transform)
    test_set = MammogramDataset(csv_file=os.path.join(data_folder, 'test.csv'),
                                root_dir=data_folder,
                                transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Loaded {n_data} samples for training")
    if is_instance:
        return train_loader, valid_loader, test_loader, n_data
    else:
        return train_loader, valid_loader, test_loader


class MammogramDatasetSample(MammogramDataset):
    """
    Mammogram Dataset with Sampling for Contrastive Learning.
    """
    def __init__(self, csv_file, root_dir, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(csv_file, root_dir, transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = self.data_frame['label'].nunique()
        self.cls_positive = [[] for _ in range(num_classes)]
        self.cls_negative = [[] for _ in range(num_classes)]

        for idx, row in self.data_frame.iterrows():
            self.cls_positive[row['label']].append(idx)

        for i in range(num_classes):
            for j in range(num_classes):
                if j != i:
                    self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[:n] for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, idx):
        image, label, _ = super().__getitem__(idx)

        if not self.is_sample:
            return image, label, idx
        else:
            if self.mode == 'exact':
                pos_idx = idx
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[label], 1)[0]
            else:
                raise NotImplementedError(self.mode)

            replace = True if self.k > len(self.cls_negative[label]) else False
            neg_idx = np.random.choice(self.cls_negative[label], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return image, label, idx, sample_idx


def get_mammogram_dataloaders_sample(batch_size=32, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    Load the small dataset with true labels and return a dataloader with sampled contrastive examples.
    """
    data_folder = get_data_folder()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = MammogramDatasetSample(csv_file=os.path.join(data_folder, 'train_big_08.csv'),
                                       root_dir=data_folder,
                                       transform=transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_set = MammogramDataset(csv_file=os.path.join(data_folder, 'valid.csv'),
                                 root_dir=data_folder,
                                 transform=test_transform)
    test_set = MammogramDataset(csv_file=os.path.join(data_folder, 'test.csv'),
                                root_dir=data_folder,
                                transform=test_transform)

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Loaded {n_data} samples with contrastive sampling for training")
    return train_loader, valid_loader, test_loader, n_data


def get_mammogram_dataloaders_big(batch_size=32, num_workers=8):
    """
    Load the big dataset with pseudo labels and return a dataloader.
    """
    data_folder = get_data_folder()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = MammogramDataset(csv_file=os.path.join(data_folder, 'train_big_08.csv'),
                                 root_dir=data_folder,
                                 transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f"Loaded {len(train_set)} samples for pseudo-label training")
    return train_loader


# Example usage
# if __name__ == '__main__':
#     train_loader, valid_loader, test_loader, n_data = get_mammogram_dataloaders_sample(batch_size=32)
#     train_loader_big = get_mammogram_dataloaders_big(batch_size=32)

#     # Iterate over the small dataset
#     for i, (images, labels, indices, sample_indices) in enumerate(train_loader):
#         print(f"Batch {i}:")
#         print(f"Images: {images.shape}")
#         print(f"Labels: {labels}")
#         print(f"Indices: {indices}")
#         print(f"Sample Indices: {sample_indices}")
#         break

#     # Iterate over the big dataset
#     for i, (images, labels, indices) in enumerate(train_loader_big):
#         print(f"Batch {i}:")
#         print(f"Images: {images.shape}")
#         print(f"Labels: {labels}")
#         print(f"Indices: {indices}")
#         break
