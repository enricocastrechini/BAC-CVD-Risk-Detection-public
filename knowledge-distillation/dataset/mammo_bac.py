# from __future__ import print_function

# import os
# import socket
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# from PIL import Image

# def get_data_folder():
#     """
#     Return server-dependent path to store the data.
#     """
#     hostname = socket.gethostname()
#     if hostname.startswith('visiongpu'):
#         data_folder = '/data/vision/phillipi/rep-learn/datasets'
#     elif hostname.startswith('yonglong-home'):
#         data_folder = '/home/yonglong/Data/data'
#     else:
#         data_folder = 'data'

#     if not os.path.isdir(data_folder):
#         os.makedirs(data_folder)

#     return data_folder

# class UnlabeledMammoDataset(Dataset):
#     """Unlabeled Mammography Dataset."""
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# def get_mammo_dataloaders(batch_size=64, num_workers=8, is_instance=False):
#     """
#     Mammography data loaders.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     # Labeled dataset
#     train_labeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/train'), transform=transform_train)
#     val_labeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
    
#     train_loader_labeled = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
#     if is_instance:
#         train_unlabeled = UnlabeledMammoDataset(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)
#         n_data = len(train_unlabeled)
#     else:
#         train_unlabeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)

#     train_loader_unlabeled = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     if is_instance:
#         return train_loader_labeled, val_loader, train_loader_unlabeled, n_data
#     else:
#         return train_loader_labeled, val_loader, train_loader_unlabeled

# class MammoInstanceSample(Dataset):
#     """
#     Mammography Dataset with instance sampling for contrastive learning.
#     """
#     def __init__(self, root, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
#         self.root = root
#         self.transform = transform
#         self.k = k
#         self.mode = mode
#         self.is_sample = is_sample

#         self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
#         self.num_samples = len(self.image_paths)

#         self.cls_positive = [[i] for i in range(self.num_samples)]
#         self.cls_negative = [[j for j in range(self.num_samples) if j != i] for i in range(self.num_samples)]

#         if 0 < percent < 1:
#             for i in range(self.num_samples):
#                 n = int(len(self.cls_negative[i]) * percent)
#                 self.cls_negative[i] = np.random.permutation(self.cls_negative[i])[:n]

#         self.cls_positive = np.asarray(self.cls_positive)
#         self.cls_negative = np.asarray(self.cls_negative)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         if not self.is_sample:
#             return image, index
#         else:
#             pos_idx = index
#             replace = True if self.k > len(self.cls_negative[index]) else False
#             neg_idx = np.random.choice(self.cls_negative[index], self.k, replace=replace)
#             sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

#             return image, index, sample_idx

# def get_mammo_dataloaders_sample(batch_size=64, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
#     """
#     Mammography data loaders with instance sampling for contrastive learning.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = MammoInstanceSample(root=os.path.join(data_folder, 'processed_hbb_5000'),
#                                     transform=transform_train, k=k, mode=mode, is_sample=is_sample, percent=percent)
#     n_data = len(train_set)
#     train_loader_unlabeled = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     val_set = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader_unlabeled, val_loader, n_data

# from __future__ import print_function

# import os
# import socket
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# from PIL import Image

# def get_data_folder():
#     """
#     Return server-dependent path to store the data.
#     """
#     hostname = socket.gethostname()
#     if hostname.startswith('visiongpu'):
#         data_folder = '/data/vision/phillipi/rep-learn/datasets'
#     elif hostname.startswith('yonglong-home'):
#         data_folder = '/home/yonglong/Data/data'
#     else:
#         data_folder = 'data'

#     if not os.path.isdir(data_folder):
#         os.makedirs(data_folder)

#     return data_folder

# class UnlabeledMammoDataset(Dataset):
#     """Unlabeled Mammography Dataset."""
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# class LabeledMammoDataset(Dataset):
#     """Labeled Mammography Dataset."""
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#         self._load_data()

#     def _load_data(self):
#         class_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
#         class_folders.sort()
#         class_to_idx = {class_folder: idx for idx, class_folder in enumerate(class_folders)}

#         for class_folder in class_folders:
#             class_path = os.path.join(self.root_dir, class_folder)
#             for file_name in os.listdir(class_path):
#                 if os.path.isfile(os.path.join(class_path, file_name)):
#                     self.image_paths.append(os.path.join(class_path, file_name))
#                     self.labels.append(class_to_idx[class_folder])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')
#         label = self.labels[index]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# def get_mammo_dataloaders(batch_size=64, num_workers=8, is_instance=False):
#     """
#     Mammography data loaders.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     # Labeled dataset
#     train_labeled = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/train'), transform=transform_train)
#     val_labeled = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
    
#     train_loader_labeled = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
#     if is_instance:
#         train_unlabeled = UnlabeledMammoDataset(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)
#         n_data = len(train_unlabeled)
#     else:
#         train_unlabeled = datasets.ImageFolder(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)

#     train_loader_unlabeled = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     if is_instance:
#         return train_loader_labeled, val_loader, train_loader_unlabeled, n_data
#     else:
#         return train_loader_labeled, val_loader, train_loader_unlabeled

# class MammoInstanceSample(Dataset):
#     """
#     Mammography Dataset with instance sampling for contrastive learning.
#     """
#     def __init__(self, root, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
#         self.root = root
#         self.transform = transform
#         self.k = k
#         self.mode = mode
#         self.is_sample = is_sample

#         self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
#         self.num_samples = len(self.image_paths)

#         self.cls_positive = [[i] for i in range(self.num_samples)]
#         self.cls_negative = [[j for j in range(self.num_samples) if j != i] for i in range(self.num_samples)]

#         if 0 < percent < 1:
#             for i in range(self.num_samples):
#                 n = int(len(self.cls_negative[i]) * percent)
#                 self.cls_negative[i] = np.random.permutation(self.cls_negative[i])[:n]

#         self.cls_positive = np.asarray(self.cls_positive)
#         self.cls_negative = np.asarray(self.cls_negative)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         if not self.is_sample:
#             return image, index
#         else:
#             pos_idx = index
#             replace = True if self.k > len(self.cls_negative[index]) else False
#             neg_idx = np.random.choice(self.cls_negative[index], self.k, replace=replace)
#             sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

#             return image, index, sample_idx

# def get_mammo_dataloaders_sample(batch_size=64, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
#     """
#     Mammography data loaders with instance sampling for contrastive learning.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = MammoInstanceSample(root=os.path.join(data_folder, 'processed_hbb_5000'),
#                                     transform=transform_train, k=k, mode=mode, is_sample=is_sample, percent=percent)
#     n_data = len(train_set)
#     train_loader_unlabeled = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     val_set = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader_unlabeled, val_loader, n_data

# from __future__ import print_function

# import os
# import socket
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image

# def get_data_folder():
#     """
#     Return server-dependent path to store the data.
#     """
#     hostname = socket.gethostname()
#     if hostname.startswith('visiongpu'):
#         data_folder = '/data/vision/phillipi/rep-learn/datasets'
#     elif hostname.startswith('yonglong-home'):
#         data_folder = '/home/yonglong/Data/data'
#     else:
#         data_folder = 'data'

#     if not os.path.isdir(data_folder):
#         os.makedirs(data_folder)

#     return data_folder

# class LabeledMammoDataset(Dataset):
#     """Labeled Mammography Dataset."""
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#         self._load_data()
#         print(f"Labeled dataset initialized with {len(self.image_paths)} images from {root_dir}")

#     def _load_data(self):
#         # Mapping folder names to labels
#         class_to_idx = {
#             'cases_not_checked_for_bac': 0,
#             'exams_with_vascular_calcs': 1
#         }

#         for class_folder, class_idx in class_to_idx.items():
#             class_path = os.path.join(self.root_dir, class_folder)
#             if os.path.isdir(class_path):
#                 for file_name in os.listdir(class_path):
#                     file_path = os.path.join(class_path, file_name)
#                     if os.path.isfile(file_path):
#                         self.image_paths.append(file_path)
#                         self.labels.append(class_idx)
#                         print(f"Loaded {file_path} with label {class_idx}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')
#         label = self.labels[index]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# class UnlabeledMammoDataset(Dataset):
#     """Unlabeled Mammography Dataset."""
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
#         print(f"Unlabeled dataset initialized with {len(self.image_paths)} images from {root_dir}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, index  # Return both image and index

# class MammoInstanceSample(Dataset):
#     """
#     Mammography Dataset with instance sampling for contrastive learning.
#     """
#     def __init__(self, root, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
#         self.root = root
#         self.transform = transform
#         self.k = k
#         self.mode = mode
#         self.is_sample = is_sample

#         self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
#         self.num_samples = len(self.image_paths)

#         self.cls_positive = [[i] for i in range(self.num_samples)]
#         self.cls_negative = [[j for j in range(self.num_samples) if j != i] for i in range(self.num_samples)]

#         if 0 < percent < 1:
#             for i in range(self.num_samples):
#                 n = int(len(self.cls_negative[i]) * percent)
#                 self.cls_negative[i] = np.random.permutation(self.cls_negative[i])[:n]

#         self.cls_positive = np.asarray(self.cls_positive)
#         self.cls_negative = np.asarray(self.cls_negative)
#         print(f"Instance sample dataset initialized with {self.num_samples} images from {root}")

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         img_path = self.image_paths[index]
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         if not self.is_sample:
#             return image, index
#         else:
#             pos_idx = index
#             replace = True if self.k > len(self.cls_negative[index]) else False
#             neg_idx = np.random.choice(self.cls_negative[index], self.k, replace=replace)
#             sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

#             return image, index, sample_idx

# def get_mammo_dataloaders(batch_size=64, num_workers=8):
#     """
#     Mammography data loaders.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     # Labeled dataset
#     train_labeled = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/train'), transform=transform_train)
#     val_labeled = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
    
#     train_loader_labeled = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader_labeled, val_loader

# def get_mammo_dataloaders_unlabeled(batch_size=64, num_workers=8):
#     """
#     Mammography data loaders for unlabeled data.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_unlabeled = UnlabeledMammoDataset(os.path.join(data_folder, 'processed_hbb_5000'), transform=transform_train)
#     train_loader_unlabeled = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     return train_loader_unlabeled

# def get_mammo_dataloaders_sample(batch_size=64, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
#     """
#     Mammography data loaders with instance sampling for contrastive learning.
#     """
#     data_folder = get_data_folder()

#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     transform_val = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train_set = MammoInstanceSample(root=os.path.join(data_folder, 'processed_hbb_5000'),
#                                     transform=transform_train, k=k, mode=mode, is_sample=is_sample, percent=percent)
#     n_data = len(train_set)
#     train_loader_unlabeled = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     val_set = LabeledMammoDataset(os.path.join(data_folder, 'processed_hbb/val'), transform=transform_val)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader_unlabeled, val_loader, n_data

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_data_folder():
    """
    Return the local dataset directory configured by BAC_DATA_DIR.
    """
    data_folder = os.environ.get('BAC_DATA_DIR', 'data')
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_folder}. "
            "Set BAC_DATA_DIR to an authorized local dataset directory."
        )
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


def get_mammogram_dataloaders(batch_size=32, num_workers=8, is_instance=False):
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

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

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