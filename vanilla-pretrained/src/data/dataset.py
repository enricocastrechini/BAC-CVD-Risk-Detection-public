# Copyright 2024 Enrico Castrechini
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: [Brief description of what the file or class does]

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from os.path import exists
import pandas as pd
import numpy as np
from torchvision.datasets.folder import default_loader
import torch


class Dataset(Dataset):
    def __init__(self, label_path, transform=None, transform_flag=True):
        self._image_paths = []
        self._labels = []
        self.indices = []
        self._folds = []
        self._label_names = {}
        self.transform = transform
        self.transform_flag = transform_flag

        # Reading the dataframe
        df = pd.DataFrame()
        for path in label_path:
            df_ = pd.read_csv(path)
            # df = df.append(df_)
            df = pd.concat([df, df_])


        #df['labels'] = df['labels'].apply(self.group_labels)
        df = df.fillna(' ')
        df = df[df['label']!=' ']
        self._image_paths = df['image_path'].to_list()
        self._labels = df['label'].to_list()
        self._folds = df['fold'].to_list()


        for i,labels in enumerate(list(set(self._labels))):
            self._label_names['label'+str(i)]=labels

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        img = default_loader(self._image_paths[idx])
        if self.transform is not None and self.transform_flag:
            img = self.transform(img)
        
        labels = np.array(self._labels[idx]).astype(np.float32)
        self.indices.append(idx)
        
        return img, labels
    
    def get_mean_std(self):
        images = []
        for img, label in self:
            img = np.array(img)
            images.append(img)
        images = np.array(images)
        print(images.shape)
        mean = images.mean(axis=(0,2,3))
        std = images.std(axis=(0,2,3))

        return mean, std
    

    def get_class_weights(self):
        class_count = self.get_class_distribution()
        class_freq = class_count / np.sum(class_count)

        class_weights = 1.0 / class_freq
        norm_weights = class_weights / np.sum(class_weights)

        return norm_weights

    def get_class_distribution(self):
        return np.bincount(self._labels)

    def get_labels(self):
        labels = np.array(self._labels).astype(np.float32)
        
        return labels
    
    def filter_data(self, column_name, fold):
        """
        Filter the dataset based on a specific column and fold value.
        Args:
            column_name (str): The name of the column to filter on.
            fold (int): The fold number to filter the dataset on.
        """
        self._image_paths = [self._image_paths[i] for i in range(len(self._image_paths)) if self._folds[i] == fold]
        self._labels = [self._labels[i] for i in range(len(self._labels)) if self._folds[i] == fold]



        
# import torch
# from torchvision.datasets import DatasetFolder
# from torchvision import transforms
# import pandas as pd
# import numpy as np
# from transformer import DataTransformer, MaskCropping, MakeRGB
# from breast_segment import BreastSegmenter

# # Debugging code
# label_path = ['train.csv']  # Assuming 'train.csv' is your label file
# transform = transforms.Compose([#transforms.Resize((224, 224)), 
#                                 MaskCropping(output_size=(2400, 3040)),
#                                 MakeRGB(),
#                                 transforms.ToTensor()
#                                 ])
# dataset = Dataset(label_path=label_path, transform=transform)

# # Test __getitem__ method
# print("Testing __getitem__ method:")
# for i in range(3):  # Adjust this value depending on how many samples you want to check
#     sample = dataset[i]
#     print("Sample", i)
#     print("Image shape:", sample[0].shape)
#     print("Label:", sample[1])

# # Test get_mean_std method
# # print("\nTesting get_mean_std method:")
# # mean, std = dataset.get_mean_std()
# # print("Mean:", mean)
# # print("Std:", std)

# # Test get_class_weights method
# print("\nTesting get_class_weights method:")
# class_weights = dataset.get_class_weights()
# print("Class weights:", class_weights)

# # Test get_class_distribution method
# print("\nTesting get_class_distribution method:")
# class_distribution = dataset.get_class_distribution()
# print("Class distribution:", class_distribution)

# # Test get_labels method
# print("\nTesting get_labels method:")
# labels = dataset.get_labels()
# print("Labels:", labels)


# print(dataset[0])