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

from src.data.transformer import DataTransformer
from src.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

class BuildData:
    def __init__(self, trans_params, batch_size=32, shuffle=True, path=[], fold=None):
        # Data Loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path = path
        self.trans_params = trans_params
        self.fold = fold

    def build_data(self, transform_type='train'):        
        
        mean = None
        std = None

        data_trans_param = dict(height=self.trans_params['height'], width=self.trans_params['width'], set_type=transform_type, mean=mean, std=std)
        transform=DataTransformer(**data_trans_param).transformer

        dataset = Dataset(self.path,transform=transform)


        if self.fold is not None:
            # Filter data based on fold column
            dataset.filter_data('fold', self.fold)

        if self.trans_params['normalize']:
            # mean, std = dataset.get_mean_std()
            # print(mean)
            # print(std)
            mean = [0.28935949767803943, 0.28935949767803943, 0.28935949767803943]
            std = [0.17990232913372892, 0.17990232913372892, 0.17990232913372892]
            dataset.transform.transforms.append(transforms.Normalize(mean, std)) 

        # self.dataset = dataset

        return dataset

   
    
    def build_loader(self, dataset):
        batch_size = self.batch_size
        sample = None
        shuffle = self.shuffle
        workers = 24
        self.dataset = dataset

        loader =  DataLoader(dataset=self.dataset, sampler=sample, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        
        # Print the data distribution
        # print('\nClass distribution: {}, Class weights: {}'.format(self.dataset.get_class_distribution(), self.dataset.get_class_weights()))

        return loader