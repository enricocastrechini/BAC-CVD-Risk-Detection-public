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

import torch
from math import cos, pi

class Optimizer:
    def __init__(self, optim=None, lr=None, scheduler=None, patience=None, factor=None, params=None, iterations=None, dataloader=None, epochs=None, wd=0):
        self.optim = optim
        self.lr = lr
        self.scheduler = scheduler
        self.patience = patience
        self.factor = factor
        self.params = params
        self.iterations = iterations
        self.wd = wd
        self.dataloader = dataloader
        self.epochs = epochs
    
    def get_optimizer(self):
        if self.optim=='Adam':
            optimizer = torch.optim.Adam(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='AMSgrad':
            optimizer = torch.optim.Adam(self.params,lr=self.lr, amsgrad=True,weight_decay=self.wd)
        elif self.optim=='Adadelta':
            optimizer = torch.optim.Adadelta(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='RAdam':
            optimizer = torch.optim.RAdam(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='AdamW':
            optimizer = torch.optim.AdamW(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='SGD':
            optimizer = torch.optim.SGD(self.params,lr=self.lr,weight_decay=self.wd,momentum=0.9,nesterov=True)
        self.optimizer = optimizer
        return optimizer
    
    def get_scheduler(self):
        scheduler = ''
        if self.scheduler!='':
            if self.scheduler=='ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=self.factor)
                
            elif self.scheduler=='CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.01,mode='triangular2',cycle_momentum=False,
                                                step_size_up=350)
            elif self.scheduler=='OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, steps_per_epoch=len(self.dataloader), epochs=self.epochs)

            elif self.scheduler=='CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.iterations) #eta_min=1e-10
                
            elif self.scheduler=='CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer) #TO FIX
            
        
        return scheduler
