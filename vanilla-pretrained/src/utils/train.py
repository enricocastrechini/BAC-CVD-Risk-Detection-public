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

# from src.utils.optimizers import Optimizer
# from src.utils.loss import Loss
# from src.utils.visualize import Visualize
# from src.data.databuilder import BuildData
# from src.models.models import BuildModel
# from src.utils.eval import EvaluateModel
# import torch
# import numpy as np
# import pandas as pd
# import os
# import datetime
# import torchvision.utils as vutils
# from tqdm.auto import tqdm
# import pickle
# from src.utils.reproducibility import reproducibility
# import math
# import wandb
# from torch.utils.data import ConcatDataset, Subset
# from torch.cuda.amp import autocast, GradScaler


# class ExperimentSetup:
#     def __init__(self, train_params, model_params, vis_params, trans_params):
#         self.train_params = train_params  # Training parameters from the config file
#         self.model_params = model_params  # Model parameters from the config file
#         self.vis_params = vis_params  # Visualization of training and validation samples
#         self.trans_params = trans_params  # Preprocessing parameters

#         # Curriculum learning parameters
#         self.use_curriculum = self.train_params.get('use_curriculum', False)
#         self.num_phases = self.train_params.get('num_phases', 5)
#         self.initial_phase_proportion = self.train_params.get('initial_phase_proportion', 0.1)
#         self.max_phase_proportion = self.train_params.get('max_phase_proportion', 0.5)

#     def get_curriculum_dataset(self, labeled_dataset, pseudo_labeled_dataset, proportion):
#         labeled_size = int(len(labeled_dataset) * (1 - proportion))
#         pseudo_size = int(len(pseudo_labeled_dataset) * proportion)
#         labeled_indices = list(range(labeled_size))
#         pseudo_indices = list(range(pseudo_size))
#         combined_dataset = ConcatDataset([
#             Subset(labeled_dataset, labeled_indices),
#             Subset(pseudo_labeled_dataset, pseudo_indices)
#         ])
#         return combined_dataset

#     def checkncreate_dir(self, dir):
#         if not os.path.exists(dir):
#             os.makedirs(dir)

#     def save_model(self, model, mod_name):
#         path = mod_name + "weights.pth"
#         torch.save(model.state_dict(), path)

#     def visualize_samples(self, dataset, data_type, path=''):
#         vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
#                          mode=data_type, idx_list=[],
#                          samples=10, cols=5, random_img=True,
#                          save_path=path)
#         idx_list = vis_.visualize_samples(transform=0)
#         vis_.idx_list = idx_list
#         vis_.random_img = False
#         vis_.visualize_samples(transform=1)
#         vis_.visualize_histograms(transform=0)
#         vis_.visualize_histograms(transform=1)

#     def predict_proba(self, dataset, model, criterion, device, out_neurons):
#         model.eval()
#         pred_prob = []
#         _lab = []
#         total_loss = 0
#         batches = 0
#         with torch.no_grad():
#             progress_bar = tqdm(enumerate(dataset), total=len(dataset))
#             for i, (X, y) in progress_bar:
#                 X, y = X.to(device), y.to(device)

#                 out = model(X)

#                 if out_neurons == 1:
#                     loss = criterion(out, y.unsqueeze(1))
#                     prob = torch.sigmoid(out)
#                 else:
#                     loss = criterion(out, y.long())
#                     prob = torch.softmax(out.data, 1)

#                 _lab.extend(y.detach().cpu().numpy().tolist())
#                 pred_prob.extend(prob.detach().cpu().numpy().tolist())

#                 total_loss += loss.item()
#                 batches += 1

#         avg_loss = total_loss / batches

#         return _lab, pred_prob, avg_loss

#     key = 'd35f2244112b09c039f0b033fcfd38331e9f46a4'
#     wandb.login(anonymous='never', key=key)

#     def train(self):
#         # Training params
#         epochs = self.train_params['epochs']
#         device = self.train_params['device']
#         batch_size = self.train_params['batch_size']
#         patience = self.train_params['patience']
#         save_path = self.train_params['save_path']
#         optimizer_type = self.train_params['optimizer']
#         lr = self.train_params['lr']
#         scheduler_type = self.train_params['scheduler']
#         lr_decay_factor = self.train_params['lr_decay_factor']
#         weight_decay = self.train_params['weight_decay']
#         loss_type = self.train_params['loss']
#         tr_data_path = self.train_params['tr_path']
#         pseudo_path = self.train_params['pseudo_path']
#         vd_data_path = self.train_params['vd_path']
#         num_folds = self.train_params['num_folds']
#         lr_iterations = self.train_params['lr_iterations']
#         seed = self.train_params['seed']
#         use_amp = self.train_params.get('amp', False)  # AMP flag
#         early_stopping = self.train_params.get('early_stopping', False)  # Early Stopping flag

#         # Transform params
#         normalize = self.trans_params['normalize']
#         img_height = self.trans_params['height']
#         img_width = self.trans_params['width']
#         augmentation = self.trans_params.get('augmentation', False)  # Augmentation flag

#         # torch.cuda.empty_cache()

#         # Initialize GradScaler for AMP
#         scaler = GradScaler() if use_amp else None

#         device = torch.device(device)

#         reproducibility(seed)

#         # Visualization
#         vis_samples = self.vis_params.vis
#         vis_path = self.vis_params.path

#         # Model params
#         fr_layer = self.model_params['fr_layer']
#         out_neurons = self.model_params['out_neurons']
#         model_name = self.model_params['model_name']
#         pretrained = self.model_params['pretrained']

#         model_params = dict(fr_layer=fr_layer,
#                             out_neurons=out_neurons,
#                             model_name=model_name,
#                             pretrained=pretrained)

#         model_builder = BuildModel(**model_params)

#         # Curriculum learning proportions
#         phase_proportions = np.linspace(self.initial_phase_proportion, self.max_phase_proportion, self.num_phases)

#         for fold in tqdm(range(num_folds)):
#             # Directories
#             inst_name = self.model_params['model_name']
#             version = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#             self.curr_version = version
#             logs_dir = save_path + inst_name + "/" + version + "/" + f"fold{fold}" + "/"
#             self.checkncreate_dir(logs_dir)
#             # writer = SummaryWriter(log_dir=logs_dir)
#             print('\nModel version: {}'.format(version))

#             wandb.init(
#                 # Set the project where this run will be logged
#                 project="BAC_project",
#                 # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#                 name=f"{inst_name}_{version}_{fold}",
#                 # Track hyperparameters and run metadata
#                 config={
#                     "architecture": model_name,
#                     "learning_rate": lr,
#                     "epochs": epochs,
#                     "batch_size": batch_size,
#                     "patience": patience,
#                     "optimizer": optimizer_type,
#                     "scheduler": scheduler_type,
#                     "loss": loss_type,
#                     "num_folds": num_folds,
#                     "seed": seed,
#                     "img_height": img_height,
#                     "img_width": img_width,
#                     "fr_layer": fr_layer,
#                     "normalize": normalize,
#                     "lr_iterations": lr_iterations,
#                     "augmentation": augmentation,
#                     "amp": use_amp,
#                     "early_stopping": early_stopping,
#                     "weight_decay": weight_decay,
#                 })

#             model = model_builder.get_model()

#             model = model.to(device)

#             # Log the network weight histograms (optional)
#             wandb.watch(model)

#             print(model)

#             train_data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=tr_data_path, fold=fold)
#             valid_data_builder = BuildData(self.trans_params, batch_size, shuffle=False, path=vd_data_path, fold=fold)

#             train_dataset_orig = train_data_builder.build_data('train')
#             datasets = [train_dataset_orig]

#             if augmentation:
#                 train_dataset_hflip = train_data_builder.build_data('train_hflip')
#                 datasets.append(train_dataset_hflip)

#             # Use ConcatDataset only if there's more than one dataset to concatenate
#             if len(datasets) > 1:
#                 train_dataset = ConcatDataset(datasets)
#             else:
#                 train_dataset = datasets[0]

#             valid_dataset = valid_data_builder.build_data('valid')

#             train_loader = train_data_builder.build_loader(train_dataset)
#             valid_loader = valid_data_builder.build_loader(valid_dataset)

#             n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)

#             class_weights = train_dataset_orig.get_class_weights()
#             class_weights = torch.tensor(class_weights).float().to(device)

#             # Visualize Samples
#             if vis_samples:
#                 self.visualize_samples(train_dataset_orig, 'train', vis_path)
#                 if augmentation:
#                     self.visualize_samples(train_dataset_hflip, 'train_hflip', vis_path)
#                 self.visualize_samples(valid_dataset, 'valid', vis_path)

#             # Optimizer and Scheduler
#             optim = Optimizer(optim=optimizer_type,
#                               lr=lr,
#                               scheduler=scheduler_type,
#                               patience=patience,
#                               factor=lr_decay_factor,
#                               wd=weight_decay,
#                               iterations=lr_iterations,
#                               dataloader=train_loader,
#                               epochs=epochs,
#                               params=model.parameters())

#             optimizer = optim.get_optimizer()

#             scheduler = optim.get_scheduler()

#             # Loss function
#             crit = Loss(loss=loss_type,
#                         weights=class_weights)

#             criterion = crit.get_loss()

#             # Results
#             best_loss = 99999.0
#             trigger = 0

#             # Training with Curriculum Learning
#             for phase in range(self.num_phases if self.use_curriculum else 1):
#                 current_proportion = phase_proportions[phase] if self.use_curriculum else 0
#                 print(f"Phase {phase + 1}/{self.num_phases}, Proportion of pseudolabeled data: {current_proportion:.2f}")

#                 if self.use_curriculum:
#                     pseudo_data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=pseudo_path, fold=fold)
#                     pseudo_labeled_dataset = pseudo_data_builder.build_data('pseudo_train')
#                     current_dataset = self.get_curriculum_dataset(train_dataset, pseudo_labeled_dataset, current_proportion)
#                     train_loader = train_data_builder.build_loader(current_dataset)

#                 for epoch in tqdm(range(epochs)):
#                     model.train()
#                     loss_epoch = 0
#                     itr = 0
#                     y_proba = []
#                     y_labels = []

#                     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
#                     for i, (X, y) in progress_bar:
#                         X, y = X.to(device), y.to(device)

#                         optimizer.zero_grad()

#                         # AMP or normal execution
#                         if use_amp:
#                             with autocast():
#                                 outputs = model(X)
#                                 if out_neurons == 1:
#                                     loss = criterion(outputs, y.unsqueeze(1))
#                                     prob = torch.sigmoid(outputs)
#                                 else:
#                                     loss = criterion(outputs, y.long())
#                                     prob = torch.softmax(outputs.data, 1)
#                             scaler.scale(loss).backward()
#                             scaler.step(optimizer)
#                             scaler.update()
#                         else:
#                             outputs = model(X)
#                             if out_neurons == 1:
#                                 loss = criterion(outputs, y.unsqueeze(1))
#                                 prob = torch.sigmoid(outputs)
#                             else:
#                                 loss = criterion(outputs, y.long())
#                                 prob = torch.softmax(outputs.data, 1)
#                             loss.backward()
#                             optimizer.step()

#                         loss_epoch += loss.item()
#                         y_proba.extend(prob.detach().cpu().numpy())
#                         y_labels.extend(y.detach().cpu().numpy())

#                     # Post-epoch: Calculate metrics, evaluate model, log to W&B...
#                     tr_loss = loss_epoch / len(progress_bar)
#                     tr_model_eval = EvaluateModel(y_labels, y_proba).scores

#                     metrics = {"train/train_loss": tr_loss,
#                                "train/epoch": (i + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
#                                "train/train_accuracy": tr_model_eval['acc'],
#                                "train/auc": tr_model_eval['auroc'],
#                                "train/aupr": tr_model_eval['aupr'],
#                                "train/f1": tr_model_eval['F1'],
#                                "train/mcc": tr_model_eval['MCC'],
#                                "train/precision": tr_model_eval['precision'],
#                                "train/recall": tr_model_eval['recall'],
#                                }

#                     # Validation Data Evaluation
#                     y_vd_labels, y_vd_proba, vd_loss = self.predict_proba(valid_loader, model, criterion, device, out_neurons)

#                     vd_model_eval = EvaluateModel(y_vd_labels, y_vd_proba).scores

#                     val_metrics = {"val/val_loss": vd_loss,
#                                    "val/val_accuracy": vd_model_eval['acc'],
#                                    "val/auc": vd_model_eval['auroc'],
#                                    "val/aupr": vd_model_eval['aupr'],
#                                    "val/f1": vd_model_eval['F1'],
#                                    "val/mcc": vd_model_eval['MCC'],
#                                    "val/precision": vd_model_eval['precision'],
#                                    "val/recall": vd_model_eval['recall'],
#                                    }

#                     wandb.log({**metrics, **val_metrics})

#                     print('Epoch {}, train loss {}, valid loss {}, valid auc {}'.format(epoch + 1, tr_loss, vd_loss, vd_model_eval['auroc']))

#                     if best_loss > vd_loss:
#                         self.save_model(model, logs_dir)
#                         best_loss = vd_loss
#                         best_model = model
#                         trigger = 0
#                     else:
#                         trigger += 1
#                         print('Trigger count: {}'.format(trigger))

#                     if early_stopping and trigger > patience:
#                         print('Early stopping! ' + inst_name + '_' + version + '_' + f'{fold}')
#                         wandb.finish()
#                         break

#                     scheduler.step(vd_loss)

#                 print('Ending training for -- ' + inst_name + '_' + version + '_' + f'{fold}')
#                 wandb.finish()

#             print('Training completed for all folds')

from src.utils.optimizers import Optimizer
from src.utils.loss import Loss
from src.utils.visualize import Visualize
from src.data.databuilder import BuildData
from src.models.models import BuildModel
from src.utils.eval import EvaluateModel
import torch
import numpy as np
import pandas as pd
import os
import datetime
import torchvision.utils as vutils
from tqdm.auto import tqdm
import pickle
from src.utils.reproducibility import reproducibility
import math
import wandb
from torch.utils.data import ConcatDataset, Subset
from torch.cuda.amp import autocast, GradScaler

WANDB_API_KEY = os.getenv('WANDB_API_KEY')
WANDB_MODE = os.getenv('WANDB_MODE', 'offline')
if WANDB_API_KEY and WANDB_MODE == 'online':
    wandb.login(anonymous='never', key=WANDB_API_KEY)


class ExperimentSetup:
    def __init__(self, train_params, model_params, vis_params, trans_params):
        self.train_params = train_params  # Training parameters from the config file
        self.model_params = model_params  # Model parameters from the config file
        self.vis_params = vis_params  # Visualization of training and validation samples
        self.trans_params = trans_params  # Preprocessing parameters

        # Curriculum learning parameters
        self.use_curriculum = self.train_params.get('use_curriculum', False)
        self.num_phases = self.train_params.get('num_phases', 5)
        self.initial_phase_proportion = self.train_params.get('initial_phase_proportion', 0.1)
        self.max_phase_proportion = self.train_params.get('max_phase_proportion', 0.5)

    def get_curriculum_dataset(self, labeled_dataset, pseudo_labeled_dataset, proportion):
        labeled_size = int(len(labeled_dataset) * (1 - proportion))
        pseudo_size = int(len(pseudo_labeled_dataset) * proportion)
        labeled_indices = list(range(labeled_size))
        pseudo_indices = list(range(pseudo_size))
        combined_dataset = ConcatDataset([
            Subset(labeled_dataset, labeled_indices),
            Subset(pseudo_labeled_dataset, pseudo_indices)
        ])
        return combined_dataset

    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def save_model(self, model, mod_name):
        path = mod_name + "weights.pth"
        torch.save(model.state_dict(), path)

    def visualize_samples(self, dataset, data_type, path=''):
        vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
                         mode=data_type, idx_list=[],
                         samples=10, cols=5, random_img=True,
                         save_path=path)
        idx_list = vis_.visualize_samples(transform=0)
        vis_.idx_list = idx_list
        vis_.random_img = False
        vis_.visualize_samples(transform=1)
        vis_.visualize_histograms(transform=0)
        vis_.visualize_histograms(transform=1)

    def predict_proba(self, dataset, model, criterion, device, out_neurons):
        model.eval()
        pred_prob = []
        _lab = []
        total_loss = 0
        batches = 0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(dataset), total=len(dataset))
            for i, (X, y) in progress_bar:
                X, y = X.to(device), y.to(device)

                out = model(X)

                if out_neurons == 1:
                    loss = criterion(out, y.unsqueeze(1))
                    prob = torch.sigmoid(out)
                else:
                    loss = criterion(out, y.long())
                    prob = torch.softmax(out.data, 1)

                _lab.extend(y.detach().cpu().numpy().tolist())
                pred_prob.extend(prob.detach().cpu().numpy().tolist())

                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / batches

        return _lab, pred_prob, avg_loss

    def train(self):
        # Training params
        epochs = self.train_params['epochs']
        device = self.train_params['device']
        batch_size = self.train_params['batch_size']
        patience = self.train_params['patience']
        save_path = self.train_params['save_path']
        optimizer_type = self.train_params['optimizer']
        lr = self.train_params['lr']
        scheduler_type = self.train_params['scheduler']
        lr_decay_factor = self.train_params['lr_decay_factor']
        weight_decay = self.train_params['weight_decay']
        loss_type = self.train_params['loss']
        tr_data_path = self.train_params['tr_path']
        pseudo_path = self.train_params['pseudo_path']
        vd_data_path = self.train_params['vd_path']
        num_folds = self.train_params['num_folds']
        lr_iterations = self.train_params['lr_iterations']
        seed = self.train_params['seed']
        use_amp = self.train_params.get('amp', False)  # AMP flag
        early_stopping = self.train_params.get('early_stopping', False)  # Early Stopping flag

        # Transform params
        normalize = self.trans_params['normalize']
        img_height = self.trans_params['height']
        img_width = self.trans_params['width']
        augmentation = self.trans_params.get('augmentation', False)  # Augmentation flag

        # torch.cuda.empty_cache()

        # Initialize GradScaler for AMP
        scaler = GradScaler() if use_amp else None

        device = torch.device(device)

        reproducibility(seed)

        # Visualization
        vis_samples = self.vis_params.vis
        vis_path = self.vis_params.path

        # Model params
        fr_layer = self.model_params['fr_layer']
        out_neurons = self.model_params['out_neurons']
        model_name = self.model_params['model_name']
        pretrained = self.model_params['pretrained']

        model_params = dict(fr_layer=fr_layer,
                            out_neurons=out_neurons,
                            model_name=model_name,
                            pretrained=pretrained)

        model_builder = BuildModel(**model_params)

        # Curriculum learning proportions
        phase_proportions = np.linspace(self.initial_phase_proportion, self.max_phase_proportion, self.num_phases)

        for fold in tqdm(range(num_folds)):
            # Directories
            inst_name = self.model_params['model_name']
            version = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.curr_version = version
            logs_dir = save_path + inst_name + "/" + version + "/" + f"fold{fold}" + "/"
            self.checkncreate_dir(logs_dir)
            print('\nModel version: {}'.format(version))

            for phase in range(self.num_phases if self.use_curriculum else 1):
                current_proportion = phase_proportions[phase] if self.use_curriculum else 0
                phase_name = f"phase_{phase + 1}" if self.use_curriculum else "no_curriculum"
                print(f"Phase {phase + 1}/{self.num_phases}, Proportion of pseudolabeled data: {current_proportion:.2f}")

                wandb.init(
                    mode=WANDB_MODE,
                    # Set the project where this run will be logged
                    project="BAC_project",
                    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                    name=f"{inst_name}_{version}_{fold}_{phase_name}",
                    # Track hyperparameters and run metadata
                    config={
                        "architecture": model_name,
                        "learning_rate": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "patience": patience,
                        "optimizer": optimizer_type,
                        "scheduler": scheduler_type,
                        "loss": loss_type,
                        "num_folds": num_folds,
                        "seed": seed,
                        "img_height": img_height,
                        "img_width": img_width,
                        "fr_layer": fr_layer,
                        "normalize": normalize,
                        "lr_iterations": lr_iterations,
                        "augmentation": augmentation,
                        "amp": use_amp,
                        "early_stopping": early_stopping,
                        "weight_decay": weight_decay,
                        "curriculum_proportion": current_proportion,
                    })

                model = model_builder.get_model()
                model = model.to(device)

                # Log the network weight histograms (optional)
                wandb.watch(model)

                print(model)

                train_data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=tr_data_path, fold=fold)
                valid_data_builder = BuildData(self.trans_params, batch_size, shuffle=False, path=vd_data_path, fold=fold)

                train_dataset_orig = train_data_builder.build_data('train')
                datasets = [train_dataset_orig]

                if augmentation:
                    train_dataset_hflip = train_data_builder.build_data('train_hflip')
                    datasets.append(train_dataset_hflip)

                # Use ConcatDataset only if there's more than one dataset to concatenate
                if len(datasets) > 1:
                    train_dataset = ConcatDataset(datasets)
                else:
                    train_dataset = datasets[0]

                valid_dataset = valid_data_builder.build_data('valid')

                train_loader = train_data_builder.build_loader(train_dataset)
                valid_loader = valid_data_builder.build_loader(valid_dataset)

                if self.use_curriculum:
                    pseudo_data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=pseudo_path, fold=fold)
                    pseudo_labeled_dataset = pseudo_data_builder.build_data('pseudo_train')
                    current_dataset = self.get_curriculum_dataset(train_dataset, pseudo_labeled_dataset, current_proportion)
                    train_loader = train_data_builder.build_loader(current_dataset)

                n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)

                class_weights = train_dataset_orig.get_class_weights()
                class_weights = torch.tensor(class_weights).float().to(device)

                # Visualize Samples
                if vis_samples:
                    self.visualize_samples(train_dataset_orig, 'train', vis_path)
                    if augmentation:
                        self.visualize_samples(train_dataset_hflip, 'train_hflip', vis_path)
                    self.visualize_samples(valid_dataset, 'valid', vis_path)

                # Optimizer and Scheduler
                optim = Optimizer(optim=optimizer_type,
                                  lr=lr,
                                  scheduler=scheduler_type,
                                  patience=patience,
                                  factor=lr_decay_factor,
                                  wd=weight_decay,
                                  iterations=lr_iterations,
                                  dataloader=train_loader,
                                  epochs=epochs,
                                  params=model.parameters())

                optimizer = optim.get_optimizer()
                scheduler = optim.get_scheduler()

                # Loss function
                crit = Loss(loss=loss_type,
                            weights=class_weights)

                criterion = crit.get_loss()

                # Results
                best_loss = 99999.0
                trigger = 0

                for epoch in tqdm(range(epochs)):
                    model.train()
                    loss_epoch = 0
                    itr = 0
                    y_proba = []
                    y_labels = []

                    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
                    for i, (X, y) in progress_bar:
                        X, y = X.to(device), y.to(device)

                        optimizer.zero_grad()

                        # AMP or normal execution
                        if use_amp:
                            with autocast():
                                outputs = model(X)
                                if out_neurons == 1:
                                    loss = criterion(outputs, y.unsqueeze(1))
                                    prob = torch.sigmoid(outputs)
                                else:
                                    loss = criterion(outputs, y.long())
                                    prob = torch.softmax(outputs.data, 1)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(X)
                            if out_neurons == 1:
                                loss = criterion(outputs, y.unsqueeze(1))
                                prob = torch.sigmoid(outputs)
                            else:
                                loss = criterion(outputs, y.long())
                                prob = torch.softmax(outputs.data, 1)
                            loss.backward()
                            optimizer.step()

                        loss_epoch += loss.item()
                        y_proba.extend(prob.detach().cpu().numpy())
                        y_labels.extend(y.detach().cpu().numpy())

                    # Post-epoch: Calculate metrics, evaluate model, log to W&B...
                    tr_loss = loss_epoch / len(progress_bar)
                    tr_model_eval = EvaluateModel(y_labels, y_proba).scores

                    metrics = {"train/train_loss": tr_loss,
                               "train/epoch": (i + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                               "train/train_accuracy": tr_model_eval['acc'],
                               "train/auc": tr_model_eval['auroc'],
                               "train/aupr": tr_model_eval['aupr'],
                               "train/f1": tr_model_eval['F1'],
                               "train/mcc": tr_model_eval['MCC'],
                               "train/precision": tr_model_eval['precision'],
                               "train/recall": tr_model_eval['recall'],
                               }

                    # Validation Data Evaluation
                    y_vd_labels, y_vd_proba, vd_loss = self.predict_proba(valid_loader, model, criterion, device, out_neurons)

                    vd_model_eval = EvaluateModel(y_vd_labels, y_vd_proba).scores

                    val_metrics = {"val/val_loss": vd_loss,
                                   "val/val_accuracy": vd_model_eval['acc'],
                                   "val/auc": vd_model_eval['auroc'],
                                   "val/aupr": vd_model_eval['aupr'],
                                   "val/f1": vd_model_eval['F1'],
                                   "val/mcc": vd_model_eval['MCC'],
                                   "val/precision": vd_model_eval['precision'],
                                   "val/recall": vd_model_eval['recall'],
                                   }

                    wandb.log({**metrics, **val_metrics})

                    print('Epoch {}, train loss {}, valid loss {}, valid auc {}'.format(epoch + 1, tr_loss, vd_loss, vd_model_eval['auroc']))

                    if best_loss > vd_loss:
                        self.save_model(model, logs_dir)
                        best_loss = vd_loss
                        best_model = model
                        trigger = 0
                    else:
                        trigger += 1
                        print('Trigger count: {}'.format(trigger))

                    if early_stopping and trigger > patience:
                        print('Early stopping! ' + inst_name + '_' + version + '_' + f'{fold}_{phase_name}')
                        wandb.finish()
                        break

                    scheduler.step(vd_loss)

                print('Ending training for -- ' + inst_name + '_' + version + '_' + f'{fold}_{phase_name}')
                wandb.finish()

            print('Training completed for all folds')
