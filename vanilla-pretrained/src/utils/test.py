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

from src.data.databuilder import BuildData
from src.utils.visualize import Visualize
from src.models.models import BuildModel
from src.utils.eval import EvaluateModel
from src.utils.interpret import Interpret
import torch
import pandas as pd
import os
import pickle
from src.utils.reproducibility import reproducibility
import wandb
from tqdm import tqdm
import csv

WANDB_API_KEY = os.getenv('WANDB_API_KEY')
WANDB_MODE = os.getenv('WANDB_MODE', 'offline')
if WANDB_API_KEY and WANDB_MODE == 'online':
    wandb.login(anonymous='never', key=WANDB_API_KEY)


class TestModel:
    def __init__(self, test_params, model_params, vis_params, trans_params):
        self.test_params = test_params # Testing parameters from the config file
        self.model_params = model_params # Model parameters from the config file
        self.vis_params = vis_params # Visualization of samples
        self.trans_params = trans_params # Preprocessing parameters

    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def visualize_samples(self, dataset, data_type, path):
        vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
                                mode=data_type, idx_list=[], 
                                samples=10, cols=5, random_img = True, 
                                save_path=path)
        idx_list = vis_.visualize_samples(transform=0)
        vis_.idx_list = idx_list
        vis_.random_img = False
        vis_.visualize_samples(transform=1)
        vis_.visualize_histograms(transform=0)
        vis_.visualize_histograms(transform=1)
    
    def visualize_predictions(self, dataset, data_type, y_pred, y_prob, path):
        vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
                                mode=data_type, idx_list=[], 
                                samples=2, cols=5, random_img = True, 
                                save_path=path)
        idx_list = vis_.visualize_predictions(y_pred, y_prob)
        return idx_list

    def predict_proba(self, dataset, model, device, out_neurons):
        model.eval()
        pred_prob = []
        pred = []
        _lab = []
        batches = 0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(dataset), total=len(dataset))
            for i, (X, y) in progress_bar:
                X , y = X.to(device), y.to(device)

                out = model(X)

                if out_neurons == 1:
                    prob = torch.sigmoid(out)
                else:
                    prob = torch.softmax(out.data, 1)
                
                _lab.extend(y.detach().cpu().numpy().tolist())
                pred_prob.extend(prob.detach().cpu().numpy().tolist())

                batches += 1
        
        return _lab, pred_prob


    def save_predictions_to_csv(self, labels, predictions, save_path, test_path, fold):
        
        test_path = test_path[0]

        # Read the test CSV file
        test_df = pd.read_csv(test_path)
        
        # Filter rows based on the 'fold' column
        test_df = test_df[test_df['fold'] == fold]
        
        # Extract columns from the filtered test CSV
        test_columns = test_df.columns.tolist()
        
        # Initialize data_for_csv with label and prediction columns
        data_for_csv = [['label', 'prediction']]

        # Append test columns to data_for_csv
        data_for_csv[0].extend(test_columns)

        # Append label and prediction values
        for i in range(len(labels)):
            data_for_csv.append([labels[i], predictions[i]] + [test_df.iloc[i][col] for col in test_columns])

        # Create CSV file path
        csv_file_path = os.path.join(save_path, "predictions.csv")

        # Write data to CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_for_csv)

        # Save CSV file to Weights & Biases
        wandb.save(csv_file_path)
        print("Predictions saved to:", csv_file_path)

    def test(self):
        # Testing params
        device = self.test_params['device']
        batch_size = self.test_params['batch_size']
        save_path = self.test_params['save_path']
        data_path = self.test_params['ts_path']
        interpret_layer = self.test_params['interpret_layer']
        seed = self.test_params['seed']
        fold = self.test_params['fold']
        
        device = torch.device(device)

        reproducibility(seed)

        # Model params
        out_neurons=self.model_params['out_neurons']
        model_name=self.model_params['model_name']
        model_weights = self.model_params['model_weights']

        # Visualization
        vis_samples = self.vis_params.vis
        vis_path = self.vis_params.path
        
        model_params = dict(out_neurons=out_neurons,
                            model_name=model_name)

        model_builder = BuildModel(**model_params)


        exp = self.test_params['save_path'].split('/')[-2]
        
        wandb.init(
            mode=WANDB_MODE,
                # Set the project where this run will be logged
                project="BAC_project", 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=f"{model_name}_{exp}_test_{fold}", 
                # Track hyperparameters and run metadata
                  config={
                  "architecture": model_name,
                  "batch_size": batch_size,
                  "seed": seed,
                  })


        model = model_builder.get_model()

        # Load weights
        model.load_state_dict(torch.load(model_weights))

        # state_dict = torch.load(model_weights)
        # state_dict = state_dict["model"]
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        # model.load_state_dict(model_dict, strict=False)

        model.to(device)

        # Data loader
        data_builder = BuildData(self.trans_params, batch_size, shuffle=False, path=data_path, fold=fold)
        test_dataset = data_builder.build_data('test')
        test_loader = data_builder.build_loader(test_dataset)

        # Visualize Samples
        if vis_samples:
            self.visualize_samples(test_dataset, 'test', vis_path)
            
        # Data Evaluation
        y_ts_labels, y_ts_proba = self.predict_proba(test_loader, model, device, out_neurons)


        # Save predictions to CSV
        self.save_predictions_to_csv(y_ts_labels, y_ts_proba, save_path, data_path, fold)

        
        ts_model_eval = EvaluateModel(y_ts_labels, y_ts_proba).scores
        # index_list = self.visualize_predictions(test_dataset, 'test', ts_model_eval['y_pred'], ts_model_eval['y_prob'], save_path)



        metrics = {"test/auc": ts_model_eval['auroc'],
                    "test/aupr": ts_model_eval['aupr'],
                    }

        wandb.log(metrics)

        # Interpret.get_gradcams(test_dataset, model, ts_model_eval['y'], ts_model_eval['y_pred'], ts_model_eval['y_prob'], index_list, False, save_path, interpret_layer, device)