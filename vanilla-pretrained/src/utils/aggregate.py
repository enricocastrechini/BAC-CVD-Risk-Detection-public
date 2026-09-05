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

import os
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def aggregate_predictions(csv_folders, save_path=None):
    all_predictions = []

    # Iterate over all CSV folders
    for csv_folder in csv_folders:
        file_path = os.path.join(csv_folder, "predictions.csv")
        print("Reading file:", file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_predictions.append(df)
        else:
            print("File not found:", file_path)

    # Concatenate all predictions by row
    aggregated_predictions = pd.concat(all_predictions, axis=0, ignore_index=True)

    # Remove square brackets from predicted probabilities
    aggregated_predictions["prediction"] = aggregated_predictions["prediction"].str.strip("[]").astype(float)

    # Save concatenated file for debugging
    if save_path:
        aggregated_predictions.to_csv(save_path, index=False)
        print("Saved aggregated predictions to:", save_path)

    return aggregated_predictions

def calculate_patient_scores(predictions):
    print("\nCalculating patient scores...")

    # Group predictions by patient_id and calculate max prediction for each patient
    # patient_predictions = predictions.groupby('patient_id')['prediction'].max().reset_index()
    patient_predictions = predictions.groupby('patient_id')['prediction'].mean().reset_index()

    print("Patient predictions:")
    print(patient_predictions.head())
    print(len(patient_predictions))

    # Merge with true labels to calculate AUROC and AUPR
    patient_labels = predictions.groupby('patient_id')['label'].any().astype(int).reset_index()
    print("Patient labels:")
    print(patient_labels.head())

    patient_scores = pd.merge(patient_predictions, patient_labels, on='patient_id', how='left')
    print("Patient scores:")
    print(patient_scores.head())

    # Calculate AUROC and AUPR
    auroc = roc_auc_score(patient_scores['label'], patient_scores['prediction'])
    aupr = average_precision_score(patient_scores['label'], patient_scores['prediction'])
    print("Patient-based AUROC:", auroc)
    print("Patient-based AUPR:", aupr)

    return auroc, aupr

def calculate_image_metrics(predictions):
    print("\nCalculating image-based metrics...")

    # Assuming you have the ground truth labels and predictions
    y_true = predictions["label"]
    y_pred = predictions["prediction"]

    # Print unique labels and min/max prediction values
    # print("\nUnique Labels:")
    # print(y_true.unique())
    # print("\nMin Prediction Value:", y_pred.min())
    # print("Max Prediction Value:", y_pred.max())

    # Calculate aggregate metrics
    aupr = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    # Print aggregate metrics
    print("\nAggregate AUROC:", auroc)
    print("Aggregate AUPR:", aupr)

    return auroc, aupr

if __name__ == "__main__":
    # Define the folders containing predictions CSV files
    csv_folders = [
        "results/vgg16/2024-06-15_19-07-18",  # Update with the path to the folder containing predictions CSV files
        "results/vgg16/2024-06-15_19-07-18",  # Update with the path to the folder containing predictions CSV files
        "results/vgg16/2024-06-15_19-07-18",  # Update with the path to the folder containing predictions CSV files
        # Add more folders if needed...
    ]

    # Aggregate predictions and save concatenated file
    save_path = "aggregated_predictions.csv"
    aggregated_predictions = aggregate_predictions(csv_folders, save_path=save_path)

    print("\nAggregated Predictions:")
    print(aggregated_predictions.head())

    # Calculate AUROC and AUPR on a patient-based level
    patient_auroc, patient_aupr = calculate_patient_scores(aggregated_predictions)

    # Calculate classic aggregated metrics on an image-based level
    image_auroc, image_aupr = calculate_image_metrics(aggregated_predictions)
