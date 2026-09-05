# Vanilla Pretrained

This is the supervised reference pipeline for BAC classification. It fine-tunes ImageNet-pretrained CNN and transformer architectures on labeled mammograms and provides the baseline used to compare pseudolabeling, distillation and self-supervised pretraining.

## Before running

The mammography images and annotations are private and are not included in this repository. Prepare CSV files containing an image path and a binary BAC label, then update the paths in `src/config/cfg_train.yaml` and `src/config/cfg_test.yaml`.

Use patient-level train/validation/test splits. Do not place patient data, checkpoints or credentials in the repository. See the root [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the project-wide data and output policy.

## Environment

Create a Python environment with a PyTorch build appropriate for the available hardware and install the dependencies required by the selected models. Run commands from this directory so the relative configuration paths resolve:

```powershell
cd vanilla-pretrained
$env:WANDB_MODE = "offline"
```

Set `WANDB_MODE=online` only when external logging is intentional, and provide `WANDB_API_KEY` through the environment rather than source files.

## Train and test

```powershell
python train_caller.py
python test_caller.py
```

The callers load the YAML configuration, construct the model and dataloaders, and delegate to `src/utils/train.py` and `src/utils/test.py`.

## Code map

- `src/data/`: CSV dataset loading, mammogram preprocessing and transforms.
- `src/models/models.py`: model construction and optional layer freezing.
- `src/utils/loss.py`: binary and multiclass loss functions, including weighted losses.
- `src/utils/eval.py`: ROC/PR metrics and threshold selection.
- `src/utils/interpret.py`: Grad-CAM-based interpretation utilities.
- `src/utils/aggregate.py`: image-, exam- and patient-level aggregation.

## Outputs

Runs are written to `results/<model_name>/<timestamp>/`, including the best weights, logs, predictions, metric files and Grad-CAM outputs. Generated results are ignored by Git and should be retained with the configuration, dependency versions, seed and checkpoint provenance needed to interpret them.
