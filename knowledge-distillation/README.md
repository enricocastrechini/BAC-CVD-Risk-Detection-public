# Knowledge Distillation

This pipeline adapts [RepDistiller](https://github.com/HobbitLong/RepDistiller) to binary BAC classification on mammograms. A teacher network provides softened predictions and, for representation-based methods, intermediate features to a student network. Available objectives include KD, CRD, FitNet, Attention Transfer, RKD, PKT, VID and other methods implemented in `distiller_zoo/`.

## Data and environment

The private dataset is not included. The active mammography loader is `dataset/mammo_bac.py`; it reads the project-specific labeled and unlabeled image directories used by the training code. Confirm the local data layout and paths before running. `dataset/cifar100.py` and `dataset/imagenet.py` are kept as upstream reference loaders and are not the BAC experiment path.

Install the dependencies listed in `constraints.txt` into a PyTorch environment. Run commands from this directory. See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for private-data, checkpoint and output rules.

## 1. Train a teacher

`train_teacher.py` trains a binary mammogram classifier. The default model is `resnet110`; supported models are listed in `parse_option()`.

```powershell
cd knowledge-distillation
python train_teacher.py --dataset mammo --model resnet110 --epochs 240 --trial 1
```

Teacher checkpoints are written below `save/models/` by default. Record the exact checkpoint path and configuration before using it for distillation.

## 2. Train a student

Pass the teacher checkpoint with `--path_t`, choose a supported student with `--model_s`, and select the distillation method with `--distill`:

```powershell
python train_student.py `
  --dataset mammo `
  --path_t .\save\models\<teacher>\<checkpoint>.pth `
  --model_s resnet8 `
  --distill kd `
  -r 1.0 -a 0.9 -b 0 `
  --trial 1
```

For CRD, use `--distill crd` and tune `--nce_k`, `--nce_t` and `--nce_m` for the available dataset size. `-r`, `-a` and `-b` weight the classification, KD and additional distillation losses respectively. Student checkpoints and TensorBoard logs are written below `save/student_model/` and `save/student_tensorboards/`.

## Code map

- `dataset/mammo_bac.py`: mammography loaders, including CRD instance sampling.
- `helper/loops.py`: training and validation loops.
- `distiller_zoo/`: feature and prediction distillation losses.
- `crd/`: contrastive representation distillation criterion and memory bank.
- `models/`: teacher and student architectures.
- `scripts/`: upstream CIFAR-oriented examples; adapt them before reuse.

## Provenance

The distillation implementations originate from RepDistiller and the CRD paper by Tian, Krishnan and Isola (ICLR 2020). This repository changes the dataset integration, binary classification setup and experiment defaults for the BAC task. The upstream benchmark tables are intentionally not repeated here because they do not describe the mammography experiments.
