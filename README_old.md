# BAC-CVD-Risk-Detection

This repository collects the experimental work carried out for my thesis/research activity on detecting **Breast Arterial Calcifications (BAC)** from mammograms, as a possible surrogate marker of cardiovascular disease (CVD) risk. The goal is to train image classification models in a realistic setting of a **small and imbalanced medical dataset**, comparing three different training strategies:

1. **Supervised baseline** with ImageNet-pretrained models ([vanilla-pretrained/](vanilla-pretrained/))
2. **Knowledge Distillation** to transfer knowledge from a larger "teacher" model to a lighter/more efficient "student" model ([knowledge-distillation/](knowledge-distillation/))
3. **Self-Supervised Learning** to pre-train the encoder on unlabeled mammography images, prior to fine-tuning on the supervised task ([self-supervised-learning/](self-supervised-learning/))

The last two folders start from two public research repositories, which I adapted and reconfigured for the mammography dataset and for the specific BAC/cardiovascular risk classification task (dataloaders, preprocessing, models, training/testing scripts). The original code (methods, losses, architectures) remains substantially the one proposed by the authors cited below; the contribution of this repo is the adaptation to mammography imaging and the integration of the three pipelines into a single experimental comparison project.

> **Research status:** This is an experimental research repository, not a clinical decision-support tool. The private mammography dataset, patient metadata, model checkpoints, and experiment outputs are intentionally excluded. Reproduction therefore requires authorized access to compatible data and the environment described by each pipeline.

## Public-release guidance

- Start with the pipeline-specific README before running a training script.
- Use [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for data-path conventions, environment variables, and validation levels.
- Replace local dataset and checkpoint paths with paths on your machine; personal filesystem paths are not part of the public project contract.
- Keep credentials such as `WANDB_API_KEY` in the environment. The baseline pipeline uses offline W&B logging by default; set `WANDB_MODE=online` only when external logging is intentional.
- Put safe diagrams and verified figures under [`docs/assets/`](docs/README.md), with captions and provenance. Do not publish patient images or unverified results.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and [`CITATION.cff`](CITATION.cff) for repository policies.

## Why compare these three techniques

In the medical domain, and in particular for a task such as BAC detection, the main bottleneck is the scarcity of annotated data: annotations require expert radiologists and positive cases are relatively rare. The three strategies address this problem from different angles:

- The **pretrained baseline** (standard "vanilla" transfer learning from ImageNet) is the standard reference point.
- **Knowledge distillation** makes it possible to leverage a teacher model (with greater capacity or pretrained on more data) to guide the training of a student model, improving its performance for the same amount of available annotated data.
- **Self-supervised pre-training** leverages *unlabeled* mammography images (far more abundant) to learn useful representations, later refined with a handful of labeled images.

Comparing these approaches makes it possible to evaluate which strategy is more effective/robust when the available annotated dataset is small.

---

## 1. [vanilla-pretrained/](vanilla-pretrained/)

Standard supervised classification pipeline: CNN/Transformer models pretrained on ImageNet (via `torchvision`/`timm`), fine-tuned on the mammography dataset for BAC classification. It represents the comparison baseline for the other two techniques.

**Main structure:**
- [train_caller.py](vanilla-pretrained/train_caller.py) / [test_caller.py](vanilla-pretrained/test_caller.py): entry points that read the configuration files (`config/`) and launch training and testing respectively.
- [src/models/models.py](vanilla-pretrained/src/models/models.py): model definitions (`BuildModel`), with support for selective layer freezing, parameter counting, etc.
- [src/data/](vanilla-pretrained/src/data/): CSV-based dataset/dataloader (image path + label) and transformation/preprocessing pipeline.
- [src/utils/](vanilla-pretrained/src/utils/): loss functions (weighted/unweighted, binary/multi-class), evaluation metrics (optimal threshold via PR/ROC curve), Grad-CAM for interpretability, TensorBoard/Weights & Biases logging.

**Output:** best model weights, training/validation logs, Grad-CAM and prediction images, per-epoch CSV files with probabilities and true labels, saved under `results/<model_name>/<timestamp>/`.

See the folder's [README](vanilla-pretrained/README.md) for details on setup, commands and execution.

---

## 2. [knowledge-distillation/](knowledge-distillation/)

Adaptation of [**RepDistiller**](https://github.com/HobbitLong/RepDistiller) (Tian, Krishnan, Isola — *"Contrastive Representation Distillation"*, ICLR 2020), which implements and benchmarks 12+ state-of-the-art knowledge distillation methods (classic KD, FitNet, Attention Transfer, CRD, VID, RKD, PKT, FSP, NST, etc.).

**What I adapted:**
- The original CIFAR-100/ImageNet dataloaders ([dataset/cifar100.py](knowledge-distillation/dataset/cifar100.py), [dataset/imagenet.py](knowledge-distillation/dataset/imagenet.py)) are complemented by a dedicated mammography dataloader, [dataset/mammo_bac.py](knowledge-distillation/dataset/mammo_bac.py) (`MammogramDataset`/`MammogramDatasetInstance` classes), which reads images and labels from CSV files (train/valid/test) instead of the standard benchmarks, with support for instance/contrastive sampling (needed for CRD).
- [train_teacher.py](knowledge-distillation/train_teacher.py) and [train_student.py](knowledge-distillation/train_student.py) were rewired to the new mammography dataloader (`get_mammogram_dataloaders`) instead of CIFAR-100, keeping the general training/distillation logic unchanged (`helper/loops.py`, `distiller_zoo/`, `crd/`).
- Default hyperparameters (batch size, learning rate, epochs) recalibrated for a much smaller dataset and for high-resolution medical images compared to CIFAR-100.

**Core idea:** a "teacher" model (e.g., pretrained with more capacity or on more data, potentially the model from the vanilla-pretrained pipeline) guides the training of a more compact "student", transferring not only the labels but also structural information about the internal representations (relevant in the case of CRD, based on contrastive learning between teacher and student features).

See the folder's [original README](knowledge-distillation/README.md) for training commands (`train_teacher.py`, `train_student.py`) and the full list of available distillation methods.

---

## 3. [self-supervised-learning/](self-supervised-learning/)

Adaptation of [**SSL-MedicalImaging-CL-MAE**](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE) (Wolf et al., *"Self-supervised pre-training with contrastive and masked autoencoder methods for dealing with small datasets in deep learning for medical imaging"*, Scientific Reports 2023), which compares self-supervised pre-training methods (contrastive learning: SwAV, MoCo, BYOL; masked autoencoder: SparK) on medical images (CT), then evaluates their effectiveness when fine-tuned on small datasets.

**Structure:**
- [Pre-Training/](self-supervised-learning/Pre-Training/): self-supervised pre-training code, split into `Data_Preprocessing`, `Contrastive_Learning` (SwAV/MoCo/BYOL) and `Masked_Autoencoder` (SparK).
- [Downstream/](self-supervised-learning/Downstream/): fine-tuning and evaluation notebooks that load the pretrained checkpoints and adapt them to the final classification task.

**What I adapted:** the pre-training logic and architectures (ResNet50 with SwAV/MoCo/BYOL/SparK) remain those of the original repo, designed for CT images; my contribution is adapting the downstream flow and dataloaders to the mammography dataset for BAC classification, reusing the pretrained checkpoints as encoder initialization to be fine-tuned on the target task, following the same "pre-training on unlabeled images → fine-tuning on a few labeled images" approach described in the original paper.

See the folder's [README](self-supervised-learning/README.md) for details on the methods, the available pretrained checkpoints, and usage instructions.

---

## Repository structure

```
BAC-CVD-Risk-Detection/
├── vanilla-pretrained/        # baseline: transfer learning from ImageNet
├── knowledge-distillation/    # from RepDistiller (Tian et al., ICLR 2020) — adapted to mammography
└── self-supervised-learning/  # from SSL-MedicalImaging-CL-MAE (Wolf et al., Sci Rep 2023) — adapted to mammography
```

## Credits

- Knowledge Distillation: [RepDistiller](https://github.com/HobbitLong/RepDistiller) — Yonglong Tian, Dilip Krishnan, Phillip Isola, *"Contrastive Representation Distillation"*, ICLR 2020.
- Self-Supervised Learning: [SSL-MedicalImaging-CL-MAE](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE) — Daniel Wolf et al., *"Self-supervised pre-training with contrastive and masked autoencoder methods for dealing with small datasets in deep learning for medical imaging"*, Scientific Reports 13, 2023.

Both repositories were adapted (dataloaders, preprocessing, and training/testing scripts) for the mammography dataset and the BAC detection task as part of my research/thesis work.
