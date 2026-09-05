# Contrastive Learning

This component provides SwAV, MoCoV2 and BYOL pretraining through the adapted PyTorch Lightning/Bolts modules under `pl_bolts/`. It is intended to learn an encoder from unlabeled images before downstream BAC fine-tuning.

## Run

Install the dependencies from `requirements.txt`, set the data and output directories, then run the method-specific script from Bash, Git Bash or WSL:

```bash
export BAC_DATA_DIR=/path/to/authorized/mammograms
export BAC_OUTPUT_DIR=/path/to/outputs/byol
bash run_exp.sh
```

The committed script currently launches BYOL. The SwAV and MoCoV2 modules in `pl_bolts/models/self_supervised/` expose their own command-line arguments; inspect them before launching a different method. Keep W&B credentials in the environment and use `WANDB_MODE=offline` for local runs.

## Data and provenance

The original project used preprocessed LIDC CT slices. This repository's adaptation supplies a mammography data directory instead, but the exact loader layout must be checked against the selected module before a full run. Do not assume that every upstream augmentation or channel convention is valid for mammograms.

The implementation is based on PyTorch Lightning Bolts and the SSL-MedicalImaging-CL-MAE project. Record the method, checkpoint, data version, seed and dependency versions for each experiment.
