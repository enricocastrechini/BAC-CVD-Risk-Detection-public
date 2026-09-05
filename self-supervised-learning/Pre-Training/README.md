# Self-Supervised Pre-Training

Pretraining learns an encoder from unlabeled mammograms before downstream BAC fine-tuning. The available families are:

- contrastive learning: SwAV, MoCoV2 and BYOL;
- masked autoencoding: SparK for convolutional networks.

## Data contract

Set `BAC_DATA_DIR` to an authorized local image directory and keep all outputs outside version control. The scripts in this directory use the adapted mammography loaders and experiment settings; the original upstream LIDC/CT-oriented documentation is not the target data contract for this repository.

## Components

- [`Data_Preprocessing/`](Data_Preprocessing/): upstream DICOM slice conversion, useful only when the source data is authorized and compatible.
- [`Contrastive_Learning/`](Contrastive_Learning/): PyTorch Lightning/Bolts implementations and `run_exp.sh` entry points.
- [`Masked_Autoencoder/`](Masked_Autoencoder/): SparK implementation, launcher and `run_exp.sh` entry point.

## Quick start

From this directory, use Bash, Git Bash or WSL:

```bash
export BAC_DATA_DIR=/path/to/authorized/mammograms
export BAC_OUTPUT_DIR=/path/to/outputs
cd Masked_Autoencoder
bash run_exp.sh
```

For contrastive methods, select the corresponding script under `Contrastive_Learning/`. Check its `requirements.txt` and record the PyTorch, CUDA, Lightning and Bolts versions used for the run.

Pretrained encoders are consumed by the notebooks in [`../Downstream/`](../Downstream/).
