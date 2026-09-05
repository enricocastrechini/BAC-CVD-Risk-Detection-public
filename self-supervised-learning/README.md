# Self-Supervised Learning

This pipeline explores whether unlabeled mammograms can provide useful representations for BAC classification. It contains two stages:

1. self-supervised pretraining on unlabeled images;
2. supervised downstream fine-tuning on the labeled BAC dataset.

The code is adapted from [SSL-MedicalImaging-CL-MAE](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE), which originally targets CT data. In this repository, the documented experiment path uses the mammography data variables and image dimensions shown below. The original CT/LIDC instructions are retained only as method provenance, not as a public dataset requirement.

## Directory guide

- [`Pre-Training/`](Pre-Training/): preprocessing, contrastive methods (SwAV, MoCoV2, BYOL) and SparK masked autoencoder training.
- [`Downstream/`](Downstream/): notebooks and dependencies for fine-tuning and evaluating pretrained encoders.
- [`Pre-Training/Data_Preprocessing/`](Pre-Training/Data_Preprocessing/): upstream DICOM-to-PNG utilities; use only with authorized data.

## Environment and data

The private mammography images and checkpoints are not included. Set the following variables in the shell before running the supplied scripts:

```powershell
$env:BAC_DATA_DIR = "D:\authorized\mammograms"
$env:BAC_OUTPUT_DIR = "D:\experiments\bac-ssl"
```

`BAC_DATA_DIR` must point to the local image directory expected by the selected loader. `BAC_OUTPUT_DIR` is optional and defaults to a local `results/` folder. Use `WANDB_MODE=offline` for local logging; provide `WANDB_API_KEY` only via the environment when online logging is required.

## Pretraining

For SparK, run from `Pre-Training/Masked_Autoencoder/` in Bash, Git Bash or WSL:

```bash
export BAC_DATA_DIR=/path/to/authorized/mammograms
export BAC_OUTPUT_DIR=/path/to/outputs/spark
bash run_exp.sh
```

The script uses ConvNeXt-Small, mammography input dimensions `1120 x 576`, 800 epochs and the optional `BAC_RESUME_CHECKPOINT`. For contrastive pretraining, use the method-specific scripts in `Pre-Training/Contrastive_Learning/`.

## Downstream evaluation

Open the relevant notebook in `Downstream/` after installing its `requirements.txt`. Set the local data, checkpoint and output paths in the first parameter cell, then run the notebook with the authorized labeled BAC split. Keep the encoder checkpoint, notebook parameters, seed and dependency versions with every reported result.

## Results and limitations

In the reported experiments, the SparK reconstruction objective reached a low loss but downstream BAC classification remained around chance level (AUC-ROC 0.50). This is a research result, not a claim that self-supervised learning cannot work for mammography; pretraining duration, augmentations, domain matching and checkpoint selection remain open variables.

## References

- Wolf et al., *Self-supervised pre-training with contrastive and masked autoencoder methods for dealing with small datasets in deep learning for medical imaging*, Scientific Reports (2023).
- Tian et al., *Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling* (SparK, ICLR 2023).
