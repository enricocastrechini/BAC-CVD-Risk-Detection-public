# SparK Masked Autoencoder

This component adapts the SparK masked autoencoder for convolutional encoders. It masks image patches during pretraining and reconstructs the missing content from the visible patches. The resulting encoder can be evaluated in the downstream BAC classification notebooks.

## Run

Install `requirements.txt`, then run from this directory with Bash, Git Bash or WSL:

```bash
export BAC_DATA_DIR=/path/to/authorized/mammograms
export BAC_OUTPUT_DIR=/path/to/outputs/spark
bash run_exp.sh
```

The committed experiment uses ConvNeXt-Small, input dimensions `1120 x 576`, mask ratio `0.6`, 800 epochs and one CUDA device. Set `BAC_RESUME_CHECKPOINT` to resume an existing local checkpoint. Adjust these values in `run_exp.sh` only after recording the resulting configuration.

## Data

The upstream implementation expected an image-folder layout with training and validation data. Confirm the loader and directory layout before starting a full run; the private mammography data is not part of this repository. Keep images, checkpoints and reconstructed samples outside version control.

## Reference

The implementation is based on [SparK](https://github.com/keyu-tian/SparK) and the paper *Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling* by Tian et al. (ICLR 2023). This pipeline's reported downstream result was AUC-ROC 0.50; it should be treated as an experimental negative result, not a clinical conclusion.
