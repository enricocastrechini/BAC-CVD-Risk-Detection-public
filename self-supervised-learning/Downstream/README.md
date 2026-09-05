# Downstream BAC Evaluation

The notebooks in this directory fine-tune a pretrained encoder on the labeled BAC mammography task. They are the second stage of the pipeline described in [`../README.md`](../README.md): pretraining uses unlabeled images, while this stage requires authorized labels and patient-level splits.

## Setup

Install the dependencies from `requirements.txt` in a compatible Python and PyTorch environment. Launch Jupyter, open the notebook matching the experiment, and set the data, checkpoint and output paths in its parameter cell before running the cells.

```bash
cd self-supervised-learning/Downstream
pip install -r requirements.txt
jupyter lab
```

The notebooks currently include `Brain.ipynb`, `COVID-19.ipynb` and `OrgMNIST.ipynb` inherited from the upstream project. For BAC experiments, adapt the data and label cells to the mammography CSV/layout used by the baseline and record that adaptation with the run. The notebook names alone do not imply that those public CT tasks are part of this repository's BAC dataset.

## Reproducibility

Record the pretrained method, checkpoint provenance, labeled split, seed, framework versions and output directory. Use offline W&B logging unless online logging is explicitly required. Do not commit patient data, metadata, checkpoints, credentials or notebook outputs containing private information.

See the root [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) for the shared publication boundary and validation levels.
