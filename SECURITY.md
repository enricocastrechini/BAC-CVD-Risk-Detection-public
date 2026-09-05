# Security and Sensitive Data

This repository contains research code for medical-image analysis. It does not include the private mammography dataset, patient identifiers, model checkpoints, or experiment outputs.

## Do not publish

- patient images, DICOM headers, identifiers, or private CSV files;
- API keys, tokens, passwords, or local credential files;
- model checkpoints or generated artifacts unless their redistribution rights are clear;
- logs or screenshots containing sensitive paths or metadata.

The training and testing pipelines use Weights & Biases optionally. Keep `WANDB_API_KEY` in the environment and use `WANDB_MODE=offline` for local runs that should not sync externally.

## Reporting a problem

For a suspected credential leak or sensitive-data exposure, do not open a public issue. Contact the repository maintainer privately and include the affected path and commit without reproducing the secret.
