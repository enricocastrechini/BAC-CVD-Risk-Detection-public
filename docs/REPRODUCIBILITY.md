# Reproducibility

The repository contains research code, not the private mammography dataset or trained checkpoints. Reproduction requires authorized access to compatible data and hardware.

## Baseline data layout

The baseline configuration expects CSV annotation files and image paths supplied by the user. Start from `vanilla-pretrained/src/config/` and replace the example paths with local paths before running training or testing.

## Environment variables

- `BAC_DATA_DIR`: dataset directory used by the knowledge-distillation mammogram loader and SSL launch scripts.
- `BAC_NEGATIVE_IMAGE_DIR`: authorized negative-image directory for the preprocessing script.
- `BAC_PSEUDO_WEIGHTS`: checkpoint used by the optional `convnextsmall_pseudo` model.
- `BAC_SPARK_WEIGHTS`: checkpoint used by the optional `convnextsmall_spark` model.
- `BAC_STUDENT_WEIGHTS`: checkpoint used by the optional `convnextsmall_student_2` model.
- `BAC_OUTPUT_DIR`: output directory used by the SSL launch scripts.
- `BAC_RESUME_CHECKPOINT`: optional masked-autoencoder checkpoint to resume from.
- `WANDB_MODE`: `offline` by default; use `online` only when external logging is intended.
- `WANDB_API_KEY`: W&B credential supplied through the environment for online logging.

Do not commit private data, credentials, checkpoints, or generated experiment directories. See [`SECURITY.md`](../SECURITY.md) for the publication boundary.

## Validation levels

1. Run the repository quality workflow or its commands locally for syntax and path checks.
2. Run a one-batch smoke test with an authorized fixture or dataset subset.
3. Run full training/evaluation only after recording the exact configuration, dependency versions, hardware, random seed, and checkpoint provenance.
4. Publish only figures and metrics that can be traced to a recorded run and are safe to redistribute.
