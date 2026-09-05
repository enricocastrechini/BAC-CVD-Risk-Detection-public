#!/bin/bash

set -euo pipefail

: "${BAC_DATA_DIR:?Set BAC_DATA_DIR to an authorized local image directory}"
OUTPUT_DIR="${BAC_OUTPUT_DIR:-./results/byol}"

python ./pl_bolts/models/self_supervised/byol/byol_module.py --gpus 1 \
 --data_dir "${BAC_DATA_DIR}" \
 --batch_size 16 \
 --savepath "${OUTPUT_DIR}" \
 --group BYOL \
 --name "${WANDB_RUN_NAME:-byol}" \