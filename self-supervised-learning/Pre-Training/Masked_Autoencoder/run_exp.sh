#!/bin/bash

set -euo pipefail

: "${BAC_DATA_DIR:?Set BAC_DATA_DIR to an authorized local image directory}"
OUTPUT_DIR="${BAC_OUTPUT_DIR:-./results/spark}"
RESUME_ARGS=()
if [[ -n "${BAC_RESUME_CHECKPOINT:-}" ]]; then
	RESUME_ARGS=(--resume_from "${BAC_RESUME_CHECKPOINT}")
fi

python ./main.py \
--exp_name=convnextsmall_4 \
--data_path="${BAC_DATA_DIR}" \
--model=convnext_small \
--bs=16 \
--mask=0.6 \
--base_lr=4e-4 \
--exp_dir="${OUTPUT_DIR}" \
--ep=800 \
--input_width=576 \
--input_height=1120 \
--device=cuda:0 \
--dataloader_workers=24 "${RESUME_ARGS[@]}"