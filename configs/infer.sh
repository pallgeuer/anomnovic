#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 IMAGE_DIR MAE_CHECKPOINT NOVIC_CHECKPOINT OUTPUT_JSON" >&2
  exit 2
fi
anomnovic-infer --image_dir "$1" --anom_model "$2" --novic_model "$3" --output "$4" --anom_model_type mae_vit_base_patch16 --input_size 224 --table_mask --anomaly_mask
