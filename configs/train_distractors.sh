#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 TRAIN_DATA_DIR VAL_DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi
data_dir="$1"
val_dir="$2"
output_dir="$3"
python -m anomnovic.mae.train --data_path "$data_dir" --val_data "$val_dir" --virtual_anomalies "$data_dir/virtual_anomalies" --output_dir "$output_dir" --log_dir "$output_dir" --model mae_vit_base_patch16 --input_size 224 --batch_size 192 --epochs 2000 --lr 0.0006 --num_workers 10 --mask_ratio 0.3 --drop_mask_zero 0.8 --color_jitter 0.6 --max_resize_zoom 448 --min_nbr_anomalies 2 --max_nbr_anomalies 25 --rotation_angle 30 --min_anomaly_size 0.15 --max_anomaly_size 0.5 --dynamic_anomalies --anomaly_shadows --anomaly_shadow_prob 0.7 --anomaly_shadow_dark_max 0.25 --anomaly_shadow_offset_std_factor 0.15 --anomaly_mask --anomaly_reconstruction_weight 5 --anomaly_mask_weight 10 --nonanomaly_mask_weight 1 --table_reconstruction_weight 10 --table_mask_weight 2
