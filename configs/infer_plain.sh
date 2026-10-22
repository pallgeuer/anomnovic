#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 IMAGE_DIR MAE_CHECKPOINT NOVIC_CHECKPOINT OUTPUT_JSON [INFERENCE_OPTIONS...]" >&2
  exit 2
fi

image_dir="$1"
mae_checkpoint="$2"
novic_checkpoint="$3"
output_json="$4"
shift 4

args=(
  --image_dir "$image_dir"
  --anom_model "$mae_checkpoint"
  --novic_model "$novic_checkpoint"
  --output "$output_json"
  --anom_model_type mae_vit_base_patch16
  --input_size 224
  --mask_ratio 0
  --table_mask
  --anomaly_mask
  --novic_gencfg beam_k10_vnone_gp_t1_a0
  --novic_border_ratio 0.2
  --novic_min_border 2
  --novic_max_border 5
  --novic_score_base 1.15
  --table_mask_avg_n 4
  --table_mask_thres 0.3
  --table_mask_erode_n 0
  --anomaly_mask_thres 0.45
  --anomaly_mask_open_n 0
  --recon_err_norm 1
  --recon_err_thres 1.25
  --recon_denoise_n 0
  --anomaly_area_min 39
  --recon_area_min 24
  --pred_isect_thres 0.25
  --unmatched_score_factor 0.8
  --match_bbox_iou_thres 0.65
  --mean_bbox_iou_thres 0.3
  --sam_enlarge 0
  --sam_area_min 20
  --sam_min_table_overlap 0.4
  --sam_split_penalty 0.75
  --sam_nms_thres 0.85
)

python -m anomnovic.inference "${args[@]}" "$@"
