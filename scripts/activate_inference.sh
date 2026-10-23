#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this script from the AnomNOVIC repository root\n' >&2
  exit 1
fi

anomnovic_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$anomnovic_root/.venv/bin/anomnovic-infer" || ! -f "$anomnovic_root/external/novic/infer.py" ]]; then
  printf 'Inference setup is incomplete. Run bash scripts/setup_inference.sh first.\n' >&2
  return 1
fi
anomnovic_clip_ref="$anomnovic_root/assets/huggingface/hub/models--apple--DFN5B-CLIP-ViT-H-14-378/refs/main"
anomnovic_sam_ref="$anomnovic_root/assets/huggingface/hub/models--facebook--sam2.1-hiera-large/refs/main"
if [[ ! -f "$anomnovic_clip_ref" || ! -f "$anomnovic_sam_ref" || "$(< "$anomnovic_clip_ref")" != '01b771ed0d1395ca5ffdd279897d665ebe00dfd2' || "$(< "$anomnovic_sam_ref")" != '665f8e2ad61cf5f53d65644ff27c8ee525124610' ]]; then
  printf 'Verified model cache is incomplete. Run bash scripts/setup_inference.sh first.\n' >&2
  return 1
fi

export PATH="$anomnovic_root/.venv/bin:$PATH"
export PYTHONPATH="$anomnovic_root/external/novic${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$anomnovic_root/assets/huggingface"
export HF_HUB_OFFLINE=1
unset anomnovic_root anomnovic_clip_ref anomnovic_sam_ref
