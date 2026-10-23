#!/usr/bin/env bash
set -Eeuo pipefail

step="initialization"
trap 'printf "Inference setup failed during %s\n" "$step" >&2' ERR

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

step="checking uv"
if ! command -v uv >/dev/null 2>&1; then
  printf 'Install uv before running this setup script\n' >&2
  exit 1
fi

step="creating the Python 3.10 environment"
if [[ ! -e .venv/bin/python ]]; then
  uv venv --python 3.10 .venv
fi
if ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10)'; then
  printf 'The existing .venv does not use Python 3.10\n' >&2
  exit 1
fi

step="installing the locked inference dependencies"
SAM2_BUILD_CUDA=0 uv pip sync --python .venv/bin/python --torch-backend cu118 requirements-infer.lock

step="installing AnomNOVIC without changing locked dependencies"
uv pip install --python .venv/bin/python --no-deps --editable .

step="checking out NOVIC v1.0.2"
novic_commit="110de7e3aaeb65fc30138b7be53b7dbbdac9ccd5"
if [[ ! -e external/novic ]]; then
  mkdir -p external
  git clone --branch v1.0.2 --depth 1 https://github.com/pallgeuer/novic.git external/novic
fi
if [[ "$(git -C external/novic rev-parse HEAD)" != "$novic_commit" ]]; then
  printf 'external/novic has the wrong HEAD. Expected %s\n' "$novic_commit" >&2
  exit 1
fi
if [[ "$(git -C external/novic rev-parse 'v1.0.2^{commit}')" != "$novic_commit" ]]; then
  printf 'external/novic has the wrong v1.0.2 tag\n' >&2
  exit 1
fi
if [[ -n "$(git -C external/novic status --porcelain)" ]]; then
  printf 'external/novic contains local changes\n' >&2
  exit 1
fi

step="patching the installed timm import"
.venv/bin/python scripts/patch_timm.py

step="preparing and verifying public inference assets"
HF_HOME="$repo_root/assets/huggingface" HF_HUB_OFFLINE=0 .venv/bin/python scripts/prepare_inference_assets.py

step="checking imports and command help"
export PYTHONPATH="$repo_root/external/novic${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$repo_root/assets/huggingface"
export HF_HUB_OFFLINE=1
.venv/bin/python -c 'import timm, anomnovic, infer, open_clip, sam2; print("Inference imports passed")'
.venv/bin/python -c 'import importlib.util; assert importlib.util.find_spec("sam2._C") is None, "SAM2 CUDA extension was unexpectedly installed"'
.venv/bin/anomnovic-infer --help >/dev/null
.venv/bin/anomnovic-evaluate --help >/dev/null

printf 'Inference setup completed. Source scripts/activate_inference.sh before running inference.\n'
