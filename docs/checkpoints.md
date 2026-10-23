# Checkpoints

Download the four checkpoint files using the links below.

| Asset | Use |
| --- | --- |
| [mae_plain.pth](https://github.com/pallgeuer/anomnovic/releases/download/v1.0.0/mae_plain.pth) | MAE anomaly localizer for the plain workspace dataset |
| [mae_distractors.pth](https://github.com/pallgeuer/anomnovic/releases/download/v1.0.0/mae_distractors.pth) | MAE anomaly localizer for the distractor workspace dataset |
| [novic_n_ft0.train](https://github.com/pallgeuer/anomnovic/releases/download/v1.0.0/novic_n_ft0.train) | Custom NICOL-tuned NOVIC N-FT0 decoder |
| [novic_n_ft2.train](https://github.com/pallgeuer/anomnovic/releases/download/v1.0.0/novic_n_ft2.train) | Custom NICOL-tuned NOVIC N-FT2 decoder |

The MAE assets contain the required `model` state dictionary. The NOVIC assets contain the model state, target vocabulary, target configuration, and inference configuration, as required by the public NOVIC loader. Exact asset hashes are in [SHA256SUMS.txt](SHA256SUMS.txt).

The public FT0 and FT2 NOVIC checkpoints are already available as [NOVIC v1.0.0](https://github.com/pallgeuer/novic/releases/tag/v1.0.0) release assets. Their archives are `dfn5bl_ft0_ye2_ovod_20240628_142131.zip` and `dfn5bl_ft2_ya2_ovod_20240620_162925.zip`. Extract the `.train` checkpoint from the selected archive and pass it to `--novic_model`. All four NOVIC variants can be used with either MAE asset.

For the tested inference setup, run `bash scripts/setup_inference.sh` from the repository root. It obtains and verifies both public NOVIC archives and extracts their checkpoints to `assets/checkpoints/novic_ft0.train` and `assets/checkpoints/novic_ft2.train`. It also prepares the exact OpenCLIP and SAM2 files in a dedicated Hugging Face cache. Their source revisions and file hashes are recorded in [THIRD_PARTY_SHA256SUMS.txt](THIRD_PARTY_SHA256SUMS.txt). Run `python scripts/prepare_inference_assets.py --verify-only` after activation to check every prepared file again. The four AnomNOVIC assets above remain covered by [SHA256SUMS.txt](SHA256SUMS.txt).

| Third-party source | Pinned revision |
| --- | --- |
| [NOVIC inference code](https://github.com/pallgeuer/novic) | `v1.0.2` at `110de7e3aaeb65fc30138b7be53b7dbbdac9ccd5` |
| [SAM2 inference code](https://github.com/facebookresearch/sam2) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| [OpenCLIP DFN5B model](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) | `01b771ed0d1395ca5ffdd279897d665ebe00dfd2` |
| [SAM2.1 Hiera Large model](https://huggingface.co/facebook/sam2.1-hiera-large) | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
