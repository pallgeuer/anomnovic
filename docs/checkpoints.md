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
