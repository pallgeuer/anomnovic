# Dataset archives

Download the five AnomNOVIC ZIP archives using the links below.

| Archive | Contents |
| --- | --- |
| [anomnovic_train_plain.zip](https://www2.informatik.uni-hamburg.de/wtm/corpora/anomnovic/anomnovic_train_plain.zip) | 7,498 normal workspace image and table-mask pairs, plus 53 virtual anomaly cutouts |
| [anomnovic_train_distractors.zip](https://www2.informatik.uni-hamburg.de/wtm/corpora/anomnovic/anomnovic_train_distractors.zip) | 7,500 normal workspace image and table-mask pairs with distractors, plus 565 virtual anomaly cutouts |
| [anomnovic_test_plain.zip](https://www2.informatik.uni-hamburg.de/wtm/corpora/anomnovic/anomnovic_test_plain.zip) | 585 camera PNGs and paired LabelMe rectangle annotations |
| [anomnovic_test_distractors.zip](https://www2.informatik.uni-hamburg.de/wtm/corpora/anomnovic/anomnovic_test_distractors.zip) | 560 camera PNGs and paired LabelMe rectangle annotations |
| [anomnovic_test_wild.zip](https://www2.informatik.uni-hamburg.de/wtm/corpora/anomnovic/anomnovic_test_wild.zip) | 38 curated camera PNGs |

## Training layout

Each training archive extracts to a single directory, as shown below.

```text
train_plain/
  scene_0/right_eye/0.jpg
  scene_0/right_eye/mask_0.png
  scene_1/right_eye/...
  scene_2/right_eye/...
  virtual_anomalies/*.png
  LICENSE_YCB.txt
```

The SHA-256 hash for each complete ZIP is in [SHA256SUMS.txt](SHA256SUMS.txt). Matching that hash verifies the contents of the downloaded archive.

The training archives contain camera images and matching table masks. During training, the loader selects a random scale and then takes matching random 224×224 crops from each image and mask. The loader selects scene images only from `scene_*/right_eye/` and pairs each with its table mask. It loads cutouts separately from `virtual_anomalies/` and pastes them into scene images during training.

## Test layout

The two annotated test archives contain 1920×1440 PNG images and same-stem LabelMe JSON pairs in one directory. `anomnovic-evaluate ground-truth` converts the rectangles to COCO format. Pass `--remap_file evaluation/nicol_label_remap.json` to use the NICOL evaluation categories. The Wild archive supplies camera images for qualitative evaluation. The author scored the Wild set manually by inspection of each model's predictions.
