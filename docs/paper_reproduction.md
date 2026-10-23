# Paper inference and evaluation

The [paper](https://arxiv.org/abs/2606.26829) evaluates the plain and distractor datasets with prompt-free and prompted NOVIC classification, with and without SAM 2.1 refinement. The matching MAE settings are in [`configs/infer_plain.sh`](../configs/infer_plain.sh) and [`configs/infer_distractors.sh`](../configs/infer_distractors.sh). These scripts accept an image directory, MAE checkpoint, NOVIC checkpoint, output JSON, and optional `anomnovic-infer` arguments in that order.

Download the [datasets](datasets.md) and [checkpoints](checkpoints.md) first. Install NOVIC v1.0.2 and optional SAM2 as described in the [README](../README.md). Use the supplied plain MAE for the plain test set and the supplied distractor MAE for the distractor test set. The scripts keep the checkpoint-specific localization settings while the optional arguments select the classification and refinement modes.

## Ground truth and candidate classes

Run these commands from the repository root after changing the paths to your local files.

```bash
PLAIN_IMAGES=/PATH/TO/TEST_PLAIN_DIR
DISTRACTOR_IMAGES=/PATH/TO/TEST_DISTRACTORS_DIR
PLAIN_MAE=/PATH/TO/mae_plain.pth
DISTRACTOR_MAE=/PATH/TO/mae_distractors.pth
NOVIC_CHECKPOINT=/PATH/TO/novic_n_ft2.train
RESULTS=/PATH/TO/RESULTS
mkdir -p "$RESULTS"
anomnovic-evaluate ground-truth "$PLAIN_IMAGES" "$RESULTS/gt_plain.json" --remap_file evaluation/nicol_label_remap.json
anomnovic-evaluate ground-truth "$DISTRACTOR_IMAGES" "$RESULTS/gt_distractors.json" --remap_file evaluation/nicol_label_remap.json
```

The examples below use the custom N-FT2 NOVIC checkpoint. The paper's prompted runs with custom N-FT0 and N-FT2 checkpoints used the following candidate nouns. The candidate `minicar` occupied the same category position as `minibus` in the ground truth.

```bash
CUSTOM_CLASSES=(apricot banana baseball beachball bowl "cereal box" cube cup "granny smith" "green pepper" jello lemon minicar orange pear tin tomato)
```

For the public FT0 and FT2 NOVIC checkpoints, omit `--classes "${CUSTOM_CLASSES[@]}"` from the prompted commands below. Those paper runs used the candidate names from the generated ground truth, including `minibus`. The prompt-free commands do not use a candidate list. To compare all four NOVIC variants, repeat the commands with each `.train` checkpoint described in [checkpoints.md](checkpoints.md) and set a separate `RESULTS` directory for each checkpoint.

## Run the inference variants

Each pair below runs without SAM first and then with SAM. The same `--coco_gt_path` supplies image IDs in both classification modes. Prompted runs also use its category IDs for COCO evaluation.

```bash
bash configs/infer_plain.sh "$PLAIN_IMAGES" "$PLAIN_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/plain_free.json" --coco_gt_path "$RESULTS/gt_plain.json"
bash configs/infer_plain.sh "$PLAIN_IMAGES" "$PLAIN_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/plain_free_sam.json" --coco_gt_path "$RESULTS/gt_plain.json" --sam_model facebook/sam2.1-hiera-large
bash configs/infer_plain.sh "$PLAIN_IMAGES" "$PLAIN_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/plain_prompted.json" --coco_gt_path "$RESULTS/gt_plain.json" --fixed_classes --classes "${CUSTOM_CLASSES[@]}"
bash configs/infer_plain.sh "$PLAIN_IMAGES" "$PLAIN_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/plain_prompted_sam.json" --coco_gt_path "$RESULTS/gt_plain.json" --fixed_classes --classes "${CUSTOM_CLASSES[@]}" --sam_model facebook/sam2.1-hiera-large
bash configs/infer_distractors.sh "$DISTRACTOR_IMAGES" "$DISTRACTOR_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/distractors_free.json" --coco_gt_path "$RESULTS/gt_distractors.json"
bash configs/infer_distractors.sh "$DISTRACTOR_IMAGES" "$DISTRACTOR_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/distractors_free_sam.json" --coco_gt_path "$RESULTS/gt_distractors.json" --sam_model facebook/sam2.1-hiera-large
bash configs/infer_distractors.sh "$DISTRACTOR_IMAGES" "$DISTRACTOR_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/distractors_prompted.json" --coco_gt_path "$RESULTS/gt_distractors.json" --fixed_classes --classes "${CUSTOM_CLASSES[@]}"
bash configs/infer_distractors.sh "$DISTRACTOR_IMAGES" "$DISTRACTOR_MAE" "$NOVIC_CHECKPOINT" "$RESULTS/distractors_prompted_sam.json" --coco_gt_path "$RESULTS/gt_distractors.json" --fixed_classes --classes "${CUSTOM_CLASSES[@]}" --sam_model facebook/sam2.1-hiera-large
```

The plain script uses the B1999 settings at a 224-pixel MAE input size. The area thresholds apply at MAE resolution, and the code scales NOVIC crop borders to each input image. The distractor script uses the OB1999 threshold values from the paper experiments. Both scripts allow additional inference flags after the four positional paths.

## Save prediction images

Render an inference JSON after it has been written. Prediction files that contain `image_id` need the corresponding COCO ground-truth JSON to map IDs to image filenames. The first example uses free-form names from NOVIC. The second uses the prompted class list above so the image displays the actual candidate `minicar` rather than the ground-truth category `minibus`. Its optional ground-truth overlay draws annotations in red and predictions in blue.

```bash
anomnovic-evaluate visualize "$PLAIN_IMAGES" "$RESULTS/plain_free.json" "$RESULTS/plain_free_images" --ground_truth "$RESULTS/gt_plain.json"
anomnovic-evaluate visualize "$PLAIN_IMAGES" "$RESULTS/plain_prompted.json" "$RESULTS/plain_prompted_images" --ground_truth "$RESULTS/gt_plain.json" --classes "${CUSTOM_CLASSES[@]}" --show_ground_truth
anomnovic-evaluate visualize /PATH/TO/WILD_IMAGES /PATH/TO/WILD_PREDICTIONS.json /PATH/TO/WILD_VISUALIZATIONS
```

Without a ground-truth overlay, prediction boxes are red. The command saves every input image, including images with no detections, using its original filename and format. Labels appear above boxes by default. Add `--random_labels` to vary their vertical positions and make overlapping predictions easier to see. Use `--seed 7` to repeat those placements or a `--num_images 50` random sample. Existing output images are protected unless `--overwrite` is supplied. For prompted predictions without ground truth, pass the inference candidate names in ID order with `--classes`. If only an ID is available, its label appears as `class 1`, `class 2`, and so on.

## Evaluate the predictions

Prompt-free predictions contain generated category names. Align those names with the NICOL evaluation categories before COCO evaluation. The [alignment file](../evaluation/nicol_category_alignment.json) captures the category judgments available for the paper experiments. Review any new names printed as missing by `align` before interpreting a prompt-free AP value.

```bash
anomnovic-evaluate align "$RESULTS/plain_free.json" "$RESULTS/gt_plain.json" evaluation/nicol_category_alignment.json "$RESULTS/plain_free_coco.json"
anomnovic-evaluate align "$RESULTS/plain_free_sam.json" "$RESULTS/gt_plain.json" evaluation/nicol_category_alignment.json "$RESULTS/plain_free_sam_coco.json"
anomnovic-evaluate align "$RESULTS/distractors_free.json" "$RESULTS/gt_distractors.json" evaluation/nicol_category_alignment.json "$RESULTS/distractors_free_coco.json"
anomnovic-evaluate align "$RESULTS/distractors_free_sam.json" "$RESULTS/gt_distractors.json" evaluation/nicol_category_alignment.json "$RESULTS/distractors_free_sam_coco.json"
anomnovic-evaluate evaluate "$RESULTS/gt_plain.json" "$RESULTS/plain_free_coco.json"
anomnovic-evaluate evaluate "$RESULTS/gt_plain.json" "$RESULTS/plain_free_sam_coco.json"
anomnovic-evaluate evaluate "$RESULTS/gt_plain.json" "$RESULTS/plain_prompted.json"
anomnovic-evaluate evaluate "$RESULTS/gt_plain.json" "$RESULTS/plain_prompted_sam.json"
anomnovic-evaluate evaluate "$RESULTS/gt_distractors.json" "$RESULTS/distractors_free_coco.json"
anomnovic-evaluate evaluate "$RESULTS/gt_distractors.json" "$RESULTS/distractors_free_sam_coco.json"
anomnovic-evaluate evaluate "$RESULTS/gt_distractors.json" "$RESULTS/distractors_prompted.json"
anomnovic-evaluate evaluate "$RESULTS/gt_distractors.json" "$RESULTS/distractors_prompted_sam.json"
```

For class-agnostic localization, add `--declass --fixed_classes` to either inference script and evaluate its JSON with `anomnovic-evaluate evaluate /PATH/TO/GT.json /PATH/TO/PREDICTIONS.json --class_agnostic`. The NOVIC checkpoint positional argument is ignored by inference in this mode.

The Wild dataset has no automatic scoring file. Its accuracy in the paper was scored manually by inspecting predictions.
