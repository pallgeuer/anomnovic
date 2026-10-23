"""Draw saved AnomNOVIC predictions on their source images."""
from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BASE_TEXT_AREA = 1920 * 1440


def _image_name(value: object) -> str:
    if not isinstance(value, str) or not value or '/' in value or '\\' in value:
        raise ValueError(f'Expected a filename without directories, got {value!r}')
    return value


def _bbox(value: object) -> tuple[float, float, float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
            or value[2] < 0 or value[3] < 0):
        raise ValueError(f'Expected a finite [x, y, width, height] box, got {value!r}')
    return tuple(value)


def _load_ground_truth(path: Path | None):
    if path is None:
        return {}, {}, {}
    with path.open(encoding='utf-8') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError('Ground truth must be a COCO JSON object')

    image_names = {}
    for image in data.get('images', []):
        image_id = image['id']
        name = _image_name(image['file_name'])
        if image_id in image_names or name in image_names.values():
            raise ValueError(f'Duplicate ground-truth image ID or filename for {name!r}')
        image_names[image_id] = name

    category_names = {}
    for category in data.get('categories', []):
        category_id = category['id']
        if category_id in category_names:
            raise ValueError(f'Duplicate ground-truth category ID {category_id!r}')
        category_names[category_id] = category['name']

    annotations = collections.defaultdict(list)
    for annotation in data.get('annotations', []):
        image_id = annotation['image_id']
        if image_id not in image_names:
            raise ValueError(f'Ground-truth annotation refers to unknown image ID {image_id!r}')
        category_id = annotation['category_id']
        label = category_names.get(category_id, f'class {category_id}')
        annotations[image_names[image_id]].append((_bbox(annotation['bbox']), str(label)))
    return image_names, category_names, annotations


def _prediction_label(prediction: dict, category_names: dict, classes: list[str] | None) -> str:
    name = prediction.get('category_name')
    if name is not None:
        if not isinstance(name, str) or not name:
            raise ValueError(f'Invalid prediction category name {name!r}')
        return name
    category_id = prediction.get('category_id')
    if isinstance(category_id, bool) or not isinstance(category_id, int):
        raise ValueError('Each prediction needs a category_name or an integer category_id')
    if classes is not None and 1 <= category_id <= len(classes):
        return classes[category_id - 1]
    return str(category_names.get(category_id, f'class {category_id}'))


def _draw_box(image, bbox, label: str, color, font_scale: float, text_thickness: int,
              random_labels: bool, rng: random.Random):
    x, y, width, height = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    if random_labels:
        text_y = int(y + 40 + max(height - 40, 0) * rng.random())
    else:
        above = int(y) - 18
        text_y = above if above >= text_height else int(y + height + text_height + baseline - 12)
    text_x = min(max(int(x), 0), max(image.shape[1] - text_width, 0))
    text_y = min(max(text_y, text_height), image.shape[0] - 1)
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, text_thickness)


def visualize_predictions(image_dir: str, predictions_json: str, output_dir: str, *,
                          ground_truth: str | None = None, show_ground_truth: bool = False,
                          classes: list[str] | None = None, num_images: int = 0,
                          seed: int = 0, random_labels: bool = False,
                          overwrite: bool = False) -> int:
    """Render saved predictions and return the number of images written."""
    source_dir = Path(image_dir)
    destination_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise ValueError(f'Image directory does not exist: {source_dir}')
    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError('Output directory must differ from the image directory')
    if destination_dir.exists() and not destination_dir.is_dir():
        raise ValueError(f'Output path is not a directory: {destination_dir}')
    if show_ground_truth and ground_truth is None:
        raise ValueError('--show_ground_truth requires --ground_truth')
    if num_images < 0:
        raise ValueError('--num_images must be nonnegative')
    if classes is not None and (not classes or any(not name for name in classes)):
        raise ValueError('--classes requires one or more nonempty names')

    image_paths = {path.name: path for path in sorted(source_dir.iterdir())
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if not image_paths:
        raise ValueError(f'No supported images found in {source_dir}')

    image_names, category_names, annotations = _load_ground_truth(
        Path(ground_truth) if ground_truth else None)
    with Path(predictions_json).open(encoding='utf-8') as fp:
        predictions = json.load(fp)
    if not isinstance(predictions, list):
        raise ValueError('Predictions must be a JSON list')

    by_image = collections.defaultdict(list)
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f'Prediction {index} must be a JSON object')
        if 'image_name' in prediction:
            name = _image_name(prediction['image_name'])
            if 'image_id' in prediction and image_names.get(prediction['image_id']) != name:
                raise ValueError(f'Prediction {index} has mismatched image ID and filename')
        elif 'image_id' in prediction:
            image_id = prediction['image_id']
            if image_id not in image_names:
                raise ValueError(f'Prediction {index} has image ID {image_id!r} without a ground-truth mapping')
            name = image_names[image_id]
        else:
            raise ValueError(f'Prediction {index} has no image_name or image_id')
        if name not in image_paths:
            raise ValueError(f'Prediction {index} refers to an image missing from {source_dir}: {name}')
        score = prediction.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f'Prediction {index} needs a finite score')
        label = _prediction_label(prediction, category_names, classes)
        by_image[name].append((_bbox(prediction.get('bbox')), f'{label}: {score:.2f}'))

    selected_names = sorted(image_paths)
    if num_images and num_images < len(selected_names):
        selected_names = sorted(random.Random(seed).sample(selected_names, num_images))
    if show_ground_truth:
        missing = [name for name in selected_names if name not in image_names.values()]
        if missing:
            raise ValueError(f'Ground truth has no entry for image {missing[0]!r}')
    conflicts = [name for name in selected_names if (destination_dir / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f'Output image already exists: {destination_dir / conflicts[0]}. Pass --overwrite to replace it')

    destination_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for name in selected_names:
        image = cv2.imread(str(image_paths[name]))
        if image is None:
            raise ValueError(f'Could not read image: {image_paths[name]}')
        factor = math.sqrt(max(image.shape[0] * image.shape[1], 1) / BASE_TEXT_AREA)
        font_scale = 1.5 * factor
        text_thickness = max(1, round(2 * factor))
        if show_ground_truth:
            for bbox, label in annotations.get(name, []):
                _draw_box(image, bbox, label, (0, 0, 255), font_scale, text_thickness, False, rng)
        prediction_color = (255, 0, 0) if show_ground_truth else (0, 0, 255)
        for bbox, label in by_image.get(name, []):
            _draw_box(image, bbox, label, prediction_color, font_scale, text_thickness,
                      random_labels, rng)
        output_path = destination_dir / name
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f'Could not write image: {output_path}')
    return len(selected_names)
