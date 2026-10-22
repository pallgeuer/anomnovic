"""Prepare LabelMe ground truth and evaluate AnomNOVIC detections with COCO metrics."""
from __future__ import annotations
import argparse
import collections
import dataclasses
import datetime
import glob
import json
import math
import os
import pycocotools.coco
import pycocotools.cocoeval

def category_info(labelme_path: str, labelme_remap: dict[str, str]) -> tuple[dict[str, int], list[str]]:
	"""
	Scans all LabelMe JSON files in the provided directory and computes a
	mapping from each unique label (found in the "shapes" field) to a unique
	category ID. Labels are sorted alphabetically and IDs start at 1.
	"""

	labels = set()
	for file_path in sorted(glob.glob(os.path.join(labelme_path, "*.json"))):

		with open(file_path, "r") as f:
			data = json.load(f)
		if not isinstance(data, dict) or 'shapes' not in data:
			continue

		for shape in data.get("shapes", []):
			label = shape.get("label", None)
			shape_type = shape.get("shape_type", None)
			if not label or shape_type != 'rectangle':
				raise ValueError(f"Invalid or non-rectangle annotation: {shape}")
			label = labelme_remap.get(label, label)
			labels.add(label)

	sorted_labels = sorted(labels)
	category_map = {label: idx + 1 for idx, label in enumerate(sorted_labels)}
	assert 'other' not in category_map
	category_map['other'] = len(sorted_labels) + 1
	category_list = list(category_map.keys())

	print("CATEGORY MAP:")
	print(json.dumps(category_map, indent=4, ensure_ascii=False))
	print()

	print("CATEGORY LIST:")
	print(json.dumps(category_list, indent=4, ensure_ascii=False))
	print()

	return category_map, category_list

@dataclasses.dataclass(frozen=True)
class BoundingBox:

	index: int
	label: str
	bbox: tuple[float, float, float, float]  # (x, y, w, h)

	def iou(self, OB: BoundingBox) -> float:
		return BoundingBox.compute_iou(self, OB)

	@staticmethod
	def compute_iou(A: BoundingBox, B: BoundingBox) -> float:
		x1, y1, w1, h1 = A.bbox
		x2, y2, w2, h2 = B.bbox
		inter_x1 = max(x1, x2)
		inter_y1 = max(y1, y2)
		inter_x2 = min(x1 + w1, x2 + w2)
		inter_y2 = min(y1 + h1, y2 + h2)
		if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
			return 0.0
		inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
		area1 = w1 * h1
		area2 = w2 * h2
		union_area = area1 + area2 - inter_area
		return inter_area / union_area

	def __repr__(self) -> str:
		return f"BBox({self.index}, {self.label}, ({self.bbox[0]:.0f}, {self.bbox[1]:.0f}), ({self.bbox[2]:.0f}, {self.bbox[3]:.0f}))"

def gen_coco_gt(labelme_path: str, labelme_remap: dict[str, str], coco_gt_path: str, declass: bool = False, iou_thres: float = 0.8):

	if not coco_gt_path:
		raise ValueError("Provide a valid output COCO ground truth annotations file path")

	# First, get the category mapping (label -> category_id)
	category_map, category_list = category_info(labelme_path=labelme_path, labelme_remap=labelme_remap)

	# Compile the categories information
	categories = [dict(id=categ_id, name=categ, supercategory='object') for categ, categ_id in category_map.items()]

	if declass:
		for categ in category_map:
			category_map[categ] = 1
		categories = [dict(id=1, name='object', supercategory='object')]

	# Counters for unique IDs
	image_id_counter = 1
	annotation_id_counter = 1

	# Data
	images = []
	anns = []
	sizes = []
	duplicate_num = 0

	# Process each LabelMe JSON file in sorted order
	for file_path in sorted(glob.glob(os.path.join(labelme_path, "*.json"))):

		with open(file_path, "r") as f:
			data = json.load(f)
		if not isinstance(data, dict) or 'shapes' not in data:
			continue

		# Build the image entry
		image_entry = {
			"id": image_id_counter,
			"width": data["imageWidth"],
			"height": data["imageHeight"],
			"file_name": data["imagePath"],
		}
		images.append(image_entry)

		bboxes = set()
		for i, shape in enumerate(data.get("shapes", [])):

			label = shape.get("label", None)
			shape_type = shape.get("shape_type", None)
			if not label or shape_type != 'rectangle':
				raise ValueError(f"Invalid or non-rectangle annotation: {shape}")
			label = labelme_remap.get(label, label)

			# Get the two points and compute the bounding box.
			# LabelMe typically provides two points [x1, y1] and [x2, y2].
			points: list[list[float]] = shape.get("points")  # Note: LabelMe provides floating point AXIS coordinates (origin is top-left of top-left pixel)
			if not points or len(points) != 2:
				raise ValueError(f"Invalid points: {points}")
			x1, y1 = points[0]
			x2, y2 = points[1]
			x1, x2, y1, y2 = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
			bbox = (x1, y1, x2 - x1, y2 - y1)

			B = BoundingBox(index=i, label=label, bbox=bbox)
			dupl_boxes = {OB for OB in bboxes if B.iou(OB) >= iou_thres}
			bboxes.difference_update(dupl_boxes)
			for OB in dupl_boxes:
				print(f"[{image_entry['file_name']}] TAKING {B} INSTEAD OF {OB}")
			duplicate_num += len(dupl_boxes)
			bboxes.add(B)

		for B in sorted(bboxes, key=lambda b: b.index):
			x1, y1, w, h = B.bbox
			x2, y2 = x1 + w, y1 + h
			area = w * h
			ann = {
				"id": annotation_id_counter,
				"image_id": image_id_counter,
				"category_id": category_map[B.label],
				"bbox": list(B.bbox),
				"area": area,
				"segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
				"iscrowd": 0,
			}
			anns.append(ann)
			sizes.append(math.sqrt(area))
			annotation_id_counter += 1

		image_id_counter += 1

	if duplicate_num > 0:
		print(f"FOUND {duplicate_num} duplicates")

	# Build the final COCO ground truth dictionary for this image
	now = datetime.datetime.now()
	coco_dict = dict(
		info=dict(
			description='Anomalies Dataset',
			version='0.1',
			year=now.year,
			contributor='Knowledge Technology',
			date_created=now.strftime("%Y/%m/%d"),
		),
		categories=categories,
		images=images,
		annotations=anns,  # Note: These are saved as floating point AXIS coordinates (origin is top-left of top-left pixel)
	)

	# Save the new JSON file. The output filename is based on the original filename.
	with open(coco_gt_path, "w") as f:
		json.dump(coco_dict, f, indent=2, ensure_ascii=False)  # noqa

def align_category_name(named_pred_path: str, coco_pred_path: str, coco_gt_path: str, category_names_path: str, min_bbox_size: float, max_bbox_size: float):

	if not named_pred_path:
		raise ValueError("Provide a valid named predictions file path")
	if not coco_gt_path:
		raise ValueError("Provide a valid output COCO ground truth annotations file path")

	with open(coco_gt_path, 'r') as f:
		coco_data = json.load(f)
	categories = {cat['name']: cat['id'] for cat in coco_data['categories']}
	category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
	assert len(categories) == len(category_id_to_name) == len(coco_data['categories'])

	anns = collections.defaultdict(list)
	for ann in coco_data['annotations']:
		bbox = BoundingBox(index=ann['id'], label=category_id_to_name[ann['category_id']], bbox=tuple(ann['bbox']))  # noqa
		anns[ann['image_id']].append(bbox)

	with open(category_names_path, 'r') as f:
		cat_names = json.load(f)

	with open(named_pred_path, 'r') as f:
		pred_data = json.load(f)

	out_data = []
	missing = collections.defaultdict(set)
	toggle = True
	for i, pred in enumerate(pred_data, 1):

		category_name = pred['category_name']

		x, y, w, h = pred['bbox']
		bbox_size = math.sqrt(w * h)
		if not min_bbox_size <= bbox_size <= max_bbox_size:
			continue

		if category_name not in categories:
			B = BoundingBox(index=i, label=category_name, bbox=tuple(pred['bbox']))  # noqa
			G = max(anns[pred['image_id']], key=lambda b: B.iou(b), default=None)
			iou = B.iou(G) if G is not None else 0.0
			if iou < 0.49:
				category_name = 'other'
			else:
				entry = cat_names[G.label]
				if B.label in entry['correct']:
					category_name = G.label
				elif B.label in entry['close']:
					category_name = G.label if toggle else 'other'
					toggle = not toggle
				else:
					category_name = 'other'
					if B.label not in entry['incorrect']:
						missing[G.label].add(B.label)

		out_pred = {(key if key != 'category_name' else 'category_id'): (value if key != 'category_name' else categories[category_name]) for key, value in pred.items()}
		out_data.append(out_pred)

	with open(coco_pred_path, 'w') as f:
		json.dump(out_data, f, indent=2, ensure_ascii=False)  # noqa

	print("MISSING:")
	print(json.dumps({categ: sorted(misses) for categ, misses in sorted(missing.items())}, indent=2, ensure_ascii=False))

def eval_pred(coco_gt_path: str, coco_pred_path: str, class_agnostic: bool):

	# Load ground truth COCO annotations
	coco_gt = pycocotools.coco.COCO(coco_gt_path)

	# Load the predictions (results)
	coco_dt = coco_gt.loadRes(coco_pred_path)

	# Initialize COCOeval object for bounding box evaluation
	coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')  # Note: COCO evaluation assumes both ground truths and predictions are in AXIS coordinates (origin is top-left of top-left pixel)

	coco_eval.params.areaRng = [
		[0 ** 2, 1e5 ** 2],  # all
		[0 ** 2, 180 ** 2],  # small
		[180 ** 2, 300 ** 2],  # medium
		[300 ** 2, 1e5 ** 2]  # large
	]

	if class_agnostic:
		coco_eval.params.useCats = 0
		print("Class agnostic")
	else:
		coco_eval.params.useCats = 1
		print("Using classes as normal")

	# Run evaluation:
	# 1. Evaluate per image and per category
	coco_eval.evaluate()
	# 2. Accumulate evaluation results across images
	coco_eval.accumulate()
	# 3. Summarize and print the performance metrics (AP, AR, etc.)
	coco_eval.summarize()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gt = sub.add_parser('ground-truth', help='Convert LabelMe rectangles to COCO ground truth')
    gt.add_argument('labelme_dir')
    gt.add_argument('output_json')
    gt.add_argument('--declass', action='store_true')
    gt.add_argument('--remap_file', help='JSON object mapping LabelMe labels to evaluation category names')
    gt.add_argument('--remap', action='append', default=[], metavar='OLD=NEW')
    gt.add_argument('--duplicate_iou', type=float, default=0.8)
    align = sub.add_parser('align', help='Map open-vocabulary predictions to COCO category IDs')
    align.add_argument('named_predictions')
    align.add_argument('ground_truth')
    align.add_argument('category_names')
    align.add_argument('output_json')
    align.add_argument('--min_bbox_size', type=float, default=0.0)
    align.add_argument('--max_bbox_size', type=float, default=math.inf)
    ev = sub.add_parser('evaluate', help='Compute COCO bounding box metrics')
    ev.add_argument('ground_truth')
    ev.add_argument('predictions')
    ev.add_argument('--class_agnostic', action='store_true')
    args = parser.parse_args()
    if args.command == 'ground-truth':
        if args.remap_file:
            with open(args.remap_file) as fp:
                remap = json.load(fp)
        else:
            remap = {}
        if not isinstance(remap, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in remap.items()):
            raise ValueError('The label remap file must contain a JSON object of string pairs')
        remap.update(dict(item.split('=', 1) for item in args.remap))
        gen_coco_gt(args.labelme_dir, remap, args.output_json, args.declass, args.duplicate_iou)
    elif args.command == 'align':
        align_category_name(args.named_predictions, args.output_json, args.ground_truth,
                            args.category_names, args.min_bbox_size, args.max_bbox_size)
    elif args.command == 'evaluate':
        eval_pred(args.ground_truth, args.predictions, args.class_agnostic)


if __name__ == '__main__':
    main()
