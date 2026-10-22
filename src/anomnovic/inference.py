#!/usr/bin/env python3

from __future__ import annotations
import os
import json
import math
import time
import random
import logging
import pathlib
import argparse
import operator
import warnings
import importlib
import itertools
import dataclasses
import contextlib
from typing import Iterable, Optional, Sequence, Union, NamedTuple
import PIL
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mae.anomaly_detection import AnomalyDetector

warnings.simplefilter('ignore', FutureWarning)

try:
	from infer import NOVICModel
except ImportError:
	NOVICModel = None

try:
	from sam2.sam2_image_predictor import SAM2ImagePredictor
	sam2_dir = pathlib.Path(importlib.import_module("sam2").__file__).parent
	class DropSam2(logging.Filter):
		def filter(self, record: logging.LogRecord) -> bool:
			return not pathlib.Path(record.pathname).is_relative_to(sam2_dir)
	logging.getLogger().addFilter(DropSam2())
except ImportError:
	SAM2ImagePredictor = None

@dataclasses.dataclass(frozen=True)
class Anomaly:
	bbox: tuple[float, ...]  # (x, y, w, h) bounding box in floating point AXIS coordinates
	score: float

	def __repr__(self):
		return f"{self.__class__.__name__}(offset=({self.bbox[0]:.1f}, {self.bbox[1]:.1f}), size={self.bbox[2]:.1f}x{self.bbox[3]:.1f}, score={self.score:.2f})"

	def iou(self, other: Anomaly) -> float:
		self_x, self_y, self_w, self_h = self.bbox
		other_x, other_y, other_w, other_h = other.bbox
		isect_x1 = max(self_x, other_x)
		isect_y1 = max(self_y, other_y)
		isect_x2 = min(self_x + self_w, other_x + other_w)
		isect_y2 = min(self_y + self_h, other_y + other_h)
		inter_w = max(0.0, isect_x2 - isect_x1)
		inter_h = max(0.0, isect_y2 - isect_y1)
		inter = inter_w * inter_h
		if inter == 0:
			return 0.0
		else:
			union = self_w * self_h + other_w * other_h - inter
			return inter / union

@dataclasses.dataclass(frozen=True)
class LabeledAnomaly:

	anomaly: Anomaly
	category_name: str
	category_id: int  # <0 = Invalid category ID (just trust category_name instead), i implies cfg.classes[i], zero-indexed
	category_log_score: float
	category_score: float

	def __repr__(self):
		return f"{self.__class__.__name__}(offset=({self.anomaly.bbox[0]:.1f}, {self.anomaly.bbox[1]:.1f}), size={self.anomaly.bbox[2]:.1f}x{self.anomaly.bbox[3]:.1f}, anom_score={self.anomaly.score:.2f}, categ={self.category_name!r}, categ_id={self.category_id}, categ_score={self.category_score:.2f}/{self.category_log_score:.2f})"

@dataclasses.dataclass(frozen=True, eq=False)
class PredContour:

	# Note: All points, widths, heights, coordinates AND bounding box are in integral PIXEL coordinates (NOT floating point axis coordinates)
	points: np.ndarray               # N x 1 x 2 array of (x, y) points
	w: int                           # Contour width (W)
	h: int                           # Contour height (H)
	x1: int                          # Left edge index in full image
	x2: int                          # One past the right edge index in full image (for slicing)
	y1: int                          # Top edge index in full image
	y2: int                          # One past the bottom edge index in full image (for slicing)
	bbox: tuple[int, int, int, int]  # (x1, y1, w, h) bounding box => Integral PIXEL coordinates, where width/height is how many pixels wide/high the contour mask is
	bbox_area: int                   # Bounding box area (pixel count)
	mask: np.ndarray                 # Contour binary mask (H x W of type np.uint8)
	area: int                        # Contour area (geometric area = pixel count)
	score: float                     # Contour score (mean score over lower-scoring half of contour area)

	def __repr__(self):
		return f"{self.__class__.__name__}(offset=({self.x1}, {self.y1}), size={self.w}x{self.h}, area={self.area}, score={self.score:.2f})"

	@classmethod
	def from_points(cls, points: np.ndarray, scores: Union[np.ndarray, float]):
		x1, y1, w, h = bbox = cv2.boundingRect(points)
		x2 = x1 + w
		y2 = y1 + h
		bbox_area = w * h
		mask = cv2.drawContours(np.zeros((h, w), dtype=np.uint8), contours=(points,), contourIdx=-1, color=1, thickness=cv2.FILLED, offset=(-x1, -y1))
		area = int(mask.sum())
		assert area >= 1
		if isinstance(scores, float):
			score = scores
		else:
			area_scores = scores[y1:y2, x1:x2][mask.astype(bool)]
			num_worst_scores = (area_scores.size + 1) // 2
			area_scores.partition(kth=num_worst_scores - 1)
			score = float(area_scores[:num_worst_scores].mean())
		return cls(points=points, w=w, h=h, x1=x1, x2=x2, y1=y1, y2=y2, bbox=bbox, bbox_area=bbox_area, mask=mask, area=area, score=score)

	def isect_area(self, other: PredContour) -> tuple[int, int]:
		# Returns the contour mask intersection area and the bounding box intersection area
		isect_x1 = max(self.x1, other.x1)
		isect_x2 = min(self.x2, other.x2)
		isect_y1 = max(self.y1, other.y1)
		isect_y2 = min(self.y2, other.y2)
		if isect_x2 <= isect_x1 or isect_y2 <= isect_y1:
			return 0, 0
		else:
			isect_self = self.mask[(isect_y1 - self.y1):(isect_y2 - self.y1), (isect_x1 - self.x1):(isect_x2 - self.x1)]
			isect_other = other.mask[(isect_y1 - other.y1):(isect_y2 - other.y1), (isect_x1 - other.x1):(isect_x2 - other.x1)]
			return int(np.logical_and(isect_self, isect_other).sum()), isect_self.size

	def isect_union_area(self, other: PredContour) -> tuple[int, int, int, int]:
		isect_area, isect_bbox_area = self.isect_area(other)
		union_area = self.area + other.area - isect_area
		union_bbox_area = self.bbox_area + other.bbox_area - isect_bbox_area
		return isect_area, union_area, isect_bbox_area, union_bbox_area

class ContourIsect(NamedTuple):
	anom_ratio: float   # isect / anom_area
	recon_ratio: float  # isect / recon_area
	iou: float          # isect / union
	bbox_iou: float     # bbox isect / union

def load_images_from_dir(image_dir):
	supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
	return sorted(str(p) for p in pathlib.Path(image_dir).glob("*") if p.suffix.lower() in supported_exts)

def unnormalise_image_to_bgr(img: torch.Tensor):
	img = img * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]
	img = img.permute(1, 2, 0).numpy()
	img = (img * 255.0).clip(0, 255).astype(np.uint8)
	return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def show_normalised_image(img: torch.Tensor, title: str):
	cv2.imshow(title, unnormalise_image_to_bgr(img))

def show_float_mask(mask: torch.Tensor, title: str, min_val: float, max_val: float):
	mask = mask.numpy()
	mask = ((mask - min_val) * (255.0 / (max_val - min_val))).clip(0, 255)
	mask = mask.astype(np.uint8)
	cv2.imshow(title, mask)

def show_binary_mask(mask: torch.Tensor, title: str):
	mask = mask.numpy()
	cv2.imshow(title, mask.astype(np.uint8) * 255)

def load_anom_model(cfg: argparse.Namespace, device: torch.device):
	anom_model = AnomalyDetector(
		weights_path=cfg.anom_model,
		device=device,
		model_type=cfg.anom_model_type,
		mask_ratio=cfg.mask_ratio,
		no_mask=not cfg.table_mask,
		anomaly_mask=cfg.anomaly_mask,
		input_size=cfg.input_size,
	)
	anom_tfrm = torchvision.transforms.Compose([
		torchvision.transforms.Resize((cfg.input_size, cfg.input_size), interpolation=PIL.Image.BICUBIC),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
	])
	return anom_model, anom_tfrm

def avg_float_mask(mask: torch.Tensor, n: int, pad_value: float):
	return F.avg_pool2d(mask - pad_value, kernel_size=(2 * n + 1), stride=1, padding=n, count_include_pad=True) + pad_value  # Input must be CxHxW or BxCxHxW

def circ_kernel(n: int):
	u = torch.arange(-n, n + 1).square_()
	return (u[None, None, None, :] + u[None, None, :, None] <= (n + 0.2) ** 2).float()

def circ_avg_float_mask(mask: torch.Tensor, n: int, pad_value: float):
	mask = F.pad(mask, pad=(n, n, n, n), mode='constant', value=pad_value)
	weight = circ_kernel(n=n)
	weight.div_(weight.sum())
	return F.conv2d(mask, weight=weight, bias=None, stride=1, padding='valid')

def circ_erode_bool_mask(mask: torch.Tensor, n: int):
	mask = (~mask).float()
	weight = circ_kernel(n=n)
	return (F.conv2d(mask, weight=weight, bias=None, stride=1, padding='same') == 0)

def circ_open_mask(mask: torch.Tensor, n: int):
	weight = circ_kernel(n=n)
	mask = (~mask).float()
	torch.logical_not(F.conv2d(mask, weight=weight, bias=None, stride=1, padding='same'), out=mask)
	return (F.conv2d(mask, weight=weight, bias=None, stride=1, padding='same') != 0)

def denoise_bool_mask(mask: torch.Tensor, n: int, unset_thres: float, set_thres: float):
	# Unsets any pixels for which within a kernel of size 2n+1 centred around it <=unset_thres (ratio between 0 and 1) of the pixels are set
	# Sets any pixels for which within a kernel of size 2n+1 centred around it >=set_thres (ratio between 0 and 1) of the pixels are set
	s = 2 * n + 1
	ss = s * s
	denoise = F.avg_pool2d(mask.float(), kernel_size=s, stride=1, padding=n, count_include_pad=True, divisor_override=1)
	return (denoise > unset_thres * ss).logical_and_(mask).logical_or_(denoise >= set_thres * ss)

def extract_mask_contours(mask: torch.Tensor, scores: torch.Tensor, min_area: int, show: bool, img: torch.Tensor, title: str):

	mask = mask.squeeze(dim=(0, 1)).numpy().astype(np.uint8)
	scores = scores.squeeze(dim=(0, 1)).numpy()

	contours = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]

	pred_contours = []
	for contour in contours:
		pred_contour = PredContour.from_points(points=contour, scores=scores)
		if pred_contour.area >= min_area:
			pred_contours.append(pred_contour)

	if show:
		vis = unnormalise_image_to_bgr(img.squeeze(dim=0))
		assert vis.shape == (mask.shape[0], mask.shape[1], 3) and vis.dtype == np.uint8
		for pred_contour in pred_contours:
			color = [random.randint(200, 255), random.randint(0, 255), random.randint(0, 128)]
			random.shuffle(color)
			cv2.drawContours(vis, (pred_contour.points,), contourIdx=-1, color=color, thickness=1)
		large_vis = cv2.resize(vis, dsize=None, dst=None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
		for pred_contour in pred_contours:
			cv2.putText(large_vis, format(pred_contour.score, '.2f'), (pred_contour.x1 * 4, max(pred_contour.y1 * 4 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(80, 80, 180), thickness=2)
		cv2.imshow(title, vis)
		cv2.imshow(title + ' (Zoom)', large_vis)

	return pred_contours

def _find_contour(parent_contour: dict[PredContour, PredContour], c: PredContour) -> PredContour:
	while parent_contour[c] is not c:
		parent_contour[c] = parent_contour[parent_contour[c]]
		c = parent_contour[c]
	return c

def merge_contours(
	anom_contours: Sequence[PredContour],
	recon_contours: Sequence[PredContour],
	pred_isect_thres: float,
	unmatched_score_factor: float,
	match_bbox_iou_thres: float,
	mean_bbox_iou_thres: float,
):

	ratios_map: dict[tuple[PredContour, PredContour], ContourIsect] = {}   # Format: Maps (anom contour, recon contour) to ContourIsect
	parent_contour: dict[PredContour, PredContour] = {c: c for c in itertools.chain(anom_contours, recon_contours)}

	assert pred_isect_thres > 0
	for anom_contour in anom_contours:
		for recon_contour in recon_contours:
			isect_area, union_area, isect_bbox_area, union_bbox_area = anom_contour.isect_union_area(recon_contour)
			if isect_area <= 0 and isect_bbox_area <= 0:
				continue
			ratios = ContourIsect(
				anom_ratio=isect_area / anom_contour.area,
				recon_ratio=isect_area / recon_contour.area,
				iou=isect_area / union_area,
				bbox_iou=isect_bbox_area / union_bbox_area,
			)
			ratios_map[(anom_contour, recon_contour)] = ratios
			if ratios.anom_ratio >= pred_isect_thres or ratios.recon_ratio >= pred_isect_thres:
				parent_contour[_find_contour(parent_contour, anom_contour)] = _find_contour(parent_contour, recon_contour)

	root_clusters: dict[PredContour, tuple[set[PredContour], set[PredContour]]] = {}  # Format: Maps root contour to ({anom contours}, {recon contours})
	for contour in parent_contour:
		root = _find_contour(parent_contour, contour)
		if root not in root_clusters:
			root_clusters[root] = (set(), set())
		cluster = root_clusters[root]
		if contour in anom_contours:
			cluster[0].add(contour)
		else:
			cluster[1].add(contour)
	clusters = list(root_clusters.values())

	merged_contours = []
	for anom_set, recon_set in clusters:
		if not anom_set:
			for recon_contour in recon_set:
				merged_contours.append(dataclasses.replace(recon_contour, score=recon_contour.score * unmatched_score_factor))
		elif not recon_set:
			for anom_contour in anom_set:
				merged_contours.append(dataclasses.replace(anom_contour, score=anom_contour.score * unmatched_score_factor))
		elif len(anom_set) == len(recon_set) == 1:
			anom_contour = next(iter(anom_set))
			recon_contour = next(iter(recon_set))
			ratios = ratios_map[(anom_contour, recon_contour)]
			take_both = (ratios.bbox_iou <= match_bbox_iou_thres)
			score_factor = max(ratios.iou, ratios.bbox_iou, unmatched_score_factor)
			if anom_contour.score >= recon_contour.score or take_both:
				merged_contours.append(dataclasses.replace(anom_contour, score=anom_contour.score * score_factor))
			if recon_contour.score > anom_contour.score or take_both:
				merged_contours.append(dataclasses.replace(recon_contour, score=recon_contour.score * score_factor))
		else:
			for query_set, value_set, swapped in ((anom_set, recon_set, False), (recon_set, anom_set, True)):
				for query_contour in query_set:
					match_ratios = set()
					for value_contour in value_set:
						ratios = ratios_map.get((value_contour, query_contour) if swapped else (query_contour, value_contour), None)
						if ratios is not None and (ratios.anom_ratio >= pred_isect_thres or ratios.recon_ratio >= pred_isect_thres):
							match_ratios.add(ratios)
					assert match_ratios
					if len(match_ratios) == 1 or sum(ratios.bbox_iou for ratios in match_ratios) / len(match_ratios) >= mean_bbox_iou_thres:
						merged_contours.append(dataclasses.replace(query_contour, score=query_contour.score * max(max(ratios.iou for ratios in match_ratios), max(ratios.bbox_iou for ratios in match_ratios), unmatched_score_factor)))

	merged_contours.sort(key=operator.attrgetter('score'), reverse=True)
	return merged_contours

def show_anomalies(anomalies: Iterable[Anomaly], img: Union[PIL.Image.Image, np.ndarray], title: str):
	if isinstance(img, np.ndarray):
		vis = img.copy()
	else:
		vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
	for anomaly in anomalies:
		x, y, w, h = anomaly.bbox
		start_point = (int(x), int(y))
		end_point = (int(x + w), int(y + h))
		cv2.rectangle(vis, start_point, end_point, color=(0, 0, 255), thickness=2)  # Note: The infinitesimal floating point axis coordinates bounding box cuts through the centre of the drawn lines regardless of thickness (rectangle start/end points are inclusive pixel coordinates (origin is centre of top-left pixel))
		text_position = (int(x), int(max(y - 18, 0)))
		cv2.putText(vis, format(anomaly.score, '.2f'), text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=2)
	factor = 0.25 if max(vis.shape) >= 2000 else 0.5
	vis = cv2.resize(vis, dsize=None, dst=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
	cv2.imshow(title, vis)

def anomaly_nms(anomalies: Sequence[Anomaly], iou_thresh: float) -> list[Anomaly]:

	if not anomalies:
		return []

	order: list[Anomaly] = sorted(anomalies, key=operator.attrgetter('score'))
	keep: list[Anomaly] = []

	while order:
		current = order.pop()
		keep.append(current)
		order = [a for a in order if current.iou(a) <= iou_thresh]

	return keep

def apply_anomaly_sam(anomalies: Sequence[Anomaly], sam_model: SAM2ImagePredictor, img: PIL.Image.Image, enlarge_x: float, enlarge_y: float, min_area: float, split_penalty: float, nms_thres: float, table_mask_np: np.ndarray, min_table_overlap: float, show: bool) -> list[Anomaly]:
	# Note: Table mask is passed at anomaly model resolution

	if not anomalies:
		return []

	factor = 4 if max(img.width, img.height) >= 2000 else 2
	img_small = img.reduce(factor=(factor, factor))
	assert img_small.width == img.width / factor and img_small.height == img.height / factor, f"Image dimensions {img.width}x{img.height} must be divisible by {factor}"
	min_area_small = round(min_area / (factor * factor))

	with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
		sam_model.set_image(img_small)
		masks, mask_qualities, _ = sam_model.predict(box=[(
			max(anomaly.bbox[0] - enlarge_x, 0.0) / factor,
			max(anomaly.bbox[1] - enlarge_y, 0.0) / factor,
			min(anomaly.bbox[0] + anomaly.bbox[2] + enlarge_x, img.width) / factor,
			min(anomaly.bbox[1] + anomaly.bbox[3] + enlarge_y, img.height) / factor,
		) for anomaly in anomalies], multimask_output=False)
	assert len(anomalies) == len(masks) == len(mask_qualities)

	sam_table_mask = cv2.resize(table_mask_np, dsize=(img_small.width, img_small.height), interpolation=cv2.INTER_NEAREST)
	if show:
		cv2.imshow("SAM Table Mask", sam_table_mask * 255)

	sam_contours = []
	sam_anomalies = []
	for anomaly, mask, quality in zip(anomalies, masks, mask_qualities):
		if mask.ndim == 3 and mask.shape[0] == 1:
			mask_img = mask[0]
		elif mask.ndim == 2:
			mask_img = mask
		else:
			raise ValueError(f"Unexpected SAM mask shape {mask.shape}, expected (1,H,W) or (H,W)")
		contours = cv2.findContours(image=(mask_img > 0).astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]

		mask_sam_contours = []
		score = anomaly.score * quality.item()
		for contour in contours:
			sam_contour = PredContour.from_points(points=contour, scores=score)
			if sam_contour.area >= min_area_small:
				mask_sam_contours.append(sam_contour)

		for sam_contour in mask_sam_contours:
			table_overlap = float(np.logical_and(sam_contour.mask, sam_table_mask[sam_contour.y1:sam_contour.y2, sam_contour.x1:sam_contour.x2]).sum()) / sam_contour.area
			if table_overlap >= min_table_overlap:
				if len(mask_sam_contours) > 1:
					sam_contour = dataclasses.replace(sam_contour, score=sam_contour.score * split_penalty)
				sam_contours.append(sam_contour)
				# Note: Here we convert the PIXEL coordinates contour to an AXIS coordinates anomaly bounding box, and scale it back to the original image resolution
				sam_anomalies.append(Anomaly(bbox=(sam_contour.x1 * factor, sam_contour.y1 * factor, sam_contour.w * factor, sam_contour.h * factor), score=sam_contour.score))

	sam_anomalies = anomaly_nms(sam_anomalies, iou_thresh=nms_thres)

	if show:
		img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
		overlay = img_bgr.copy()
		for sam_contour in sam_contours:
			color = [random.randint(200, 255), random.randint(0, 255), random.randint(0, 128)]
			random.shuffle(color)
			sam_contour_mask = cv2.resize(sam_contour.mask, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
			overlay[(sam_contour.y1 * factor):(sam_contour.y2 * factor), (sam_contour.x1 * factor):(sam_contour.x2 * factor)][sam_contour_mask.astype(bool)] = color
		alpha = 0.35
		img_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
		show_anomalies(sam_anomalies, img=img_bgr, title='SAM-adjusted Anomalies')

	return sam_anomalies

def infer_anom_model(
	img: PIL.Image.Image,
	anom_model: AnomalyDetector,
	anom_tfrm: torchvision.transforms.Compose,
	sam_model: Optional[SAM2ImagePredictor],
	device: torch.device,
	show: bool,
	timed: bool,
	table_mask_avg_n: int,
	table_mask_thres: float,
	table_mask_erode_n: int,
	anomaly_mask_thres: float,
	anomaly_mask_open_n: int,
	recon_err_norm: Union[int, float],
	recon_err_thres: float,
	recon_denoise_n: int,
	recon_denoise_unset_thres: float,
	recon_denoise_set_thres: float,
	anomaly_area_min: int,
	recon_area_min: int,
	pred_isect_thres: float,
	unmatched_score_factor: float,
	match_bbox_iou_thres: float,
	mean_bbox_iou_thres: float,
	sam_enlarge: float,
	sam_area_min: int,
	sam_min_table_overlap: float,
	sam_split_penalty: float,
	sam_nms_thres: float,
):

	img_tensor_cpu = anom_tfrm(img).unsqueeze(0)  # 1xCxHxW

	start_time = time.perf_counter()
	with torch.inference_mode():
		img_tensor = img_tensor_cpu.to(device, non_blocking=True)
		latent, mask, ids_restore = anom_model.model.forward_encoder(x=img_tensor, mask_ratio=anom_model.mask_ratio)
		pred = anom_model.model.forward_decoder(x=latent, ids_restore=ids_restore)
		pred = anom_model.model.unpatchify(x=pred)
		pred_cpu = pred.cpu()
	mid_time = time.perf_counter()
	if timed:
		print(f"ELAPSED (anom model): {(mid_time - start_time) * 1000:.1f}ms")

	index = 3

	if not anom_model.no_mask:
		table_float_mask = torch.sigmoid(pred_cpu[:, index:(index + 1)])
		if table_mask_avg_n >= 1:
			table_float_mask_avg = circ_avg_float_mask(mask=table_float_mask, n=table_mask_avg_n, pad_value=0.5)
		else:
			table_float_mask_avg = table_float_mask
		table_mask_avg = (table_float_mask_avg >= table_mask_thres)
		if table_mask_erode_n >= 1:
			table_mask = circ_erode_bool_mask(mask=table_mask_avg, n=table_mask_erode_n)
		else:
			table_mask = table_mask_avg
		if show:
			show_float_mask(table_float_mask[0, 0], 'Table Mask Raw Float', min_val=0.0, max_val=1.0)
			show_binary_mask(table_float_mask[0, 0] >= table_mask_thres, 'Table Mask Raw')
			show_binary_mask(table_mask_avg[0, 0], 'Table Mask Avg')
			show_binary_mask(table_mask[0, 0], 'Table Mask Avg Erode (Final)')
		index += 1
	else:
		table_mask = None

	if anom_model.anomaly_mask:
		anomaly_float_mask = torch.sigmoid(pred_cpu[:, index:(index + 1)])
		if table_mask is not None:
			anomaly_float_mask *= table_mask
		anomaly_mask_raw = (anomaly_float_mask >= anomaly_mask_thres)
		if anomaly_mask_open_n >= 1:
			anomaly_mask = circ_open_mask(mask=anomaly_mask_raw, n=anomaly_mask_open_n)
		else:
			anomaly_mask = anomaly_mask_raw
		if show:
			show_float_mask(anomaly_float_mask[0, 0], 'Anomaly Mask Float', min_val=0.0, max_val=1.0)
			show_binary_mask(anomaly_mask_raw[0, 0], 'Anomaly Mask Raw')
			show_binary_mask(anomaly_mask[0, 0], 'Anomaly Mask Open (Final)')
		index += 1
	else:
		anomaly_mask = None

	reconstruction = pred_cpu[:, :3]
	reconstruction_error = torch.linalg.vector_norm(reconstruction - img_tensor_cpu, ord=recon_err_norm, dim=1, keepdim=True)
	if table_mask is not None:
		reconstruction_error *= table_mask
	reconstruction_mask_raw = (reconstruction_error >= recon_err_thres)
	if recon_denoise_n >= 1:
		reconstruction_mask = denoise_bool_mask(mask=reconstruction_mask_raw, n=recon_denoise_n, unset_thres=recon_denoise_unset_thres, set_thres=recon_denoise_set_thres)
	else:
		reconstruction_mask = reconstruction_mask_raw
	if show:
		show_normalised_image(img_tensor_cpu[0], 'Input Image')
		show_normalised_image(reconstruction[0], "Reconstruction")
		show_float_mask(reconstruction_error[0, 0], "Reconstruction Error", min_val=0.0, max_val=3.0)
		show_binary_mask(reconstruction_mask_raw[0, 0], "Reconstruction Mask Raw")
		show_binary_mask(reconstruction_mask[0, 0], "Reconstruction Mask Denoise (Final)")

	if anomaly_mask is not None:
		anom_contours = extract_mask_contours(mask=anomaly_mask, scores=anomaly_float_mask, min_area=anomaly_area_min, show=show, img=img_tensor_cpu, title="Anomaly Contours")
	else:
		anom_contours = []
	reconstruction_error_score = torch.clamp(reconstruction_error * (0.5 / recon_err_thres), max=1.0)
	recon_contours = extract_mask_contours(mask=reconstruction_mask, scores=reconstruction_error_score, min_area=recon_area_min, show=show, img=img_tensor_cpu, title="Reconstruction Contours")

	merged_contours = merge_contours(
		anom_contours=anom_contours,
		recon_contours=recon_contours,
		pred_isect_thres=pred_isect_thres,
		unmatched_score_factor=unmatched_score_factor,
		match_bbox_iou_thres=match_bbox_iou_thres,
		mean_bbox_iou_thres=mean_bbox_iou_thres,
	)

	# Note: Here we convert the PIXEL coordinates contour to an AXIS coordinates anomaly bounding box, and scale it back to the original image resolution
	xscale = float(img.width) / img_tensor_cpu.shape[-1]
	yscale = float(img.height) / img_tensor_cpu.shape[-2]
	anomalies = [Anomaly(bbox=(contour.x1 * xscale, contour.y1 * yscale, contour.w * xscale, contour.h * yscale), score=contour.score) for contour in merged_contours]

	post_time = time.perf_counter()
	if timed:
		print(f"ELAPSED (anom postprocess): {(post_time - mid_time) * 1000:.1f}ms")

	if show:
		show_anomalies(anomalies, img=img, title='Merged Anomalies')

	if sam_model:
		presam_time = time.perf_counter()
		table_mask_np = table_mask[0, 0].numpy().astype(np.uint8)
		anomalies = apply_anomaly_sam(anomalies, sam_model=sam_model, img=img, enlarge_x=sam_enlarge * xscale, enlarge_y=sam_enlarge * yscale, min_area=sam_area_min * xscale * yscale, split_penalty=sam_split_penalty, nms_thres=sam_nms_thres, table_mask_np=table_mask_np, min_table_overlap=sam_min_table_overlap, show=show)
		postsam_time = time.perf_counter()
		if timed:
			print(f"ELAPSED (anom SAM): {(postsam_time - presam_time) * 1000:.1f}ms")

	return anomalies, xscale, yscale

@dataclasses.dataclass(frozen=True)
class NOVICPredictor:
	model: NOVICModel
	category_ids: dict[str, int]

def infer_novic_model(img: PIL.Image.Image, anomalies: Sequence[Anomaly], novic_model: NOVICPredictor, border_ratio: float, min_border: float, max_border: float, score_base: float, show: bool, timed: bool) -> list[LabeledAnomaly]:
	if not anomalies:
		return []

	with torch.inference_mode():

		img_transform = novic_model.model.get_image_transform()

		anomaly_tensors = []
		for anomaly in anomalies:
			x, y, w, h = anomaly.bbox
			dx = max(min_border, min(max_border, border_ratio * w))
			dy = max(min_border, min(max_border, border_ratio * h))
			anomaly_tensors.append(img_transform(img.crop((
				max(0, math.floor(x - dx)),
				max(0, math.floor(y - dy)),
				min(img.width,  math.ceil(x + w + dx)),
				min(img.height, math.ceil(y + h + dy)),
			))))
		anomaly_batch = torch.stack(anomaly_tensors, dim=0)

		prenovic_time = time.perf_counter()
		output = novic_model.model.classify_images(anomaly_batch)
		postnovic_time = time.perf_counter()
		if timed:
			print(f"ELAPSED (novic): {(postnovic_time - prenovic_time) * 1000:.1f}ms")

		labeled_anomalies = []
		assert len(anomalies) == len(output.preds) == len(output.logprobs)
		for anomaly, preds, scores in zip(anomalies, output.preds, output.logprobs):
			category_name = ' '.join(preds[0].split())
			labeled_anomalies.append(LabeledAnomaly(
				anomaly=anomaly,
				category_name=category_name,
				category_id=novic_model.category_ids.get(category_name, -1),
				category_log_score=scores[0],
				category_score=math.pow(score_base, scores[0]),
			))

	if show:
		vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
		for lanom in labeled_anomalies:
			x, y, w, h = lanom.anomaly.bbox
			start_point = (int(x), int(y))
			end_point = (int(x + w), int(y + h))
			cv2.rectangle(vis, start_point, end_point, color=(0, 0, 255), thickness=2)  # Note: The infinitesimal floating point axis coordinates bounding box cuts through the centre of the drawn lines regardless of thickness (rectangle start/end points are inclusive pixel coordinates (origin is centre of top-left pixel))
			text_position = (int(x), int(max(y - 18, 0)))
			cv2.putText(vis, f'{lanom.category_name} ({lanom.category_score:.2f}/{lanom.anomaly.score:.2f})', text_position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=2)
		factor = 0.25 if max(vis.shape) >= 2000 else 0.5
		vis = cv2.resize(vis, dsize=None, dst=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
		cv2.imshow('NOVIC Predictions', vis)

	return labeled_anomalies

def run_inference(cfg: argparse.Namespace, image_ids: Optional[dict[str, int]]):

	image_paths = load_images_from_dir(cfg.image_dir)
	print(f"Found {len(image_paths)} image(s) in {cfg.image_dir}")
	if not image_paths:
		raise ValueError("No images found")
	if cfg.shuffle:
		random.shuffle(image_paths)

	device = torch.device(cfg.device)
	anom_model, anom_tfrm = load_anom_model(cfg=cfg, device=device)
	anom_output_channels = 3 + int(not anom_model.no_mask) + int(anom_model.anomaly_mask)
	model_res_scale = cfg.input_size / 224.0
	model_area_scale = model_res_scale * model_res_scale
	scaled_anomaly_area_min = max(1, int(round(cfg.anomaly_area_min * model_area_scale)))
	scaled_recon_area_min = max(1, int(round(cfg.recon_area_min * model_area_scale)))
	scaled_sam_area_min = max(1, int(round(cfg.sam_area_min * model_area_scale)))
	scaled_sam_enlarge = cfg.sam_enlarge * model_res_scale
	scaled_novic_min_border = cfg.novic_min_border * model_res_scale
	scaled_novic_max_border = cfg.novic_max_border * model_res_scale
	print(
		f"Anomaly head config: model={cfg.anom_model_type}, input_size={cfg.input_size}, mask_ratio={cfg.mask_ratio}, "
		f"table_mask={not anom_model.no_mask}, anomaly_mask={anom_model.anomaly_mask}, output_channels={anom_output_channels}"
	)
	print(
		f"Resolution scaling: linear={model_res_scale:.3f}, area={model_area_scale:.3f}, "
		f"anomaly_area_min={scaled_anomaly_area_min}, recon_area_min={scaled_recon_area_min}, sam_area_min={scaled_sam_area_min}, "
		f"sam_enlarge={scaled_sam_enlarge:.3f}, novic_min_border={scaled_novic_min_border:.3f}, novic_max_border={scaled_novic_max_border:.3f}"
	)

	if cfg.sam_model:
		assert SAM2ImagePredictor is not None
		sam_model = SAM2ImagePredictor.from_pretrained(cfg.sam_model)
	else:
		sam_model = None

	if cfg.declass:
		novic_loader = None
	else:
		if NOVICModel is None:
			raise ImportError("Install the public NOVIC repository and make its infer module importable")
		if not cfg.novic_model:
			raise ValueError("--novic_model is required unless --declass is set")
		novic_loader = NOVICModel(
			checkpoint=cfg.novic_model, gencfg=cfg.novic_gencfg,
			guide_targets=tuple(cfg.classes) if cfg.fixed_classes else None, device=device,
		)

	predictions = []
	with contextlib.nullcontext() if cfg.declass else novic_loader:
		novic_model = None if cfg.declass else NOVICPredictor(
			model=novic_loader,
			category_ids={target: i for i, target in enumerate(cfg.classes if cfg.fixed_classes else novic_loader.model_targets)},
		)
		for image_path in image_paths:

			if not cfg.quiet:
				print(f"Processing {image_path}...")
			img = PIL.Image.open(image_path).convert("RGB")

			anomalies, xscale, yscale = infer_anom_model(
				img=img,
				anom_model=anom_model,
				anom_tfrm=anom_tfrm,
				sam_model=sam_model,
				device=device,
				show=cfg.show,
				timed=cfg.timed,
				table_mask_avg_n=cfg.table_mask_avg_n,
				table_mask_thres=cfg.table_mask_thres,
				table_mask_erode_n=cfg.table_mask_erode_n,
				anomaly_mask_thres=cfg.anomaly_mask_thres,
				anomaly_mask_open_n=cfg.anomaly_mask_open_n,
				recon_err_norm=cfg.recon_err_norm,
				recon_err_thres=cfg.recon_err_thres,
				recon_denoise_n=cfg.recon_denoise_n,
				recon_denoise_unset_thres=cfg.recon_denoise_unset_thres,
				recon_denoise_set_thres=cfg.recon_denoise_set_thres,
				anomaly_area_min=scaled_anomaly_area_min,
				recon_area_min=scaled_recon_area_min,
				pred_isect_thres=cfg.pred_isect_thres,
				unmatched_score_factor=cfg.unmatched_score_factor,
				match_bbox_iou_thres=cfg.match_bbox_iou_thres,
				mean_bbox_iou_thres=cfg.mean_bbox_iou_thres,
				sam_enlarge=scaled_sam_enlarge,
				sam_area_min=scaled_sam_area_min,
				sam_min_table_overlap=cfg.sam_min_table_overlap,
				sam_split_penalty=cfg.sam_split_penalty,
				sam_nms_thres=cfg.sam_nms_thres,
			)

			if cfg.declass:
				labeled_anomalies = [LabeledAnomaly(anomaly=anomaly, category_name='object', category_id=0, category_log_score=0.0, category_score=1.0) for anomaly in anomalies]
			else:
				scale = math.sqrt(xscale * yscale)
				labeled_anomalies = infer_novic_model(
					img=img,
					anomalies=anomalies,
					novic_model=novic_model,
					border_ratio=cfg.novic_border_ratio,
					min_border=scaled_novic_min_border * scale,
					max_border=scaled_novic_max_border * scale,
					score_base=cfg.novic_score_base,
					show=cfg.show,
					timed=cfg.timed,
				)

			for lanom in labeled_anomalies:

				if image_ids is None:
					prediction = dict(image_name=os.path.basename(image_path))
				else:
					prediction = dict(image_id=image_ids[os.path.basename(image_path)])

				if cfg.fixed_classes:
					assert lanom.category_id >= 0
					prediction['category_id'] = lanom.category_id + 1  # Convert 0-indexed to 1-indexed classes
				else:
					prediction['category_name'] = lanom.category_name

				prediction.update(
					bbox=list(lanom.anomaly.bbox),
					score=lanom.anomaly.score * lanom.category_score,
				)

				predictions.append(prediction)

			if cfg.show:
				cv2.waitKey(0)

	if not cfg.quiet:
		print()

	if cfg.show:
		cv2.destroyAllWindows()

	with open(cfg.output, 'w') as f:
		json.dump(predictions, f, indent=2, ensure_ascii=False)  # noqa
	print(f"Results saved to {cfg.output}")

def main():

	parser = argparse.ArgumentParser(description="Inference NOVIC anomaly recognition")

	parser.add_argument('--coco_gt_path', type=str, help="COCO-style ground truth annotations file to get image IDs and classes from (any 'other' class is ignored)")
	parser.add_argument('--classes', nargs='+', help='Manually override the list of class names')
	parser.add_argument('--declass', action='store_true', help="Whether to skip classification and output everything as 'object'")
	parser.add_argument('--fixed_classes', action='store_true', help='Whether to restrict NOVIC to fixed known classes')
	parser.add_argument('--image_dir', type=str, required=True, help='Directory with input images')
	parser.add_argument('--shuffle', action='store_true', help='Whether to process images in shuffled order')
	parser.add_argument('--output', type=str, required=True, help='Output JSON file to save predictions')
	parser.add_argument('--device', type=str, default='cuda', help='Device to inference on')
	parser.add_argument('--show', action='store_true', help='Whether to show visualisations of the images')
	parser.add_argument('--quiet', action='store_true', help='Kind of quiet mode')
	parser.add_argument('--timed', action='store_true', help='Print timings')

	parser.add_argument('--anom_model_type', type=str, required=True, help='Anomaly detection model architecture')
	parser.add_argument('--anom_model', type=str, required=True, help='Path to the anomaly detection model weights')
	parser.add_argument('--input_size', default=224, type=int, help='Final square image size fed to the model')
	parser.add_argument('--mask_ratio', type=float, default=0.0, help='Mask ratio to use during inference')
	parser.add_argument('--table_mask', dest='table_mask', action='store_true', default=False, help='Enable table mask')
	parser.add_argument('--no_table_mask', dest='table_mask', action='store_false', default=False, help='Disable table mask')
	parser.add_argument('--anomaly_mask', dest='anomaly_mask', action='store_true', default=False, help='Enable anomaly mask')
	parser.add_argument('--no_anomaly_mask', dest='anomaly_mask', action='store_false', default=False, help='Disable anomaly mask')

	parser.add_argument('--novic_model', type=str, help='Path to the NOVIC model weights')
	parser.add_argument('--novic_gencfg', type=str, default='beam_k10_vnone_gp_t1_a0', help='NOVIC generation configuration')
	parser.add_argument('--novic_border_ratio', type=float, default=0.2, help='NOVIC border ratio (unidirectional enlargement)')
	parser.add_argument('--novic_min_border', type=float, default=2.0, help='Minimum NOVIC border in pixels at 224 model resolution (auto-scaled by input_size/224)')
	parser.add_argument('--novic_max_border', type=float, default=5.0, help='Maximum NOVIC border in pixels at 224 model resolution (auto-scaled by input_size/224)')
	parser.add_argument('--novic_score_base', type=float, default=1.15, help='Base of the score exponent for NOVIC scoring')

	parser.add_argument('--table_mask_avg_n', type=int, default=4, help='Table mask average filter n-value (0 = Disable)')
	parser.add_argument('--table_mask_thres', type=float, default=0.3, help='Table mask threshold')
	parser.add_argument('--table_mask_erode_n', type=int, default=0, help='Table mask erode n-value (0 = Disable)')
	parser.add_argument('--anomaly_mask_thres', type=float, default=0.45, help='Anomaly mask threshold')
	parser.add_argument('--anomaly_mask_open_n', type=int, default=0, help='Anomaly mask open n-value (0 = Disable)')
	parser.add_argument('--recon_err_norm', type=float, default=1, help='Pixelwise reconstruction error p-norm')
	parser.add_argument('--recon_err_thres', type=float, default=1.25, help='Reconstruction error to mask threshold')
	parser.add_argument('--recon_denoise_n', type=int, default=0, help='Reconstruction denoise n-value (0 = Disable)')
	parser.add_argument('--recon_denoise_unset_thres', type=float, default=3.5 / 9, help='Reconstruction denoise unset threshold')
	parser.add_argument('--recon_denoise_set_thres', type=float, default=6.5 / 9, help='Reconstruction denoise set threshold')
	parser.add_argument('--anomaly_area_min', type=int, default=39, help='Minimum area of an anomaly mask anomaly in pixels at 224 model resolution (auto-scaled by (input_size/224)^2)')
	parser.add_argument('--recon_area_min', type=int, default=24, help='Minimum area of a reconstruction anomaly in pixels at 224 model resolution (auto-scaled by (input_size/224)^2)')
	parser.add_argument('--pred_isect_thres', type=float, default=0.25, help='Minimum reconstruction to anomaly mask intersection threshold for matching predictions')
	parser.add_argument('--unmatched_score_factor', type=float, default=0.8, help='Multiplicative score scale factor for unmatched (anomaly mask vs reconstruction) predictions')
	parser.add_argument('--match_bbox_iou_thres', type=float, default=0.65, help='Bounding box IoU threshold for ignoring a reconstruction anomaly if there is a well-matched anomaly mask (OR)')
	parser.add_argument('--mean_bbox_iou_thres', type=float, default=0.3, help='Mean bounding box IoU threshold for keeping a multiplexed anomaly (one-to-many when matching reconstruction vs anomaly mask)')

	parser.add_argument('--sam_model', type=str, default=None, help='SAM model (e.g. facebook/sam2.1-hiera-large) to use to post-process anomaly detections (default: disabled)')
	parser.add_argument('--sam_enlarge', type=float, default=0.0, help='Number of pixels at 224 model resolution to enlarge anomaly boxes by prior to SAM (auto-scaled by input_size/224)')
	parser.add_argument('--sam_area_min', type=int, default=20, help='Minimum area of a SAM anomaly in pixels at 224 model resolution (auto-scaled by (input_size/224)^2)')
	parser.add_argument('--sam_min_table_overlap', type=float, default=0.4, help='Minimum overlap ratio with the table mask in order to keep a SAM output contour')
	parser.add_argument('--sam_split_penalty', type=float, default=0.75, help='Confidence penalty if SAM splits a bounding box')
	parser.add_argument('--sam_nms_thres', type=float, default=0.85, help='SAM NMS threshold')

	args = parser.parse_args()

	if args.quiet:
		logging.getLogger().setLevel(logging.WARNING)

	if args.coco_gt_path:
		with open(args.coco_gt_path, 'r') as f:
			coco_data = json.load(f)
		all_categories = coco_data["categories"]
		assert all(cat['id'] == i for i, cat in enumerate(all_categories, 1)), "Category IDs must be sequential"
		coco_classes = [cat['name'] for cat in all_categories if cat.get("name", "").lower() != "other"]
		image_ids = {image['file_name']: image['id'] for image in coco_data["images"]}
		if len(image_ids) != len(coco_data['images']):
			raise ValueError("Duplicate images!")
	else:
		coco_classes = None
		image_ids = None

	if args.declass:
		args.classes = ['object']
	elif args.classes is None:
		args.classes = coco_classes
	elif coco_classes is not None:
		for c, (coco_class, args_class) in enumerate(itertools.zip_longest(coco_classes, args.classes, fillvalue=None), 1):
			print(f"ID {c}: {coco_class} --> {args_class}")

	run_inference(cfg=args, image_ids=image_ids)

if __name__ == "__main__":
	main()
