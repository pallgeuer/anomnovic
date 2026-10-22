# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted for AnomNOVIC by Philipp Allgeuer and Jan-Gerrit Habekost.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
from PIL.Image import Resampling

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import math
import torchvision.transforms.functional as TF
import random
import torch


class SingleClassImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, is_train=False, eye='right_eye', ycb_anomaly_path=None, no_mask=False, anomaly_mask=False,
                 input_size=224, max_resize_zoom=448, min_nbr_anomalies=5, max_nbr_anomalies=30, rotation_angle=30, min_anomaly_size=0.1, max_anomaly_size=0.7,
                 dynamic_anomalies=False, anomaly_shadows=False, anomaly_shadow_prob=0.7, anomaly_shadow_dark_max=0.25,
                 anomaly_shadow_offset_std_factor=0.15):

        self.input_size = input_size
        self.max_resize_zoom = max_resize_zoom
        self.min_nbr_anomalies = min_nbr_anomalies
        self.max_nbr_anomalies = max_nbr_anomalies
        self.rotation_angle = rotation_angle
        self.min_anomaly_size = min_anomaly_size
        self.max_anomaly_size = max_anomaly_size
        self.virtual_anomaly_base_size = max(1, int(round(70 * self.input_size / 224)))

        self.is_train = is_train
        self.eye = eye
        self.no_mask = no_mask
        self.anomaly_mask = anomaly_mask
        self.dynamic_anomalies = dynamic_anomalies
        self.anomaly_shadows = anomaly_shadows
        self.anomaly_shadow_prob = anomaly_shadow_prob
        self.anomaly_shadow_dark_max = anomaly_shadow_dark_max
        self.anomaly_shadow_offset_std_factor = anomaly_shadow_offset_std_factor
        self.imagenet_mean_over_std = (torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32) / torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32))
        self.virtual_anomalies_path = Path(ycb_anomaly_path) if ycb_anomaly_path else None
        if self.is_train and (self.virtual_anomalies_path is None or not self.virtual_anomalies_path.is_dir()):
            raise ValueError("Training requires an existing virtual anomaly image directory")

        # Precompute anomaly file list filtered by extension.
        if self.is_train:
            self.virtual_anomaly_files = [str(p) for p in self.virtual_anomalies_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            if not self.virtual_anomaly_files:
                raise ValueError(f"No virtual anomaly images found in {self.virtual_anomalies_path}")
            self.virtual_anomalies = []
            for anomaly_file in self.virtual_anomaly_files:
                with Image.open(anomaly_file) as anomaly_img:
                    if anomaly_img.mode not in {"RGB", "RGBA"}:
                        raise ValueError(f"Virtual anomaly image must be RGB or RGBA: {anomaly_file} (got mode={anomaly_img.mode})")
                    fit_box_size = self.virtual_anomaly_base_size * 2
                    fit_scale = min(fit_box_size / anomaly_img.width, fit_box_size / anomaly_img.height)
                    resized_w = max(1, int(round(anomaly_img.width * fit_scale)))
                    resized_h = max(1, int(round(anomaly_img.height * fit_scale)))
                    self.virtual_anomalies.append(anomaly_img.resize((resized_w, resized_h), Resampling.BICUBIC))

        super(SingleClassImageFolder, self).__init__(root, loader=self.pil_loader, extensions=('jpg', 'jpeg', 'png'), transform=transform)
        self.imgs = self.samples
        self.to_tensor = transforms.ToTensor()
        self.norm_color = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.rand_perspective = transforms.RandomPerspective(distortion_scale=0.7, p=0.33)

    @staticmethod
    def _sample_truncated_gaussian_shift(std_px):
        if std_px <= 0.0:
            return 0, 0
        max_radius = 2.5 * std_px
        dx = 0.0
        dy = 0.0
        for _ in range(32):
            dx = random.gauss(0.0, std_px)
            dy = random.gauss(0.0, std_px)
            if math.hypot(dx, dy) <= max_radius:
                break
        else:
            radius = math.hypot(dx, dy)
            if radius > 0:
                scale = max_radius / radius
                dx *= scale
                dy *= scale
        return int(round(dx)), int(round(dy))

    @staticmethod
    def _extract_anomaly_alpha_and_mask(original_anomaly_np, anomaly_mode, end_y, end_x):
        if anomaly_mode == "RGBA":
            anomaly_alpha_np = original_anomaly_np[:end_y, :end_x, 3].astype(np.float32) / 255.0
        else:
            anomaly_alpha_np = np.any(original_anomaly_np[:end_y, :end_x, :] != 0, axis=2).astype(np.float32)
        anom_mask = (anomaly_alpha_np > 0).astype(np.uint8)
        return anomaly_alpha_np, anom_mask

    @staticmethod
    def _accumulate_region_max(target_map, patch_map, top_left_x, top_left_y):
        end_y, end_x = patch_map.shape
        target_view = target_map[top_left_y:top_left_y + end_y, top_left_x:top_left_x + end_x]
        np.maximum(target_view, patch_map, out=target_view)

    @staticmethod
    def _build_shifted_shadow_request_map(anom_mask, top_left_x, top_left_y, shift_dx, shift_dy, img_h, img_w):
        end_y, end_x = anom_mask.shape
        orig_x0, orig_y0 = top_left_x, top_left_y
        orig_x1, orig_y1 = orig_x0 + end_x, orig_y0 + end_y
        shift_x0, shift_y0 = orig_x0 + shift_dx, orig_y0 + shift_dy
        shift_x1, shift_y1 = shift_x0 + end_x, shift_y0 + end_y

        union_x0 = max(0, min(orig_x0, shift_x0))
        union_y0 = max(0, min(orig_y0, shift_y0))
        union_x1 = min(img_w, max(orig_x1, shift_x1))
        union_y1 = min(img_h, max(orig_y1, shift_y1))
        if union_x0 >= union_x1 or union_y0 >= union_y1:
            return None

        union_w = union_x1 - union_x0
        union_h = union_y1 - union_y0
        shifted_mask = np.zeros((union_h, union_w), dtype=np.float32)

        dst_x0 = max(shift_x0, union_x0)
        dst_y0 = max(shift_y0, union_y0)
        dst_x1 = min(shift_x1, union_x1)
        dst_y1 = min(shift_y1, union_y1)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return None

        src_x0 = dst_x0 - shift_x0
        src_y0 = dst_y0 - shift_y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        shifted_mask[dst_y0 - union_y0:dst_y1 - union_y0, dst_x0 - union_x0:dst_x1 - union_x0] = anom_mask[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
        return union_x0, union_y0, shifted_mask

    @staticmethod
    def _blend_anomaly_patch(augmented_image_tensor, anomaly_img_tensor, top_left_x, top_left_y, end_x, end_y, anomaly_alpha_np):
        alpha_mask_tensor = torch.from_numpy(anomaly_alpha_np).to(dtype=augmented_image_tensor.dtype, device=augmented_image_tensor.device).unsqueeze(0)
        augmented_patch = augmented_image_tensor[:, top_left_y:top_left_y + end_y, top_left_x:top_left_x + end_x]
        anomaly_patch = anomaly_img_tensor[:, :end_y, :end_x]
        augmented_patch.mul_(1.0 - alpha_mask_tensor).add_(anomaly_patch * alpha_mask_tensor)

    def pil_loader(self, path):
        with Image.open(path[0]) as loaded_img:
            if loaded_img.size == (4208, 3120):
                img = loaded_img.resize((1920, 1440), Resampling.BICUBIC)
            else:
                img = loaded_img.copy()
        if min(img.size) < self.input_size:
            raise ValueError(f"Input image is smaller than input_size={self.input_size}: {path[0]} (got size={img.size})")
        if img.mode != "RGB":
            raise ValueError(f"Input image must be RGB: {path[0]} (got mode={img.mode})")
        if self.is_train:
            if self.no_mask:
                mask = Image.new('L', img.size, color=255)
            else:
                with Image.open(path[1]) as loaded_mask:
                    mask = loaded_mask.copy()
                if mask.size != img.size:
                    raise ValueError(f"Mask image must be same size as input image: {mask.size} vs {img.size}")
                if mask.mode == "1":
                    mask = mask.convert("L")
                elif mask.mode != "L":
                    raise ValueError(f"Mask image must be grayscale (mode L) or binary (mode 1): {path[1]} (got mode={mask.mode})")
        else:
            mask = Image.new('L', (img.size[0], img.size[1]), color=0)  # NOTE: This used to be RGB and without specifying an explicit color

        return [img, mask]

    def find_classes(self, directory):
        return ['single_class'], {'single_class': 0}

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        img, mask = sample
        assert self.transform is not None
        if self.transform is not None:
            # Resize
            if self.is_train:
                max_resize = max(self.max_resize_zoom, self.input_size)
                random_size = random.randint(self.input_size, max_resize)
            else:
                random_size = self.input_size
            resize = transforms.Resize(size=(random_size, random_size), interpolation=Image.BICUBIC)
            img = resize(img)
            mask = resize(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.input_size, self.input_size))
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if self.is_train and random.random() < 0.15 and not self.no_mask:
                random_angle = random.randint(-self.rotation_angle, self.rotation_angle)
                random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                img = TF.rotate(img, angle=random_angle, interpolation=Image.BICUBIC,fill=random_color)
                mask = TF.rotate(mask, angle=random_angle, interpolation=Image.BICUBIC)
                # Apply identical perspective transformation to both image and mask.
                width, height = img.size
                startpoints = [(0, 0), (width, 0), (width, height), (0, height)]
                max_distortion = 0.2 * min(width, height)  # adjust distortion factor as needed
                endpoints = []
                for x, y in startpoints:
                    offset_x = random.uniform(-max_distortion, max_distortion)
                    offset_y = random.uniform(-max_distortion, max_distortion)
                    endpoints.append((x + offset_x, y + offset_y))
                img = TF.perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC,fill=random_color)
                mask = TF.perspective(mask, startpoints, endpoints, interpolation=Image.BICUBIC)

            if self.is_train and random.random() < 0.33 and self.no_mask:
                increase_factor=1.2
                tmp_resize = transforms.Resize(size=(int(increase_factor * self.input_size), int(increase_factor * self.input_size)), interpolation=Image.BICUBIC)
                random_angle = random.randint(-self.rotation_angle, self.rotation_angle)
                random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                img = TF.rotate(img, angle=random_angle, interpolation=Image.BICUBIC,fill=random_color)
                img = tmp_resize(img)
                img = TF.center_crop(img, (self.input_size, self.input_size))
                mask = TF.rotate(mask, angle=random_angle, interpolation=Image.BICUBIC)

            if self.is_train:
                sharpness_factor = (random.random() + 0.33) * 1.5
                img = TF.adjust_sharpness(img, sharpness_factor)

            anomaly_mask = np.zeros((self.input_size, self.input_size), dtype=np.uint8).reshape((self.input_size, self.input_size, 1))
            mask = self.to_tensor(mask)
            anomaly_mask = self.to_tensor(anomaly_mask)
            if self.is_train:
                img = self.transform[1](img)
                img = self.transform[2](img)
            else:
                img = self.transform(img)

            augmented_image_tensor = img.clone() if self.is_train else img

            if self.is_train:
                # Save a copy of the table mask before converting to tensor
                table_np = mask.mul(255).byte().cpu().squeeze(0).numpy()
                if self.no_mask:
                    indices = np.argwhere(table_np > 0)
                else:
                    indices = np.argwhere(table_np > 128)
                if self.dynamic_anomalies:
                    table_area_factor = float(len(indices)) / float(self.input_size * self.input_size)
                    scaled_max_nbr_anomalies = int(round(self.max_nbr_anomalies * table_area_factor))
                    max_nbr_anomalies = max(self.min_nbr_anomalies, scaled_max_nbr_anomalies)
                    nbr_anomalies = random.randint(self.min_nbr_anomalies, max_nbr_anomalies)
                else:
                    nbr_anomalies = random.randint(self.min_nbr_anomalies, self.max_nbr_anomalies)
                shadow_darkening_req_max = None
                pasted_alpha_max = None
                if self.anomaly_shadows:
                    img_h = augmented_image_tensor.shape[1]
                    img_w = augmented_image_tensor.shape[2]
                    shadow_darkening_req_max = np.zeros((img_h, img_w), dtype=np.float32)
                    pasted_alpha_max = np.zeros((img_h, img_w), dtype=np.float32)
                for kk in range(nbr_anomalies):
                    anomaly_img = random.choice(self.virtual_anomalies)
                    angle = random.uniform(-180, 180)
                    if anomaly_img.mode == "RGBA":
                        anomaly_img = anomaly_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
                    else:
                        anomaly_img = anomaly_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))

                    scale = 0.5 * random.uniform(self.min_anomaly_size, self.max_anomaly_size)

                    new_w = max(1, int(round(anomaly_img.width * scale)))
                    new_h = max(1, int(round(anomaly_img.height * scale)))
                    anomaly_img = anomaly_img.resize((new_w, new_h), resample=Image.BICUBIC)
                    if len(indices) == 0:
                        continue
                    chosen_idx = indices[random.randint(0, len(indices) - 1)]
                    y_pos, x_pos = chosen_idx  # row, col in numpy array
                    top_left_x = x_pos - new_w // 2
                    top_left_y = y_pos - new_h // 2

                    original_anomaly_img = anomaly_img.copy()
                    anomaly_img_rgb = anomaly_img.convert("RGB")

                    random_brightness = random.uniform(0.5, 1.5)
                    anomaly_img_rgb = TF.adjust_brightness(anomaly_img_rgb, random_brightness)
                    random_brightness = random.uniform(0.5, 1.5)
                    anomaly_img_rgb = TF.adjust_contrast(anomaly_img_rgb, random_brightness)
                    random_brightness = random.uniform(0.5, 1.5)
                    anomaly_img_rgb = TF.adjust_saturation(anomaly_img_rgb, random_brightness)
                    random_brightness = random.uniform(0.5, 1.5)
                    anomaly_img_rgb = TF.adjust_sharpness(anomaly_img_rgb, random_brightness)
                    random_brightness = random.uniform(0.5, 1.5)
                    anomaly_img_rgb = TF.adjust_gamma(anomaly_img_rgb, random_brightness)
                    random_brightness = random.uniform(-0.5, 0.5)
                    anomaly_img_rgb = TF.adjust_hue(anomaly_img_rgb, random_brightness)
                    anomaly_img_tensor = self.transform[2](anomaly_img_rgb)

                    # Ensure the augmented image and the original one have the same color
                    end_y = min(new_h, augmented_image_tensor.shape[1] - top_left_y)
                    end_x = min(new_w, augmented_image_tensor.shape[2] - top_left_x)
                    if end_y > 0 and end_x > 0 and top_left_y >= 0 and top_left_x >= 0:
                        original_anomaly_np = np.array(original_anomaly_img)
                        anomaly_alpha_np, anom_mask = self._extract_anomaly_alpha_and_mask(original_anomaly_np, original_anomaly_img.mode, end_y, end_x)
                        self._blend_anomaly_patch(augmented_image_tensor, anomaly_img_tensor, top_left_x, top_left_y, end_x, end_y, anomaly_alpha_np)
                        if self.anomaly_shadows:
                            self._accumulate_region_max(pasted_alpha_max, anomaly_alpha_np, top_left_x, top_left_y)
                            if random.random() < self.anomaly_shadow_prob and self.anomaly_shadow_dark_max > 0.0:
                                mask_area = int(np.count_nonzero(anom_mask))
                                if mask_area > 0:
                                    shadow_std_px = self.anomaly_shadow_offset_std_factor * math.sqrt(float(mask_area))
                                    shadow_dx, shadow_dy = self._sample_truncated_gaussian_shift(shadow_std_px)
                                    shadow_req = self._build_shifted_shadow_request_map(anom_mask, top_left_x, top_left_y, shadow_dx, shadow_dy, img_h, img_w)
                                    if shadow_req is not None:
                                        dark_strength = random.uniform(0.0, self.anomaly_shadow_dark_max)
                                        if dark_strength > 0.0:
                                            union_x0, union_y0, shifted_shadow_mask = shadow_req
                                            if np.any(shifted_shadow_mask > 0.0):
                                                self._accumulate_region_max(shadow_darkening_req_max, shifted_shadow_mask * dark_strength, union_x0, union_y0)
                        if self.anomaly_mask:
                            # Ensure anomaly_mask is uint8 and update it
                            anomaly_mask = anomaly_mask.to(dtype=torch.uint8)
                            anomaly_mask[:, top_left_y:int(top_left_y + end_y), top_left_x:int(top_left_x + end_x)] |= torch.from_numpy(anom_mask).to(dtype=torch.uint8)
                if self.anomaly_shadows:
                    shadow_strength_np = shadow_darkening_req_max * (1.0 - pasted_alpha_max)
                    if np.any(shadow_strength_np > 0.0):
                        shadow_strength_tensor = torch.from_numpy(shadow_strength_np).to(device=augmented_image_tensor.device, dtype=augmented_image_tensor.dtype).unsqueeze(0)
                        mean_over_std = self.imagenet_mean_over_std.to(device=augmented_image_tensor.device, dtype=augmented_image_tensor.dtype).view(3, 1, 1)
                        augmented_image_tensor.mul_(1.0 - shadow_strength_tensor).add_(-mean_over_std * shadow_strength_tensor)

            anomaly_mask = anomaly_mask.to(dtype=torch.float32)
            if self.anomaly_mask and not self.no_mask:
                anomaly_mask *= (mask > 0).to(dtype=anomaly_mask.dtype)
            return img, augmented_image_tensor, mask, anomaly_mask

    def make_dataset(self, directory, class_to_idx, extensions, is_valid_file=None, allow_empty=True):
        images = []
        # Added: check for flat structure (images directly in root)
        entries = os.listdir(directory)
        if entries and not any(os.path.isdir(os.path.join(directory, entry)) for entry in entries):
            # Flat structure detected
            for fname in sorted(entries):
                if fname.lower().endswith(extensions):
                    path = os.path.join(directory, fname)
                    images.append(([path, ""], class_to_idx['single_class']))
            return images

        for scene in sorted(os.listdir(directory)):
            scene_path = os.path.join(directory, scene)
            if not os.path.isdir(scene_path):
                continue
            eye_dir = os.path.join(scene_path, self.eye)
            if not os.path.isdir(eye_dir):
                continue
            for fname in sorted(os.listdir(eye_dir)):
                if fname.startswith('mask_') or not fname.lower().endswith(extensions):
                    continue
                image_path = os.path.join(eye_dir, fname)
                stem, _ = os.path.splitext(fname)
                mask_path = os.path.join(eye_dir, f'mask_{stem}.png')
                if self.is_train and not self.no_mask and not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
                images.append(([image_path, mask_path], class_to_idx['single_class']))

        return images

def build_dataset(is_train, args, test=False, ycb_anomaly_path=None, no_mask=False, anomaly_mask=False,
                  min_nbr_anomalies=5, max_nbr_anomalies=30, rotation_angle=30, min_anomaly_size=0.1, max_anomaly_size=0.7, dynamic_anomalies=False,
                  anomaly_shadows=False, anomaly_shadow_prob=0.7, anomaly_shadow_dark_max=0.25, anomaly_shadow_offset_std_factor=0.15):
    transform = build_transform(is_train, args)
    eye = getattr(args, 'eye', 'right_eye')
    root = args.data_path  # Now input is <data_root> containing scenes
    # Use the new parameter when constructing the dataset
    print(f'virtual anomalies path: {ycb_anomaly_path}')
    print(f'val data path: {args.val_data}')
    all_images = SingleClassImageFolder(root, transform=transform, is_train=is_train, ycb_anomaly_path=ycb_anomaly_path, eye=eye, input_size=args.input_size, max_resize_zoom=args.max_resize_zoom,
                                        min_nbr_anomalies=min_nbr_anomalies, max_nbr_anomalies=max_nbr_anomalies, rotation_angle=rotation_angle,
                                        min_anomaly_size=min_anomaly_size, max_anomaly_size=max_anomaly_size, dynamic_anomalies=dynamic_anomalies,
                                        anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob, anomaly_shadow_dark_max=anomaly_shadow_dark_max,
                                        anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor).imgs
    if not test:
        if args.val_data is None:
            train_images, val_images = train_test_split(all_images, test_size=0.1, random_state=42)
            if is_train:
                dataset = SingleClassImageFolder(root, transform=transform, is_train=is_train, eye=eye, ycb_anomaly_path=ycb_anomaly_path, no_mask=no_mask, anomaly_mask=anomaly_mask, input_size=args.input_size, max_resize_zoom=args.max_resize_zoom,
                                                 min_nbr_anomalies=min_nbr_anomalies, max_nbr_anomalies=max_nbr_anomalies, rotation_angle=rotation_angle,
                                                 min_anomaly_size=min_anomaly_size, max_anomaly_size=max_anomaly_size, dynamic_anomalies=dynamic_anomalies,
                                                 anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob, anomaly_shadow_dark_max=anomaly_shadow_dark_max,
                                                 anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor)
                dataset.imgs = train_images
                dataset.samples = train_images
            else:
                dataset = SingleClassImageFolder(root, transform=transform, is_train=is_train, eye=eye, no_mask=no_mask, anomaly_mask=anomaly_mask, input_size=args.input_size, max_resize_zoom=args.max_resize_zoom,
                                                 min_nbr_anomalies=min_nbr_anomalies, max_nbr_anomalies=max_nbr_anomalies, rotation_angle=rotation_angle,
                                                 min_anomaly_size=min_anomaly_size, max_anomaly_size=max_anomaly_size, dynamic_anomalies=dynamic_anomalies,
                                                 anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob, anomaly_shadow_dark_max=anomaly_shadow_dark_max,
                                                 anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor)
                dataset.imgs = val_images
                dataset.samples = val_images
        else:
            if is_train:
                dataset = SingleClassImageFolder(root, transform=transform, is_train=is_train, eye=eye, ycb_anomaly_path=ycb_anomaly_path, no_mask=no_mask, anomaly_mask=anomaly_mask, input_size=args.input_size, max_resize_zoom=args.max_resize_zoom,
                                                 min_nbr_anomalies=min_nbr_anomalies, max_nbr_anomalies=max_nbr_anomalies, rotation_angle=rotation_angle,
                                                 min_anomaly_size=min_anomaly_size, max_anomaly_size=max_anomaly_size, dynamic_anomalies=dynamic_anomalies,
                                                 anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob, anomaly_shadow_dark_max=anomaly_shadow_dark_max,
                                                 anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor)
                dataset.imgs = dataset.imgs
                dataset.samples = dataset.imgs
            else:
                print(f'val data path: {args.val_data}')
                dataset = SingleClassImageFolder(args.val_data, transform=transform, is_train=is_train, eye=eye, no_mask=no_mask, anomaly_mask=anomaly_mask, input_size=args.input_size, max_resize_zoom=args.max_resize_zoom,
                                                 min_nbr_anomalies=min_nbr_anomalies, max_nbr_anomalies=max_nbr_anomalies, rotation_angle=rotation_angle,
                                                 min_anomaly_size=min_anomaly_size, max_anomaly_size=max_anomaly_size, dynamic_anomalies=dynamic_anomalies,
                                                 anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob, anomaly_shadow_dark_max=anomaly_shadow_dark_max,
                                                 anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor)
                all_images = dataset.imgs
                dataset.imgs = all_images
                dataset.samples = all_images
    else:
        dataset = SingleClassImageFolder(root, transform=transform, is_train=False, eye=eye, input_size=args.input_size, dynamic_anomalies=dynamic_anomalies,
                                         anomaly_shadows=anomaly_shadows, anomaly_shadow_prob=anomaly_shadow_prob,
                                         anomaly_shadow_dark_max=anomaly_shadow_dark_max, anomaly_shadow_offset_std_factor=anomaly_shadow_offset_std_factor)
        dataset.imgs = all_images
        dataset.samples = all_images

    print(f"Dataset size: {len(dataset)} images")
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
            separate=True,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
