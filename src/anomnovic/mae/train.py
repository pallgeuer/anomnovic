# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted for AnomNOVIC by Philipp Allgeuer and Jan-Gerrit Habekost.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from . import misc, models_mae
from .misc import NativeScalerWithGradNormCount as NativeScaler
from .datasets import build_dataset
from .engine_pretrain import train_one_epoch

IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_DEFAULT_STD, dtype=torch.float32).view(3, 1, 1)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * world_size)')
    parser.add_argument('--epochs', default=400, type=int,
                        help='Number of training epochs')
    parser.add_argument('--stop_after_epoch', default=None, type=int, metavar='N',
                        help='Stop after completing epochs 0 through N-1, while keeping --epochs as the full schedule length')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', choices=('mae_vit_base_patch16',), metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='Final square image size fed to the model')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--drop_mask_zero', default=0.8, type=float,
                        help='Epoch fraction after which mask_ratio is set to 0.0 (e.g. 0.8 means last 20 percent are unmasked)')
    parser.add_argument('--fade_mask_ratio', default=-1, type=int,
                        help='Epoch index at which to start cosine-fading mask_ratio down to essentially 0 by the final training batch (disabled if < 0)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    # NEW: Add missing transformation parameters
    parser.add_argument('--color_jitter', default=0.4, type=float,
                        help='Color jitter factor used when AutoAugment is disabled')
    parser.add_argument('--aa', default=None, type=str,
                        help='AutoAugment policy string (set to None to disable)')
    parser.add_argument('--reprob', default=0.0, type=float,
                        help='Random erasing probability')
    parser.add_argument('--remode', default='pixel', type=str,
                        help='Random erasing mode')
    parser.add_argument('--recount', default=0, type=int,
                        help='Random erasing count')

    # Dataset parameters
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to training dataset root')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training (e.g. cuda or cpu)')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_weights_only', action='store_true', help='Do not attempt to resume epoch number, optimizer state, and such, even if present in the checkpoint')
    parser.add_argument('--resume_nonstrict', action='store_true', help='Whether to be non-strict about loading the state dict from a checkpoint')
    parser.add_argument('--resume_frozen', action='store_true', help='Whether any weights successfully loaded from a checkpoint are frozen')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder parameters after loading from resume checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch (usually set automatically when resuming)')
    parser.add_argument('--num_workers', default=14, type=int,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank passed by distributed launchers')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Use ITP-specific distributed initialization')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # additional data for NICOL paper
    parser.add_argument('--virtual_anomalies', required=True, type=str, metavar='VIRT_ANOM',
                        help='Path to virtual anomaly image dataset used for synthetic anomaly pasting')
    parser.add_argument('--val_data', default=None, type=str, metavar='VIRT_ANOM',
                        help='Optional separate validation dataset root (if unset, validation split is taken from data_path)')

    # masks
    parser.add_argument('--no_mask', action='store_true',
                        help='Ignore table masks by replacing them with ones')
    parser.add_argument('--anomaly_mask', action='store_true',
                        help='Enable anomaly mask prediction channel and loss')

    # loss weights

    parser.add_argument('--anomaly_reconstruction_weight', default=5.0, type=float,
                        help='Multiplier for reconstruction loss on anomalous pixels')
    parser.add_argument('--anomaly_mask_weight', default=10.0, type=float,
                        help='Extra multiplier for anomaly-mask BCE loss on anomalous pixels')
    parser.add_argument('--nonanomaly_mask_weight', default=1.0, type=float,
                        help='Multiplier for anomaly-mask BCE loss on non-anomalous pixels')
    parser.add_argument('--table_reconstruction_weight', default=10.0, type=float,
                        help='Multiplier for reconstruction loss inside the table mask')
    parser.add_argument('--table_mask_weight', default=1.0, type=float,
                        help='Multiplier for table-mask BCE loss')

    # dataloader augmentations

    parser.add_argument('--max_resize_zoom', default=448, type=int,
                        help='Maximum size for random square resize before random cropping to input_size')
    parser.add_argument('--min_nbr_anomalies', default=5, type=int,
                        help='Minimum number of synthetic anomalies pasted per training sample')
    parser.add_argument('--max_nbr_anomalies', default=30, type=int,
                        help='Maximum number of synthetic anomalies pasted per training sample')
    parser.add_argument('--dynamic_anomalies', action='store_true',
                        help='Scale maximum synthetic anomalies by table-mask area while keeping the minimum fixed')
    parser.add_argument('--anomaly_shadows', action='store_true',
                        help='Enable synthetic cast shadows for pasted virtual anomalies')
    parser.add_argument('--anomaly_shadow_prob', default=0.7, type=float,
                        help='Per-anomaly probability of adding a synthetic shadow')
    parser.add_argument('--anomaly_shadow_dark_max', default=0.25, type=float,
                        help='Maximum shadow darkening strength (sampled uniformly from 0 to this value)')
    parser.add_argument('--anomaly_shadow_offset_std_factor', default=0.15, type=float,
                        help='Shadow offset std factor multiplied by sqrt(pasted anomaly area in pixels)')
    parser.add_argument('--rotation_angle', default=30.0, type=float,
                        help='Maximum absolute rotation angle (degrees) for geometric augmentations')
    parser.add_argument('--min_anomaly_size', default=0.1, type=float,
                        help='Minimum scale factor applied to synthetic anomaly patches')
    parser.add_argument('--max_anomaly_size', default=0.7, type=float,
                        help='Maximum scale factor applied to synthetic anomaly patches')

    return parser


def save_epoch_preview(model, dataset_val, device, args, epoch):
    if len(dataset_val) == 0:
        return None

    sample_idx = random.randrange(len(dataset_val))
    image_cpu, _, table_mask_cpu, anomaly_mask_cpu = dataset_val[sample_idx]

    images = image_cpu.unsqueeze(0).to(device, non_blocking=True)
    table_mask = table_mask_cpu.unsqueeze(0).to(device, non_blocking=True)
    anomaly_mask = anomaly_mask_cpu.unsqueeze(0).to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            _, pred_patches, _ = model(images, images, mask_ratio=0.0, table_mask=table_mask, anomaly_mask=anomaly_mask)
            reconstruction = model.unpatchify(pred_patches).squeeze(0)

    input_path = os.path.join(args.output_dir, f"epoch_{epoch}_input.png")
    recon_path = os.path.join(args.output_dir, f"epoch_{epoch}_reconstruction.png")

    input_img = (image_cpu * IMAGENET_STD_TENSOR + IMAGENET_MEAN_TENSOR).clamp(0, 1)
    to_pil_image(input_img).save(input_path)

    reconstruction_cpu = reconstruction.float().cpu()
    recon_img = (reconstruction_cpu[:3] * IMAGENET_STD_TENSOR + IMAGENET_MEAN_TENSOR).clamp(0, 1)
    to_pil_image(recon_img).save(recon_path)

    output_paths = {
        "sample_idx": sample_idx,
        "input": input_path,
        "reconstruction": recon_path,
    }

    if not args.no_mask:
        mask_img = reconstruction_cpu[3].clamp(0, 1)
        mask_path = os.path.join(args.output_dir, f"epoch_{epoch}_mask.png")
        to_pil_image(mask_img).save(mask_path)
        output_paths["mask"] = mask_path

    if args.anomaly_mask:
        anomaly_channel_idx = 3 if args.no_mask else 4
        anomaly_mask_img = reconstruction_cpu[anomaly_channel_idx].clamp(0, 1)
        anomaly_mask_path = os.path.join(args.output_dir, f"epoch_{epoch}_anomaly_mask.png")
        to_pil_image(anomaly_mask_img).save(anomaly_mask_path)
        output_paths["anomaly_mask"] = anomaly_mask_path

    return output_paths


def main(args):
    misc.init_distributed_mode(args)

    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError(f"--stop_after_epoch must be between 1 and --epochs ({args.epochs}) => Got {args.stop_after_epoch}")
    if args.fade_mask_ratio >= args.epochs:
        raise ValueError(f"--fade_mask_ratio must be < {args.epochs} (total num epochs) or negative (disabled) => Got {args.fade_mask_ratio}")

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False  # Note: Do not use cuDNN benchmark as mask ratio is randomized (see models_mae.py)

    # Use the build_dataset function to handle a flat folder and perform a 90/10 split.
    print(f'anomaly path: {args.virtual_anomalies}')
    dataset_train = build_dataset(is_train=True, args=args, ycb_anomaly_path=args.virtual_anomalies, no_mask=args.no_mask, anomaly_mask=args.anomaly_mask,
                                  min_nbr_anomalies=args.min_nbr_anomalies, max_nbr_anomalies=args.max_nbr_anomalies, dynamic_anomalies=args.dynamic_anomalies,
                                  anomaly_shadows=args.anomaly_shadows, anomaly_shadow_prob=args.anomaly_shadow_prob,
                                  anomaly_shadow_dark_max=args.anomaly_shadow_dark_max, anomaly_shadow_offset_std_factor=args.anomaly_shadow_offset_std_factor,
                                  rotation_angle=args.rotation_angle, min_anomaly_size=args.min_anomaly_size, max_anomaly_size=args.max_anomaly_size)
    dataset_val = build_dataset(is_train=False, args=args, ycb_anomaly_path=args.virtual_anomalies, no_mask=args.no_mask, anomaly_mask=args.anomaly_mask,
                                min_nbr_anomalies=args.min_nbr_anomalies, max_nbr_anomalies=args.max_nbr_anomalies, dynamic_anomalies=args.dynamic_anomalies,
                                anomaly_shadows=args.anomaly_shadows, anomaly_shadow_prob=args.anomaly_shadow_prob,
                                anomaly_shadow_dark_max=args.anomaly_shadow_dark_max, anomaly_shadow_offset_std_factor=args.anomaly_shadow_offset_std_factor,
                                rotation_angle=args.rotation_angle, min_anomaly_size=args.min_anomaly_size, max_anomaly_size=args.max_anomaly_size)
    print("Train samples: {}".format(len(dataset_train)))
    print("Val samples: {}".format(len(dataset_val)))

    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=3 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # define the model
    model = models_mae.__dict__[args.model](img_size=args.input_size, no_mask=args.no_mask, anomaly_mask=args.anomaly_mask,
                                            anomaly_reconstruction_weight=args.anomaly_reconstruction_weight,
                                            anomaly_mask_weight=args.anomaly_mask_weight, nonanomaly_mask_weight=args.nonanomaly_mask_weight,
                                            table_reconstruction_weight=args.table_reconstruction_weight, table_mask_weight=args.table_mask_weight)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    end_epoch = args.stop_after_epoch if args.stop_after_epoch is not None else args.epochs
    if args.stop_after_epoch is not None and end_epoch <= args.start_epoch:
        raise ValueError(f"--stop_after_epoch ({end_epoch}) must be greater than the starting epoch ({args.start_epoch})")

    print(f"Start training through epoch {end_epoch - 1} (schedule: {args.epochs} epochs)")
    start_time = time.time()
    for epoch in range(args.start_epoch, end_epoch):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.drop_mask_zero,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == end_epoch):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # New: Print one input image and its unpatched reconstruction per epoch and save them to output_dir
        if global_rank == 0 and args.output_dir:
            model.eval()
            preview_paths = save_epoch_preview(model, dataset_val, device, args, epoch)
            model.train()
            if preview_paths is not None:
                print(
                    "Epoch {} - Saved preview sample {} to: {} | {}".format(
                        epoch,
                        preview_paths["sample_idx"],
                        preview_paths["input"],
                        preview_paths["reconstruction"],
                    )
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def cli(argv=None):
    parser = argparse.ArgumentParser(description='Train the NICOL MAE anomaly model', parents=[get_args_parser()])
    args = parser.parse_args(argv)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


if __name__ == '__main__':
    cli()
