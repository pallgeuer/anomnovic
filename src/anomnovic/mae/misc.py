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

import builtins
import datetime
import os
import time
import argparse
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):

    if not args.resume:
        return

    with torch.serialization.safe_globals([argparse.Namespace]):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True, weights_only=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)

    checkpoint_model = checkpoint['model']        # Dict with dot-joined weight name strings as keys (loaded weights)
    model_state = model_without_ddp.state_dict()  # Dict with dot-joined weight name strings as keys (existing weights)

    partially_initialized_keys = set()
    if (patch_size := getattr(model_without_ddp, 'patch_size', None)) is not None:
        patch_area = patch_size * patch_size
        if (
            (ckpt_w := checkpoint_model.get('decoder_pred.weight', None)) is not None and (model_w := model_state.get('decoder_pred.weight', None)) is not None and
            (ckpt_b := checkpoint_model.get('decoder_pred.bias', None)) is not None and (model_b := model_state.get('decoder_pred.bias', None)) is not None and
            ckpt_w.ndim == model_w.ndim == 2 and ckpt_b.ndim == model_b.ndim == 1 and ckpt_w.shape[0] == ckpt_b.shape[0] and model_w.shape[0] == model_b.shape[0] and ckpt_w.shape[1] == model_w.shape[1] and
            ckpt_w.shape[0] % patch_area == 0 and model_w.shape[0] % patch_area == 0 and ckpt_w.shape[0] < model_w.shape[0]
        ):
            ckpt_channels = ckpt_w.shape[0] // patch_area
            model_channels = model_w.shape[0] // patch_area
            new_ckpt_w = model_w.view(patch_area, model_channels, model_w.shape[1]).clone()
            new_ckpt_b = model_b.view(patch_area, model_channels).clone()
            new_ckpt_w[:, :ckpt_channels, :] = ckpt_w.view(patch_area, ckpt_channels, ckpt_w.shape[1])
            new_ckpt_b[:, :ckpt_channels] = ckpt_b.view(patch_area, ckpt_channels)
            checkpoint_model['decoder_pred.weight'] = new_ckpt_w.reshape_as(model_w)
            checkpoint_model['decoder_pred.bias'] = new_ckpt_b.reshape_as(model_b)
            partially_initialized_keys.add('decoder_pred.weight')
            partially_initialized_keys.add('decoder_pred.bias')
            print(f"[WARN] Partially loading 'decoder_pred.weight + bias' in channel space: Checkpoint {tuple(ckpt_w.shape)} + {tuple(ckpt_b.shape)} ({ckpt_channels} ch) -> Model {tuple(model_w.shape)} + {tuple(model_b.shape)} ({model_channels} ch) (old channels copied, new channels left random)")

    if args.resume_nonstrict:
        if 'pos_embed' in checkpoint_model and hasattr(model_without_ddp, 'pos_embed') and checkpoint_model['pos_embed'].shape[1] != model_without_ddp.pos_embed.shape[1]:
            print("[WARN] Ignoring checkpoint 'pos_embed' as it is the wrong shape")
            del checkpoint_model['pos_embed']
        if 'decoder_pos_embed' in checkpoint_model and hasattr(model_without_ddp, 'decoder_pos_embed') and checkpoint_model['decoder_pos_embed'].shape[1] != model_without_ddp.decoder_pos_embed.shape[1]:
            print("[WARN] Ignoring checkpoint 'decoder_pos_embed' as it is the wrong shape")
            del checkpoint_model['decoder_pos_embed']

    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint_model, strict=not args.resume_nonstrict)
    if missing_keys:
        missing_keys_str = ''.join(f'\n - {key}' for key in sorted(missing_keys))
        print(f"[WARN] WEIGHTS NOT FOUND IN THE RESUMED STATE DICT:{missing_keys_str}")
    if unexpected_keys:
        unexpected_keys_str = ''.join(f'\n - {key}' for key in sorted(unexpected_keys))
        print(f"[WARN] UNEXPECTED UNUSED EXTRA KEYS FOUND IN THE RESUMED STATE DICT:{unexpected_keys_str}")

    if args.resume_frozen:
        frozen_weights = []
        trainable_weights = []
        for name, param in model_without_ddp.named_parameters():
            if name not in missing_keys and name not in partially_initialized_keys:
                param.requires_grad = False
            if param.requires_grad:
                trainable_weights.append(name)
            else:
                frozen_weights.append(name)
        frozen_weights_str = ''.join(f'\n - {name}' for name in sorted(frozen_weights))
        print(f"[WARN] FROZEN WEIGHTS (WITH --resume_frozen):{frozen_weights_str}")
        trainable_weights_str = ''.join(f'\n - {name}' for name in sorted(trainable_weights))
        print(f"[WARN] TRAINABLE WEIGHTS (WITH --resume_frozen):{trainable_weights_str}")
        if not trainable_weights:
            raise RuntimeError("Nothing to train as everything is frozen!")

    if args.freeze_encoder:
        frozen_weights = []
        trainable_weights = []
        for name, param in model_without_ddp.named_parameters():
            if (
                    name == 'cls_token'
                    or name == 'pos_embed'
                    or name.startswith('patch_embed.')
                    or name.startswith('blocks.')
                    or name.startswith('norm.')
            ):
                param.requires_grad = False
            if param.requires_grad:
                trainable_weights.append(name)
            else:
                frozen_weights.append(name)
        frozen_weights_str = ''.join(f'\n - {name}' for name in sorted(frozen_weights))
        print(f"[WARN] FROZEN WEIGHTS (WITH --freeze_encoder):{frozen_weights_str}")
        trainable_weights_str = ''.join(f'\n - {name}' for name in sorted(trainable_weights))
        print(f"[WARN] TRAINABLE WEIGHTS (WITH --freeze_encoder):{trainable_weights_str}")
        if not trainable_weights:
            raise RuntimeError("Nothing to train as everything is frozen!")

    print("Resume checkpoint %s" % args.resume)

    if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval) and not args.resume_weights_only:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
