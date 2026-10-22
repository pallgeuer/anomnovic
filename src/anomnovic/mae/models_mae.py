# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted for AnomNOVIC by Philipp Allgeuer and Jan-Gerrit Habekost.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed
import random
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, no_mask=False, anomaly_mask=False,
                 anomaly_reconstruction_weight=5., anomaly_mask_weight=10., nonanomaly_mask_weight=1.0, table_reconstruction_weight=10., table_mask_weight=1.0):
        super().__init__()

        self.patch_size = patch_size
        self.anomaly_reconstruction_weight = anomaly_reconstruction_weight
        self.anomaly_mask_weight = anomaly_mask_weight
        self.nonanomaly_mask_weight = nonanomaly_mask_weight
        self.table_reconstruction_weight = table_reconstruction_weight
        self.table_mask_weight = table_mask_weight

        self.no_mask = no_mask
        self.anomaly_mask = anomaly_mask
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_loss_no_reduction = nn.BCEWithLogitsLoss(reduction='none')

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if self.anomaly_mask:
            if self.no_mask:
                self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * (in_chans + 1), bias=True) # decoder to patch
            else:
                self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * (in_chans + 2), bias=True) # decoder to patch
        else:
            if self.no_mask:
                self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            else:
                self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * (in_chans + 1), bias=True) # decoder to patch
        # --------------------------------------------------------------------------


        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        if self.anomaly_mask:
            out_channels = 4 if self.no_mask else 5
        else:
            out_channels = 3 if self.no_mask else 4

        x = x.reshape(shape=(x.shape[0], h, w, p, p, out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], out_channels, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        mask_ratio = random.uniform(0.0, mask_ratio)  # NOTE: Random mask ratio up to the configured maximum mask ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, table_mask, anomaly_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        pred = self.unpatchify(pred)
        loss = (pred[:,:3] - imgs) ** 2
        if not self.no_mask:
            loss = (loss * table_mask) * self.table_reconstruction_weight
        if self.anomaly_mask:
            loss += loss * (anomaly_mask * self.anomaly_reconstruction_weight)
        loss = loss.mean()  # [N, L], mean loss per patch

        if not self.no_mask:
            mask_shape = table_mask.size()
            loss += self.bce_loss(pred[:,3], torch.reshape(table_mask, shape=(mask_shape[0], mask_shape[2], mask_shape[3]))) * self.table_mask_weight # table mask loss

        if self.anomaly_mask:
            anomaly_mask_shape = anomaly_mask.size()
            anomaly_mask_target = torch.reshape(anomaly_mask, shape=(anomaly_mask_shape[0], anomaly_mask_shape[2], anomaly_mask_shape[3]))
            if self.no_mask:
                anomaly_mask_loss = self.bce_loss_no_reduction(pred[:,3], anomaly_mask_target)  # table mask loss
            else:
                anomaly_mask_loss = self.bce_loss_no_reduction(pred[:,4], anomaly_mask_target)  # table mask loss
            loss += (anomaly_mask_loss * (self.nonanomaly_mask_weight + anomaly_mask_target * self.anomaly_mask_weight)).mean()
        return loss

    def forward(self, imgs, augmented_imgs, mask_ratio=0.75, table_mask=None, anomaly_mask=None):
        latent, mask, ids_restore = self.forward_encoder(augmented_imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, table_mask, anomaly_mask)
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    """Six-block ViT base MAE used by both NICOL checkpoints."""
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
