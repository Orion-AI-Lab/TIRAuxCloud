# -*- coding: utf-8 -*-
# @Time    : 2024/8/8 下午3:41
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : sam.py
# @Software: PyCharm

from typing import Literal
from segment_anything import sam_model_registry
from segment_anything.modeling.mask_decoder import MaskDecoder, MLP
import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F


class SAM(nn.Module):
    def __init__(
        self, model_type="vit_h", num_classes=6, checkpoint="data/sam_check_point/sam_vit_h_4b8939.pth"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.get_model(model_type=model_type, checkpoint=checkpoint)

    def get_model(self, model_type, checkpoint=None):
        sam_model = sam_model_registry[model_type](checkpoint)

        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder

        mask_decoder = sam_model.mask_decoder

        self.mask_decoder = MaskDecoder(
            transformer_dim=mask_decoder.transformer_dim,
            transformer=mask_decoder.transformer,
            num_multimask_outputs=self.num_classes,
        )

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False

        # for name, param in self.mask_decoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, images: torch.Tensor):
        B, _, H, W = images.shape

        images = F.interpolate(
            images,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        image_embeddings = self.image_encoder(images)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_preds = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        logits = F.interpolate(
            low_res_masks,
            (H, W),
            mode="bilinear",
            align_corners=False,
        )

        return logits, iou_preds


if __name__ == "__main__":
    device = "cuda:7"
    model = SAM().to(device)
    fake_data = torch.randn(2, 3, 512, 512).to(device)
    logits,iou_preds = model(fake_data)
    print(logits.shape)
    print(logits.dtype)
    print(iou_preds.shape)
    # sigmoid
    print(torch.max(logits), torch.min(logits))
