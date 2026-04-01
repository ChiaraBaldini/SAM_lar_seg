# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class LearnableBoxGenerator(nn.Module):
    """Generates pseudo bounding boxes from image embeddings using gradients"""
    
    def __init__(self, embed_dim=256):
        super().__init__()
        # Simple network to process embeddings
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),  # Reduce from 64x64 to 16x16
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)  # Scalar output to create gradients
        )
        
    def extract_boxes_from_gradients(self, image_embeddings):
        """Extract bounding boxes using gradient-based saliency"""
        B, C, H, W = image_embeddings.shape
        boxes = []
        
        for i in range(B):
            # Process single sample
            single_embed = image_embeddings[i:i+1]  # (1, 256, 64, 64)
            single_embed.requires_grad_(True)
            
            # Forward pass to obtain gradients
            dummy_output = self.feature_processor(single_embed)  # (1, 1)
            
            # Backward pass to obtain gradients
            if single_embed.grad is not None:
                single_embed.grad.zero_()
            
            dummy_output.backward(retain_graph=True)
            grad = single_embed.grad  # (1, 256, 64, 64)
            
            # Saliency map from gradients
            saliency = torch.norm(grad.squeeze(0), dim=0)  # (64, 64)
            
            # Adaptive threshold
            threshold = torch.quantile(saliency.flatten(), 0.75)  # Top 25%
            sal_mask = saliency > threshold
            
            # Extract bounding box
            coords = torch.nonzero(sal_mask, as_tuple=False)  # (N, 2)
            
            if coords.shape[0] > 0:
                y_min, x_min = coords.min(dim=0)[0]
                y_max, x_max = coords.max(dim=0)[0]
                
                # Add padding to avoid too narrow boxes
                padding = max(2, (y_max - y_min) * 0.1, (x_max - x_min) * 0.1)
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding)
                y_max = min(H-1, y_max + padding)
                x_max = min(W-1, x_max + padding)
                
                # Scale from 64x64 to 1024x1024
                scale = 1024 / 64
                box = torch.tensor([
                    x_min * scale, y_min * scale,
                    x_max * scale, y_max * scale
                ], device=image_embeddings.device, dtype=torch.float32)
                
                boxes.append(box)
            else:
                # Fallback box (center of image)
                boxes.append(torch.tensor([256, 256, 768, 768], 
                           device=image_embeddings.device, dtype=torch.float32))
        
        return torch.stack(boxes)  # (B, 4)


class SamBB(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # ✅ NEW: Add learnable box generator
        self.box_generator = LearnableBoxGenerator(embed_dim=image_encoder.embed_dim)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        """Training forward pass with pseudo-boxes from gradients"""
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)
        
        # ✅ NEW: Extract pseudo-boxes from image embeddings using gradients
        with torch.enable_grad():  # Ensure gradients are enabled
            pseudo_boxes = self.box_generator.extract_boxes_from_gradients(image_embeddings)
        
        # Debug info
        ic(f"Generated pseudo boxes shape: {pseudo_boxes.shape}")
        ic(f"Pseudo boxes range: {pseudo_boxes.min().item():.2f} - {pseudo_boxes.max().item():.2f}")
        
        # Normalize boxes for prompt encoder (SAM expects [0, 1])
        pseudo_boxes_normalized = pseudo_boxes / 1024.0
        
        # Convert to correct format for prompt encoder
        # prompt_encoder expects (B, 1, 4) for boxes
        pseudo_boxes_formatted = pseudo_boxes_normalized.unsqueeze(1)  # (B, 1, 4)
        
        ic(f"Formatted boxes shape: {pseudo_boxes_formatted.shape}")
        ic(f"Normalized boxes range: {pseudo_boxes_formatted.min().item():.3f} - {pseudo_boxes_formatted.max().item():.3f}")
        
        # ✅ MODIFIED: Use pseudo-boxes instead of None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, 
            boxes=pseudo_boxes_formatted,  # ✅ Use pseudo-boxes!
            masks=None
        )
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks,
            'pseudo_boxes': pseudo_boxes  # ✅ NEW: Also return generated boxes
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

