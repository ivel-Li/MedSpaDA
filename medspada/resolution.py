"""Fixed high-resolution supervision path; no learned upsampling parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class HighResolutionSegmentationHead(nn.Module):
    """Parameter-free 288 -> 504 -> 1008 logit interpolation before loss."""
    def __init__(self, base_head, arm="HR1", *, half_size=504, full_size=1008):
        super().__init__()
        if arm != "HR1" or base_head.no_dec or base_head.aux_masks:
            raise ValueError("HR1 requires the standard final-layer SAM3 mask head")
        self.base_head, self.arm = base_head, arm
        self.half_size, self.full_size = int(half_size), int(full_size)
        self.aux_masks, self.no_dec = base_head.aux_masks, base_head.no_dec
        self.d_model, self.instance_keys = base_head.d_model, ["pred_masks"]


    @property
    def device(self):
        return self.base_head.device

    def _base_outputs(
        self,
        backbone_feats,
        obj_queries: Tensor,
        image_ids,
        encoder_hidden_states: Tensor,
        prompt: Optional[Tensor],
        prompt_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        base = self.base_head
        batch_size = encoder_hidden_states.shape[1]
        if base.cross_attend_prompt is not None:
            attended = base.cross_attn_norm(encoder_hidden_states)
            attended = base.cross_attend_prompt(
                query=attended,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = attended + encoder_hidden_states

        presence_logit = None
        if base.presence_head is not None:
            pooled = encoder_hidden_states.mean(0)
            presence_logit = (
                base.presence_head(
                    pooled.view(1, batch_size, 1, base.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = base._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        instance_embed = base.instance_seg_head(pixel_embed)
        queries = obj_queries[-1]
        low_masks = base.mask_predictor(queries, instance_embed)
        low_semantic = base.semantic_seg_head(pixel_embed)
        return pixel_embed, queries, low_masks, low_semantic, presence_logit

    @staticmethod
    def _resize(logits: Tensor, size: int) -> Tensor:
        dtype = logits.dtype
        return F.interpolate(
            logits.float(), size=(size, size), mode="bilinear", align_corners=False
        ).to(dtype=dtype)

    def forward(
        self,
        backbone_feats,
        obj_queries: Tensor,
        image_ids,
        encoder_hidden_states: Optional[Tensor] = None,
        prompt: Optional[Tensor] = None,
        prompt_mask: Optional[Tensor] = None,
        low_resolution_only: bool = False,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states are required")
        pixel_embed, queries, low_masks, low_semantic, presence_logit = self._base_outputs(
            backbone_feats,
            obj_queries,
            image_ids,
            encoder_hidden_states,
            prompt,
            prompt_mask,
        )
        if low_resolution_only:
            return {
                "pred_masks": low_masks,
                "semantic_seg": low_semantic,
                "presence_logit": presence_logit,
            }

        if self.arm == "HR1":
            masks = self._resize(self._resize(low_masks, self.half_size), self.full_size)
            semantic = self._resize(
                self._resize(low_semantic, self.half_size), self.full_size
            )
            return {
                "pred_masks": masks,
                "semantic_seg": semantic,
                "presence_logit": presence_logit,
            }

@dataclass(frozen=True)
class HighResolutionInstallation:
    arm: str
    module: HighResolutionSegmentationHead | None
    half_size: int
    full_size: int
    topology: str

    def parameters(self):
        if self.module is None:
            return iter(())
        base_ids = {id(parameter) for parameter in self.module.base_head.parameters()}
        return (
            parameter
            for parameter in self.module.parameters()
            if id(parameter) not in base_ids
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

def install_high_resolution_decoder(model, arm, *, half_size=504, full_size=1008):
    if arm == "HR0":
        return HighResolutionInstallation(arm, None, half_size, full_size, "original SAM3 logits")
    if arm != "HR1":
        raise ValueError("Only the fixed HR1 path and original HR0 baseline are supported")
    module = HighResolutionSegmentationHead(model.segmentation_head, arm,
                                             half_size=half_size, full_size=full_size)
    reference = next(model.segmentation_head.parameters())
    model.segmentation_head = module.to(device=reference.device, dtype=reference.dtype)
    return HighResolutionInstallation(arm, module, half_size, full_size,
                                       "bilinear 288 -> 504 -> 1008")
