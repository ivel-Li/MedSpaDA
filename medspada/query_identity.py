from __future__ import annotations

import torch
from torch import Tensor, nn

def choose_global_gland_anchor(
    union_area: Tensor,
    slice_confidence: Tensor,
    confidence_threshold: float,
) -> tuple[int, float, bool]:
    """Select one deployable full-volume anchor without using ground truth."""

    if union_area.ndim != 1 or slice_confidence.shape != union_area.shape:
        raise ValueError("anchor area and confidence must both have shape [D]")
    if union_area.numel() == 0:
        raise ValueError("cannot select an anchor from an empty volume")
    eligible = slice_confidence >= float(confidence_threshold)
    if bool(eligible.any().item()):
        anchor = int(union_area.masked_fill(~eligible, -1).argmax().item())
        confident = True
    else:
        anchor = int(slice_confidence.argmax().item())
        confident = False
    return anchor, float(slice_confidence[anchor].item()), confident

class PatientQueryFusion(nn.Module):
    """Identity query path. Three-dimensional adaptation lives in the encoder."""
    def __init__(self, arm, dim):
        super().__init__()
        if arm != "F0":
            raise ValueError("Learned query-memory variants are not part of this release")
        self.arm, self.dim = arm, dim

    def forward(self, queries, z_mm, **kwargs):
        return queries, {"arm": "F0"}

def install_patient_query_fusion(model: nn.Module, arm: str, **kwargs) -> PatientQueryFusion:
    """Register one patient-query fusion module on a SAM3 image model."""

    if hasattr(model, "prostate_patient_fusion"):
        raise RuntimeError("patient query fusion is already installed")
    dim = int(model.transformer.decoder.query_embed.weight.shape[-1])
    module = PatientQueryFusion(arm, dim, **kwargs)
    weight = model.transformer.decoder.query_embed.weight
    module.to(device=weight.device, dtype=weight.dtype)
    model.add_module("prostate_patient_fusion", module)
    return module
