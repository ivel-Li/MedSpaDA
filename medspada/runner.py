"""Patient-local context execution with identity query passthrough."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from .query_identity import PatientQueryFusion, choose_global_gland_anchor


@dataclass(frozen=True)
class PatientExecutionPolicy:
    slice_microbatch: int
    encoder_halo: int
    depth_residual_enabled: bool


def _patient_execution_policy(
    depth: int,
    *,
    slice_microbatch: int,
    encoder_halo: int,
    singleton_halo: int = 1,
) -> PatientExecutionPolicy:
    """Choose a dataset-agnostic 2D/3D execution policy.

    A singleton image stays in the shared pipeline, but it is not padded to a
    fake multi-slice core and does not use Mona's slice-axis residual.  One
    replicated halo image is retained on each side so the boundary convention
    remains explicit and identical at training and inference.
    """

    if depth < 1:
        raise ValueError("depth must be positive")
    if depth == 1:
        return PatientExecutionPolicy(1, int(singleton_halo), False)
    return PatientExecutionPolicy(
        int(slice_microbatch), int(encoder_halo), True
    )


def _set_depth_residual_enabled(model: nn.Module, enabled: bool) -> int:
    """Set the runtime Z-residual mode on compatible visual adapters."""

    updated = 0
    for module in model.modules():
        setter = getattr(module, "set_depth_residual_enabled", None)
        if callable(setter):
            setter(enabled)
            updated += 1
    return updated


def _select_backbone_batch(value: Any, indices: Tensor, source_depth: int) -> Any:
    """Select image rows recursively from a SAM3 backbone output.

    SAM3 stores feature maps both directly and inside NestedTensor objects.
    Selection happens after the checkpointed vision forward, so recomputation
    receives only tensor inputs and cannot observe stale Python chunk state.
    """

    try:
        from sam3.model.data_misc import NestedTensor
    except ImportError:
        # Older bundled SAM3 releases (including MedSAM3 v1) return plain
        # tensors here and do not define this container type.
        NestedTensor = ()

    if isinstance(value, Tensor):
        return (
            value.index_select(0, indices)
            if value.ndim > 0 and value.shape[0] == source_depth
            else value
        )
    if isinstance(value, NestedTensor):
        tensors = value.tensors.index_select(0, indices)
        mask = value.mask
        if mask is not None and mask.ndim > 0 and mask.shape[0] == source_depth:
            mask = mask.index_select(0, indices)
        return NestedTensor(tensors, mask)
    if isinstance(value, list):
        return [_select_backbone_batch(item, indices, source_depth) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _select_backbone_batch(item, indices, source_depth) for item in value
        )
    if isinstance(value, dict):
        return {
            key: _select_backbone_batch(item, indices, source_depth)
            for key, item in value.items()
        }
    return value


@dataclass
class PatientEncoding:
    depth: int
    hs: Tensor
    reference_boxes: Tensor
    decoder_presence: Tensor | None
    presence_features: Tensor | None
    query_scores: Tensor
    best_query_indices: Tensor
    union_area: Tensor
    slice_confidence: Tensor
    anchor_index: int
    anchor_confidence: float
    anchor_confident: bool
    head_chunks: list["PatientHeadChunk | None"]
    num_categories: int = 2


@dataclass
class PatientHeadChunk:
    start: int
    end: int
    backbone_out: dict[str, Any]
    encoder_out: dict[str, Any]
    prompt: Tensor
    prompt_mask: Tensor
    find_input: Any


def _find_input(
    depth: int, device: torch.device, *, num_categories: int = 2
):
    from sam3.model.data_misc import FindStage

    if depth < 1 or num_categories < 1:
        raise ValueError("depth and num_categories must be positive")
    return FindStage(
        img_ids=torch.arange(depth, device=device, dtype=torch.long).repeat(num_categories),
        text_ids=torch.arange(
            num_categories, device=device, dtype=torch.long
        ).repeat_interleave(depth),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def _merge_category_major(
    chunks: list[tuple[Tensor, int]], batch_dim: int, *, num_categories: int = 2
) -> Tensor:
    """Merge category-major chunks into one full-volume category-major tensor."""

    if num_categories < 1:
        raise ValueError("num_categories must be positive")
    category_parts: list[Tensor] = []
    for category in range(num_categories):
        pieces = [
            tensor.narrow(batch_dim, category * depth, depth)
            for tensor, depth in chunks
        ]
        category_parts.append(torch.cat(pieces, dim=batch_dim))
    return torch.cat(category_parts, dim=batch_dim)


def _patient_slice(
    tensor: Tensor,
    depth: int,
    start: int,
    end: int,
    batch_dim: int,
    *,
    num_categories: int = 2,
) -> Tensor:
    """Take a category-major slice interval and preserve category-major order."""

    pieces = [
        tensor.narrow(batch_dim, category * depth + start, end - start)
        for category in range(num_categories)
    ]
    return torch.cat(pieces, dim=batch_dim)


def _trim_category_major(
    tensor: Tensor,
    padded_depth: int,
    valid_depth: int,
    batch_dim: int,
    *,
    num_categories: int = 2,
) -> Tensor:
    """Remove repeat-padding from a category-major prompt/image batch."""

    if valid_depth == padded_depth:
        return tensor
    return torch.cat(
        [
            tensor.narrow(batch_dim, category * padded_depth, valid_depth)
            for category in range(num_categories)
        ],
        dim=batch_dim,
    )


class Sam3PatientVolumeRunner:
    """Split SAM3 around its final decoder queries for full-patient fusion."""

    def __init__(
        self,
        model: nn.Module,
        fusion: PatientQueryFusion,
        *,
        slice_microbatch: int = 4,
        head_slice_microbatch: int | None = None,
        anchor_confidence_threshold: float = 0.60,
        encoder_halo: int = 0,
        singleton_halo: int = 1,
    ):
        if slice_microbatch < 1:
            raise ValueError("slice_microbatch must be positive")
        if head_slice_microbatch is not None and head_slice_microbatch < 1:
            raise ValueError("head_slice_microbatch must be positive")
        if encoder_halo < 0:
            raise ValueError("encoder_halo cannot be negative")
        if singleton_halo < 0:
            raise ValueError("singleton_halo cannot be negative")
        self.model = model
        self.fusion = fusion
        self.slice_microbatch = int(slice_microbatch)
        self.head_slice_microbatch = int(head_slice_microbatch or slice_microbatch)
        self.anchor_confidence_threshold = float(anchor_confidence_threshold)
        self.encoder_halo = int(encoder_halo)
        self.singleton_halo = int(singleton_halo)
        self.last_execution_policy: PatientExecutionPolicy | None = None
        self.num_o2o = int(model.transformer.decoder.num_queries)

    def _decode_queries(self, memory, pos_embed, src_mask, prompt, prompt_mask, encoder_out):
        image_batch = int(memory.shape[1])
        query_embed = self.model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, image_batch, 1)
        apply_dac = self.model.transformer.decoder.dac and self.model.training
        hs, reference_boxes, presence, presence_features = self.model.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if presence is not None:
            presence = presence.transpose(1, 2)
        return hs, reference_boxes, presence, presence_features

    def _coarse_statistics(
        self,
        hs: Tensor,
        decoder_presence: Tensor | None,
        backbone_out: dict[str, Any],
        encoder_out: dict[str, Any],
        find_input,
        prompt: Tensor,
        prompt_mask: Tensor,
        depth: int,
        num_categories: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prediction-only anchor evidence from the unmodified 2D path."""

        with torch.no_grad():
            detached_hs = hs.detach()
            class_layers = self.model.dot_prod_scoring(
                detached_hs, prompt.detach(), prompt_mask
            )
            class_logits = class_layers[-1, :, : self.num_o2o].squeeze(-1)
            if decoder_presence is None:
                presence = torch.zeros(
                    (class_logits.shape[0], 1),
                    device=class_logits.device,
                    dtype=class_logits.dtype,
                )
            else:
                presence = decoder_presence[-1].detach()
                if presence.ndim == 1:
                    presence = presence[:, None]
            scores = class_logits.sigmoid() * presence.sigmoid()
            base_masks = self.model.segmentation_head(
                backbone_feats=[value.detach() for value in backbone_out["backbone_fpn"]],
                obj_queries=detached_hs[:, :, : self.num_o2o],
                image_ids=find_input.img_ids,
                encoder_hidden_states=encoder_out["encoder_hidden_states"].detach(),
                prompt=prompt.detach(),
                prompt_mask=prompt_mask,
                low_resolution_only=True,
            )["pred_masks"]
            best = scores.argmax(dim=1)
            row = torch.arange(num_categories * depth, device=scores.device)
            selected = base_masks[row, best].reshape(
                num_categories, depth, *base_masks.shape[-2:]
            )
            union_area = torch.any(selected > 0.0, dim=0).flatten(1).sum(1)
            score_volume = scores.reshape(
                num_categories, depth, self.num_o2o
            )
            slice_confidence = score_volume.amax(dim=(0, 2))
            return (
                score_volume.detach(),
                best.reshape(num_categories, depth).detach(),
                union_area.detach(),
                slice_confidence.detach(),
            )

    def encode_patient(
        self,
        image_batch: Tensor,
        prompts: list[str],
    ) -> PatientEncoding:
        """Run chunked query extraction and restore full-patient order.

        With encoder_halo > 0, each fixed-size core chunk is surrounded by
        replicated-boundary context slices.  The image encoder sees the halo,
        but its backbone outputs are immediately cropped back to the core.
        Therefore overlapping context is never decoded, supervised, or emitted
        as a duplicate prediction.
        """

        if image_batch.ndim != 4 or image_batch.shape[0] < 1:
            raise ValueError("image_batch must have shape [D,C,H,W]")
        if not prompts:
            raise ValueError("at least one category prompt is required")
        depth = int(image_batch.shape[0])
        num_categories = len(prompts)
        device = image_batch.device
        policy = _patient_execution_policy(
            depth,
            slice_microbatch=self.slice_microbatch,
            encoder_halo=self.encoder_halo,
            singleton_halo=self.singleton_halo,
        )
        self.last_execution_policy = policy
        _set_depth_residual_enabled(
            self.model, policy.depth_residual_enabled
        )
        text_output = self.model.backbone.forward_text(prompts, device=device)
        hs_chunks: list[tuple[Tensor, int]] = []
        reference_chunks: list[tuple[Tensor, int]] = []
        presence_chunks: list[tuple[Tensor, int]] = []
        presence_feature_chunks: list[tuple[Tensor, int]] = []
        score_chunks: list[Tensor] = []
        best_chunks: list[Tensor] = []
        area_chunks: list[Tensor] = []
        confidence_chunks: list[Tensor] = []
        head_chunks: list[PatientHeadChunk | None] = []

        for start in range(0, depth, policy.slice_microbatch):
            end = min(start + policy.slice_microbatch, depth)
            chunk_depth = end - start
            padded_depth = policy.slice_microbatch
            core_indices = torch.arange(
                start,
                start + padded_depth,
                device=device,
                dtype=torch.long,
            ).clamp_(0, depth - 1)
            if policy.encoder_halo:
                context_indices = torch.arange(
                    start - policy.encoder_halo,
                    start + padded_depth + policy.encoder_halo,
                    device=device,
                    dtype=torch.long,
                ).clamp_(0, depth - 1)
                images = image_batch.index_select(0, context_indices)
            else:
                images = image_batch.index_select(0, core_indices)
            backbone_out = self.model.backbone.forward_image(images)
            if policy.encoder_halo:
                context_depth = int(images.shape[0])
                keep = torch.arange(
                    policy.encoder_halo,
                    policy.encoder_halo + padded_depth,
                    device=device,
                    dtype=torch.long,
                )
                backbone_out = _select_backbone_batch(
                    backbone_out, keep, context_depth
                )
            backbone_out.update(text_output)
            padded_find_input = _find_input(
                padded_depth, device, num_categories=num_categories
            )
            geometry = self.model._get_dummy_prompt(
                num_prompts=num_categories * padded_depth
            )
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out, padded_find_input, geometry
            )
            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out, padded_find_input, prompt, prompt_mask
            )
            hs, references, presence, presence_features = self._decode_queries(
                encoder_out["encoder_hidden_states"],
                encoder_out["pos_embed"],
                encoder_out["padding_mask"],
                prompt,
                prompt_mask,
                encoder_out,
            )
            hs = _trim_category_major(
                hs,
                padded_depth,
                chunk_depth,
                batch_dim=1,
                num_categories=num_categories,
            )
            references = _trim_category_major(
                references,
                padded_depth,
                chunk_depth,
                batch_dim=1,
                num_categories=num_categories,
            )
            if presence is not None:
                presence = _trim_category_major(
                    presence,
                    padded_depth,
                    chunk_depth,
                    batch_dim=1,
                    num_categories=num_categories,
                )
            if presence_features is not None:
                presence_features = _trim_category_major(
                    presence_features,
                    padded_depth,
                    chunk_depth,
                    batch_dim=1,
                    num_categories=num_categories,
                )
            prompt = _trim_category_major(
                prompt,
                padded_depth,
                chunk_depth,
                batch_dim=1,
                num_categories=num_categories,
            )
            prompt_mask = _trim_category_major(
                prompt_mask,
                padded_depth,
                chunk_depth,
                batch_dim=0,
                num_categories=num_categories,
            )
            encoder_out = dict(encoder_out)
            encoder_out["encoder_hidden_states"] = _trim_category_major(
                encoder_out["encoder_hidden_states"],
                padded_depth,
                chunk_depth,
                batch_dim=1,
                num_categories=num_categories,
            )
            find_input = _find_input(
                chunk_depth, device, num_categories=num_categories
            )
            scores, best, area, confidence = self._coarse_statistics(
                hs,
                presence,
                backbone_out,
                encoder_out,
                find_input,
                prompt,
                prompt_mask,
                chunk_depth,
                num_categories,
            )
            hs_chunks.append((hs, chunk_depth))
            reference_chunks.append((references, chunk_depth))
            if presence is not None:
                presence_chunks.append((presence, chunk_depth))
            if presence_features is not None:
                presence_feature_chunks.append((presence_features, chunk_depth))
            score_chunks.append(scores)
            best_chunks.append(best)
            area_chunks.append(area)
            confidence_chunks.append(confidence)
            head_chunks.append(
                PatientHeadChunk(
                    start=start,
                    end=end,
                    backbone_out=backbone_out,
                    encoder_out=encoder_out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    find_input=find_input,
                )
            )

        hs = _merge_category_major(
            hs_chunks, batch_dim=1, num_categories=num_categories
        )
        references = _merge_category_major(
            reference_chunks, batch_dim=1, num_categories=num_categories
        )
        decoder_presence = (
            _merge_category_major(
                presence_chunks, batch_dim=1, num_categories=num_categories
            )
            if presence_chunks
            else None
        )
        presence_features = (
            _merge_category_major(
                presence_feature_chunks,
                batch_dim=1,
                num_categories=num_categories,
            )
            if presence_feature_chunks
            else None
        )
        scores = torch.cat(score_chunks, dim=1)
        best = torch.cat(best_chunks, dim=1)
        union_area = torch.cat(area_chunks)
        slice_confidence = torch.cat(confidence_chunks)
        anchor, anchor_confidence, anchor_confident = choose_global_gland_anchor(
            union_area, slice_confidence, self.anchor_confidence_threshold
        )
        return PatientEncoding(
            depth=depth,
            hs=hs,
            reference_boxes=references,
            decoder_presence=decoder_presence,
            presence_features=presence_features,
            query_scores=scores,
            best_query_indices=best,
            union_area=union_area,
            slice_confidence=slice_confidence,
            anchor_index=anchor,
            anchor_confidence=anchor_confidence,
            anchor_confident=anchor_confident,
            head_chunks=head_chunks,
            num_categories=num_categories,
        )

    def fuse_patient(
        self, encoding: PatientEncoding, z_mm: Tensor
    ) -> tuple[Tensor, dict[str, Tensor | str | int | bool]]:
        categories = encoding.num_categories
        final_queries = encoding.hs[-1].reshape(
            categories,
            encoding.depth,
            encoding.hs.shape[2],
            encoding.hs.shape[3],
        )[:, :, : self.num_o2o]
        anchor_queries = encoding.best_query_indices[:, encoding.anchor_index]
        return self.fusion(
            final_queries,
            z_mm,
            query_scores=encoding.query_scores,
            anchor_index=encoding.anchor_index,
            anchor_query_indices=anchor_queries,
            anchor_enabled=encoding.anchor_confident,
        )

    def iter_final_outputs(
        self,
        encoding: PatientEncoding,
        fused_queries: Tensor,
    ) -> Iterator[tuple[int, int, dict[str, Any]]]:
        """Decode cached N-slice encoder features in smaller final-head chunks.

        ``slice_microbatch`` still controls the image encoder and therefore the
        3D-adapter context. ``head_slice_microbatch`` only limits the number of
        high-resolution detector masks materialized at once.
        """

        depth = encoding.depth
        categories = encoding.num_categories
        if fused_queries.shape[:3] != (categories, depth, self.num_o2o):
            raise ValueError("fused query layout mismatch")
        query_count = int(encoding.hs.shape[2])

        for chunk_index, cached in enumerate(encoding.head_chunks):
            if cached is None:
                raise RuntimeError("patient head chunk was already consumed")
            cached_start, cached_end = cached.start, cached.end
            cached_depth = cached_end - cached_start

            for start in range(cached_start, cached_end, self.head_slice_microbatch):
                end = min(start + self.head_slice_microbatch, cached_end)
                chunk_depth = end - start
                relative_start = start - cached_start
                relative_end = end - cached_start
                image_indices = torch.arange(
                    relative_start,
                    relative_end,
                    device=encoding.hs.device,
                    dtype=torch.long,
                )
                backbone_out = _select_backbone_batch(
                    cached.backbone_out, image_indices, cached_depth
                )
                prompt = _patient_slice(
                    cached.prompt,
                    cached_depth,
                    relative_start,
                    relative_end,
                    batch_dim=1,
                    num_categories=categories,
                )
                prompt_mask = _patient_slice(
                    cached.prompt_mask,
                    cached_depth,
                    relative_start,
                    relative_end,
                    batch_dim=0,
                    num_categories=categories,
                )
                encoder_hidden_states = _patient_slice(
                    cached.encoder_out["encoder_hidden_states"],
                    cached_depth,
                    relative_start,
                    relative_end,
                    batch_dim=1,
                    num_categories=categories,
                )
                find_input = _find_input(
                    chunk_depth,
                    encoding.hs.device,
                    num_categories=categories,
                )

                local_hs = _patient_slice(
                    encoding.hs,
                    depth,
                    start,
                    end,
                    batch_dim=1,
                    num_categories=categories,
                )
                local_fused = torch.cat(
                    [
                        fused_queries[:, start:end].reshape(
                            categories * chunk_depth, self.num_o2o, -1
                        ),
                        local_hs[-1, :, self.num_o2o : query_count],
                    ],
                    dim=1,
                )
                local_hs = torch.cat(
                    [local_hs[:-1], local_fused.unsqueeze(0)], dim=0
                )
                local_references = _patient_slice(
                    encoding.reference_boxes,
                    depth,
                    start,
                    end,
                    batch_dim=1,
                    num_categories=categories,
                )
                local_presence = (
                    _patient_slice(
                        encoding.decoder_presence,
                        depth,
                        start,
                        end,
                        batch_dim=1,
                        num_categories=categories,
                    )
                    if encoding.decoder_presence is not None
                    else None
                )
                local_presence_features = (
                    _patient_slice(
                        encoding.presence_features,
                        depth,
                        start,
                        end,
                        batch_dim=1,
                        num_categories=categories,
                    )
                    if encoding.presence_features is not None
                    else None
                )
                out: dict[str, Any] = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "presence_feats": local_presence_features,
                }
                self.model._update_scores_and_boxes(
                    out,
                    local_hs,
                    local_references,
                    prompt,
                    prompt_mask,
                    dec_presence_out=local_presence,
                )
                self.model._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=cached.encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=encoder_hidden_states,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=local_hs,
                )
                yield start, end, out
            encoding.head_chunks[chunk_index] = None
