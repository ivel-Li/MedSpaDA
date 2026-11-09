"""Native SAM3 targets, transforms and detection/segmentation loss."""
from __future__ import annotations

import json
import sys
import time

def load_model(args: argparse.Namespace, torch):
    sys.path.insert(0, str(args.sam3_repo))
    from sam3 import build_sam3_image_model

    print(json.dumps({"stage": "build_model", "time": time.time()}), flush=True)
    model = build_sam3_image_model(
        bpe_path=str(args.sam3_repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device="cuda",
        eval_mode=False,
        checkpoint_path=None,
        load_from_HF=False,
    )
    print(json.dumps({"stage": "load_base", "path": str(args.stage_a_checkpoint)}), flush=True)
    checkpoint = torch.load(args.stage_a_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    official_state = {
        key.removeprefix("detector."): value
        for key, value in state_dict.items()
        if key.startswith("detector.")
    }
    if official_state:
        # The public checkpoint contains detector and tracker namespaces. This
        # is the same extraction performed by SAM3's official image builder.
        incompatible = model.load_state_dict(official_state, strict=False)
        print(
            json.dumps(
                {
                    "stage": "official_image_weights_loaded",
                    "missing_keys": incompatible.missing_keys,
                    "unexpected_keys": incompatible.unexpected_keys,
                }
            ),
            flush=True,
        )
    else:
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"base-checkpoint state mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(json.dumps({"stage": "base_loaded"}), flush=True)
    return model, int(checkpoint.get("epoch", -1)), checkpoint.get("steps", {})


def build_native_find_target(target, torch):
    """Build the official SAM3 one-object-per-semantic-query target batch."""

    from sam3.model.data_misc import BatchedFindTarget

    batch, height, width = target.shape
    target_bool = target > 0.5
    num_boxes = target_bool.flatten(1).any(1).to(dtype=torch.long)
    boxes_padded = target.new_zeros((batch, 1, 4))
    object_ids_padded = torch.full(
        (batch, 1), -1, device=target.device, dtype=torch.long
    )
    boxes = []
    segments = []
    object_ids = []
    for batch_index in range(batch):
        if not bool(num_boxes[batch_index].item()):
            continue
        ys, xs = torch.where(target_bool[batch_index])
        x0 = xs.min().float() / float(width)
        y0 = ys.min().float() / float(height)
        x1 = (xs.max().float() + 1.0) / float(width)
        y1 = (ys.max().float() + 1.0) / float(height)
        box = torch.stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0))
        boxes.append(box)
        boxes_padded[batch_index, 0] = box
        segments.append(target_bool[batch_index])
        object_id = len(object_ids)
        object_ids.append(object_id)
        object_ids_padded[batch_index, 0] = object_id
    if boxes:
        packed_boxes = torch.stack(boxes)
        packed_segments = torch.stack(segments)
        packed_object_ids = torch.tensor(
            object_ids, device=target.device, dtype=torch.long
        )
    else:
        packed_boxes = target.new_zeros((0, 4))
        packed_segments = torch.zeros(
            (0, height, width), device=target.device, dtype=torch.bool
        )
        packed_object_ids = torch.zeros(0, device=target.device, dtype=torch.long)
    return BatchedFindTarget(
        num_boxes=num_boxes,
        boxes=packed_boxes,
        boxes_padded=boxes_padded,
        repeated_boxes=packed_boxes,
        segments=packed_segments,
        semantic_segments=target_bool,
        is_valid_segment=torch.ones(
            len(packed_boxes), device=target.device, dtype=torch.bool
        ),
        is_exhaustive=torch.ones(batch, device=target.device, dtype=torch.bool),
        object_ids=packed_object_ids,
        object_ids_padded=object_ids_padded,
    )


def transform_native_window(
    images,
    target_np,
    *,
    resolution: int,
    min_resolution: int,
    rng,
    torch,
    v2,
):
    """Apply Stage-A resize/pad jointly to every slice in a 3D window."""

    scales = list(range(min_resolution, resolution + 1, 32))
    if not scales:
        raise ValueError("native resize range is empty")
    scale = rng.choice(scales)
    pad_left = rng.randint(0, resolution - scale)
    pad_top = rng.randint(0, resolution - scale)
    padding = [
        pad_left,
        pad_top,
        resolution - scale - pad_left,
        resolution - scale - pad_top,
    ]
    transformed_images = []
    for image in images:
        tensor = v2.functional.to_image(image).to("cuda")
        tensor = v2.functional.resize(
            tensor,
            [scale, scale],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = v2.functional.pad(tensor, padding, fill=0)
        tensor = v2.functional.to_dtype(tensor, torch.float32, scale=True)
        tensor = v2.functional.normalize(
            tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        transformed_images.append(tensor)
    target = torch.from_numpy(target_np).to("cuda", dtype=torch.float32)
    target = torch.nn.functional.interpolate(
        target[:, None], size=(scale, scale), mode="nearest"
    ).squeeze(1)
    target = torch.nn.functional.pad(
        target,
        (padding[0], padding[2], padding[1], padding[3]),
    )
    return torch.stack(transformed_images), target, {
        "train_resize": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }


def build_native_stage_a_loss(torch, *, o2m_topk: int):
    """Instantiate the promoted Stage-A objective with a declared top-k."""

    from sam3.train.loss.loss_fns import (
        Boxes,
        IABCEMdetr,
        Masks,
        SemanticSegCriterion,
    )
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )
    return Sam3LossWrapper(
        matcher=matcher,
        o2m_matcher=BinaryOneToManyMatcher(
            alpha=0.3, threshold=0.4, topk=o2m_topk
        ),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        normalization="local",
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_weight=10.0,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pos_focal=False,
                pad_n_queries=200,
                pad_scale_pos=1.0,
            ),
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                compute_aux=False,
            ),
        ],
        loss_fn_semantic_seg=SemanticSegCriterion(
            presence_head=True,
            presence_loss=False,
            focal=True,
            focal_alpha=0.6,
            focal_gamma=2.0,
            downsample=False,
            weight_dict={
                "loss_semantic_seg": 20.0,
                "loss_semantic_presence": 1.0,
                "loss_semantic_dice": 30.0,
            },
        ),
        scale_by_find_batch_size=True,
    ).to("cuda")


def native_stage_a_loss(output, find_target, criterion, model, torch):
    """Compute the exact Stage-A detection, presence, mask, semantic and O2M loss."""

    from sam3.train.loss.loss_fns import CORE_LOSS_KEY

    targets = model.back_convert(find_target)
    losses = criterion.compute_loss(output, targets)
    semantic_losses = criterion.loss_fn_semantic_seg(output, targets)
    total = losses[CORE_LOSS_KEY] + semantic_losses[CORE_LOSS_KEY]
    losses = {**losses, **semantic_losses}
    metrics = {
        f"native_{key}": float(value.detach())
        for key, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    metrics["total"] = float(total.detach())
    metrics["base_total"] = float(total.detach())
    return total, metrics
