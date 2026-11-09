from __future__ import annotations

from collections import defaultdict
from typing import Any

def _optimizer(model, fusion, encoder_3d, vision_fact, high_resolution, args, torch):
    fusion_ids = {id(parameter) for parameter in fusion.parameters()}
    encoder_3d_ids = {id(parameter) for parameter in encoder_3d.parameters()}
    fact_ids = {id(parameter) for parameter in vision_fact.parameters()}
    high_resolution_ids = {
        id(parameter) for parameter in high_resolution.parameters()
    }
    layer_norm_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    groups: dict[tuple[str, float, float], list[Any]] = defaultdict(list)
    decoder_prefixes = (
        "transformer.decoder.",
        "segmentation_head.",
        "dot_prod_scoring.",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in high_resolution_ids:
            group_name, learning_rate = "high_resolution", args.hr_learning_rate
        elif id(parameter) in encoder_3d_ids:
            group_name, learning_rate = "encoder_3d", args.encoder_3d_learning_rate
        elif id(parameter) in fact_ids:
            group_name, learning_rate = "vision_fact", args.fact_learning_rate
        elif id(parameter) in fusion_ids:
            group_name, learning_rate = "fusion", args.fusion_learning_rate
        elif name.startswith("backbone.vision_backbone."):
            group_name, learning_rate = "vision", args.vision_learning_rate
            trunk_prefix = "backbone.vision_backbone.trunk."
            if name.startswith(trunk_prefix) and "pos_embed" not in name:
                trunk = model.backbone.vision_backbone.trunk
                num_layers = trunk.get_num_layers() + 1
                layer_id = trunk.get_layer_id(name)
                learning_rate *= args.vision_layer_decay ** (num_layers - layer_id)
        elif name.startswith("backbone.language_backbone."):
            group_name, learning_rate = "language", args.language_learning_rate
        elif name.startswith(decoder_prefixes):
            group_name, learning_rate = "decoder", args.decoder_learning_rate
        else:
            group_name, learning_rate = "model", args.model_learning_rate
        decay = (
            0.0
            if name.endswith(".bias") or id(parameter) in layer_norm_ids
            else args.weight_decay
        )
        groups[(group_name, learning_rate, decay)].append(parameter)
    optimizer_groups = [
        {
            "params": parameters,
            "lr": learning_rate,
            "base_lr": learning_rate,
            "weight_decay": decay,
            "group_name": group_name,
        }
        for (group_name, learning_rate, decay), parameters in groups.items()
    ]
    return torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
