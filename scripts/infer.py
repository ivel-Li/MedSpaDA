"""Text-only volume inference; input NPZ needs images and z_mm, never labels."""
import argparse
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image

from medspada.cli import add_common, configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torchvision.transforms import v2
    from medspada.data import Concept, Volume
    from medspada.evaluation import build_model, concept_chunks, window_context
    from medspada.native import transform_native_window
    from medspada.refinement import GeneralistSpatialResidualC0
    from medspada.refinement_loss import selected_q0
    from medspada.runner import Sam3PatientVolumeRunner
    with np.load(args.input, allow_pickle=False) as data:
        images, z_mm = data["images"], data["z_mm"].astype(np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("images must be uint8 [D,H,W,3]")
    if z_mm.shape != (len(images),) or not np.isfinite(z_mm).all() or (np.diff(z_mm) <= 0).any():
        raise ValueError("z_mm must be finite, strictly increasing, and match image depth")
    concepts = tuple(Concept(prompt, i+1, prompt) for i, prompt in enumerate(args.prompts))
    sample = Volume([Image.fromarray(im) for im in images],
                    np.zeros((len(concepts), *images.shape[:3]), np.float32), z_mm, concepts)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model, fusion, _ = build_model(saved["model_name"], torch)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.backbone.act_ckpt_whole_vision_backbone = False
    model.backbone.act_ckpt_whole_language_backbone = False
    model.use_act_checkpoint_seg_head = False
    vsr = None
    if saved.get("c0") is not None:
        vsr = GeneralistSpatialResidualC0().cuda()
        vsr.load_state_dict(saved["c0"], strict=True); vsr.eval()
    runner = Sam3PatientVolumeRunner(model, fusion, slice_microbatch=12,
                                    head_slice_microbatch=1, encoder_halo=0, singleton_halo=1)
    masks = np.zeros_like(sample.target, dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in concept_chunks(len(concepts), 5):
            selected = sample.select_concepts(indices)
            for volume_start in range(0, sample.depth, 8):
                imgs, z, offset, valid = window_context(selected, volume_start, 8, 2)
                dummy = np.zeros((len(indices)*valid, *images.shape[1:3]), np.float32)
                image_batch, _, _ = transform_native_window(imgs, dummy, resolution=1008,
                        min_resolution=1008, rng=random.Random(0), torch=torch, v2=v2)
                encoding = runner.encode_patient(image_batch, selected.prompts)
                fused, _ = runner.fuse_patient(encoding, torch.from_numpy(z).cuda())
                for begin, end, output in runner.iter_final_outputs(encoding, fused):
                    if end <= offset: continue
                    if begin >= offset + valid: break
                    logits, scores, _ = selected_q0(output, runner.num_o2o, torch)
                    if vsr is not None:
                        logits, _ = vsr(logits.float(), scores.float())
                    resized = torch.nn.functional.interpolate(logits[:, None].float(),
                            size=images.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
                    prediction = ((resized > 0) & (scores >= 0.6)[:, None, None]).cpu().numpy()
                    masks[indices, volume_start + begin - offset] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, masks=masks, z_mm=z_mm, prompts=np.asarray(args.prompts))


if __name__ == "__main__":
    main()
