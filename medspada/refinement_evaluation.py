from __future__ import annotations

import time
import numpy as np
from medspada.data import DATASETS, LEGACY
from medspada.refinement_loss import dice, selected_q0

def evaluate(model, runner, c0, dataset, args, torch, F, v2):
    from medspada.evaluation import concept_chunks, window_context
    from medspada.native import transform_native_window

    model.eval()
    c0.eval()
    rows = []
    dataset_names = tuple(getattr(args, "datasets", DATASETS))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for dataset_name in dataset_names:
            cases = dataset.cases(
                dataset=dataset_name, split=getattr(args, "eval_split", "dev")
            )
            if args.max_dev_cases_per_dataset is not None:
                cases = cases[:args.max_dev_cases_per_dataset]
            for case in cases:
                sample = dataset.load(case)
                base_prediction = np.zeros_like(sample.target, dtype=np.bool_)
                c0_prediction = np.zeros_like(sample.target, dtype=np.bool_)
                for concept_indices in concept_chunks(len(sample.concepts), args.max_concepts):
                    selected = sample.select_concepts(concept_indices)
                    categories = len(concept_indices)
                    for volume_start in range(0, sample.depth, args.core_depth):
                        images, z_np, core_offset, valid_depth = window_context(
                            selected, volume_start, args.core_depth, args.halo
                        )
                        dummy = selected.target[:, volume_start:volume_start + valid_depth].reshape(
                            categories * valid_depth, *selected.target.shape[-2:]
                        )
                        image_batch, _, _ = transform_native_window(
                            images, dummy, resolution=1008, min_resolution=1008,
                            rng=random.Random(0), torch=torch, v2=v2,
                        )
                        z_mm = torch.from_numpy(np.asarray(z_np)).to("cuda", dtype=torch.float32)
                        encoding = runner.encode_patient(image_batch, selected.prompts)
                        fused, _ = runner.fuse_patient(encoding, z_mm)
                        core_end = core_offset + valid_depth
                        for start, end, output in runner.iter_final_outputs(encoding, fused):
                            if end <= core_offset:
                                continue
                            if start >= core_end:
                                break
                            q0, score, _ = selected_q0(output, runner.num_o2o, torch)
                            refined, _ = c0(q0.float(), score.float())
                            accepted = score >= args.score_threshold
                            base_native = F.interpolate(
                                q0[:, None].float(), selected.target.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )[:, 0] > 0
                            c0_native = F.interpolate(
                                refined[:, None], selected.target.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )[:, 0] > 0
                            base_native &= accepted[:, None, None]
                            c0_native &= accepted[:, None, None]
                            relative = start - core_offset
                            z_index = volume_start + relative
                            for local, global_index in enumerate(concept_indices):
                                base_prediction[global_index, z_index] = base_native[local].cpu().numpy()
                                c0_prediction[global_index, z_index] = c0_native[local].cpu().numpy()
                target = sample.target > 0.5
                base_scores = [dice(base_prediction[i], target[i]) for i in range(len(sample.concepts))]
                c0_scores = [dice(c0_prediction[i], target[i]) for i in range(len(sample.concepts))]
                rows.append({
                    "dataset": dataset_name,
                    "case_id": case.case_id,
                    "base_patient_macro_dice": float(np.mean(base_scores)),
                    "c0_patient_macro_dice": float(np.mean(c0_scores)),
                })
    by_dataset = {}
    for name in dataset_names:
        selected = [row for row in rows if row["dataset"] == name]
        by_dataset[name] = {
            "cases": len(selected),
            "base_patient_macro_dice": float(np.mean([row["base_patient_macro_dice"] for row in selected])),
            "c0_patient_macro_dice": float(np.mean([row["c0_patient_macro_dice"] for row in selected])),
        }
    base_macro = float(np.mean([value["base_patient_macro_dice"] for value in by_dataset.values()]))
    c0_macro = float(np.mean([value["c0_patient_macro_dice"] for value in by_dataset.values()]))
    return {"base_macro": base_macro, "c0_macro": c0_macro, "delta": c0_macro - base_macro,
            "by_dataset": by_dataset, "cases": rows}
