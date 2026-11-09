# Data interface

## Manifest

Paths are resolved relative to the manifest. IDs below are fictional examples:

```json
{
  "datasets": {
    "ExampleMRI": {
      "concepts": [
        {"name": "organ", "label_id": 1, "prompt": "organ"}
      ]
    }
  },
  "cases": [
    {"dataset": "ExampleMRI", "case_id": "example-train",
     "patient_id": "subject-train", "split": "train", "path": "train.npz"},
    {"dataset": "ExampleMRI", "case_id": "example-dev",
     "patient_id": "subject-dev", "split": "dev", "path": "dev.npz"},
    {"dataset": "ExampleMRI", "case_id": "example-test",
     "patient_id": "subject-test", "split": "test", "path": "test.npz"}
  ]
}
```

Cases from the same patient must remain in the same split within a dataset.
This includes cardiac phases, repeated scans, and related acquisitions. The
loader rejects duplicate case IDs and cross-split patient identities.

## Volume arrays

| Key | Shape and dtype | Meaning |
| --- | --- | --- |
| `images` | `[D,H,W,3]`, uint8 | Ordered RGB-encoded slices |
| `label` | `[D,H,W]`, integer | Background 0 and foreground IDs declared in the manifest |
| `z_mm` | `[D]`, float | Physical slice coordinates, finite and strictly increasing |

Store arrays using `numpy.savez_compressed`. NPZ loading disables pickle.
Image and label grids must match exactly; the loader does not silently crop
mismatched shapes. Inference-only input needs `images` and `z_mm`, not `label`.

Prepare orientation, registration, resampling, CT windows/MRI normalization,
and any body crop explicitly before exporting these arrays. Apply the same
spatial transform to images and labels, using nearest-neighbor interpolation
for labels. The training transform then applies one shared square resize and
padding to all slices in the chosen context. It does not infer physical
geometry from a slice's array index.

The default experiments use approximately 3-mm ordered sampling and eight
supervised slices with two context slices on each side. The loader accepts
other explicit physical coordinates; changing preprocessing is an experimental
choice and must be reported. A singleton is processed with a one-slice core
and its depth residual disabled, rather than treated as eight real slices.

## Multiple tasks and sparse targets

Each task declares its own foreground ontology, while all tasks share model
weights. At most five concepts are sampled for a training window. The default
recipe draws 64 patient windows per task per epoch and accumulates one window
from each task per optimizer update.

The configured foreground mixtures are keyed by dataset names:

- `MSD_Task06_Lung`: probability 0.5 of forcing a foreground-containing core.
- `AMOS22`: probability 0.25 of forcing a core containing a selected foreground
  concept.
- Other task names: uniform patient-local window sampling.

The remaining branch is uniform, not necessarily pure background. These
probabilities are not the final foreground rate. Cases are chosen first and
are not replaced just because a random window is negative.

The code treats each concept as a binary segmentation target, with its
foreground components represented together. It is not an instance-ID
benchmark. Split manifests and data licenses must be established before
training; test labels are read only after dev selection.
