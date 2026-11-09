# Installation

## SAM3 dependency

Install CUDA-enabled PyTorch and torchvision for your machine, then install this
repository and the pinned SAM3 source:

```bash
pip install -e .
mkdir -p external
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 checkout 8e451d5eb43c817b64ae7577fb7b9ae223db88a9
git -C external/sam3 apply ../../patches/sam3-training.patch
pip install -e 'external/sam3[train]'
```

Download the official SAM3 checkpoint through its upstream distribution and
license process, and place it at `weights/sam3.pt`. No model weights or access
tokens are distributed here. The CLI accepts other local checkpoint paths.

The patch contains two changes used by the training implementation:

1. Use the ordinary MLP operation when gradients are enabled, so activation
   checkpointing does not call an inference-only fused operation.
2. Raise an explicit error on nonfinite predictions, targets, or Hungarian
   costs, instead of failing later inside assignment.

It does not introduce a new assignment cost or replace the original matching
algorithm. No other local SAM3 trainer/configuration changes are required by
the public entry points.

## CPU checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q medspada scripts
```

These tests do not load SAM3 weights or require a GPU. Actual training and
inference use CUDA, BF16 autocast, activation checkpointing, and a 1008-pixel
canvas. Start with a separate `--smoke` run on the intended data.
