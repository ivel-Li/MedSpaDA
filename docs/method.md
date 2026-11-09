# Model and training protocol

## A: Hybrid spatial-depth adaptation

The SAM3 visual encoder retains its pretrained 2D attention. A spatial-depth
adapter is inserted before attention and before the MLP in every visual
transformer block. The current backbone has 32 blocks and therefore 64 adapter
sites. Each adapter compresses channels to a bottleneck of 384.

For bottleneck features $u$, the in-plane path $S(u)$ uses depthwise 3x3, 5x5,
and 7x7 filters, internal residuals, and a 1x1 projection. The depth path $Z(u)$
is a full-channel 3x1x1 convolution over ordered slices, with replicated
boundaries. The two supported connections are:

$$h_{parallel}=S(u)+Z(u),\qquad h_{serial}=S(u)+Z(S(u)).$$

GELU, dropout, an up projection, and an outer residual follow. The zero-initialized
up projection preserves the pretrained block at installation. All tokens in
one visual forward belong to one patient's consecutive slice context.

FacT-QV uses shared down/up factors of rank 32 and per-block Q/V rank-by-rank
cores. It modifies packed attention Q and V; K is unchanged. Original visual
encoder parameters are frozen. Adapters, FacT, and all modules outside the
visual encoder are trainable, including the text and detector paths.

The full-FT control removes both PSDA and FacT and trains the original visual
encoder. It is not a frozen-encoder weak baseline.

## Fixed HR1 supervision

The native mask head is preserved. Its low-resolution logits are interpolated
to 504 and then 1008 before supervision. Gradients flow through both bilinear
operations. No learned deconvolution or new semantic head is introduced.
SAM3 FT retains the original resolution path; component ablations keep HR1.

## B: Volume-supervised refinement

VSR, called C0 in checkpoint keys, is one shared 2D CNN for all concepts.
Let $L_0$ denote the selected query's logit map and $s_0$ its native score.
Its three input channels are:

$$[\tanh(L_0/4),\ \sigma(L_0),\ s_0\mathbf{1}].$$

These represent bounded signed logits, foreground probabilities, and a constant
score map. The CNN uses 24-channel full-resolution features, a 32-channel
half-resolution path, concatenation into 56 channels, and a zero-initialized
residual output. It returns $L=L_0+\Delta L$, with a bounded residual of limit 8.

Query selection uses sigmoid(object logit) times sigmoid(decoder presence).
VSR refines the selected one-to-one query; it does not fuse the top four masks
or change the acceptance threshold of 0.6. Pixel prediction uses logit > 0.

The added training loss is:

$$L_{VSR}=0.5L_{balanced\ BCE}+0.5L_{positive\ slice\ Dice}
          +L_{window\ volume\ Dice}+0.001L_{residual\ L1}.$$

The volume term aggregates the eight supervised core slices for each concept,
not an entire patient's original volume in one optimization step. The native
SAM3 detection, box, presence, mask, semantic, and O2M losses remain active.

## Optimization and evaluation

Default base learning rates are: PSDA and FacT 1e-4, text 5e-6, decoder/head
1e-5, other modules 8e-5. The visual full-FT control uses 2.5e-5 with layer
decay 0.9. AdamW weight decay is 0.1, with bias/norm exclusions; the schedule
uses the native inverse-square-root family. Gradient limits are 0.1 for base
parameters and 1.0 for adaptation parameters.

Train the base for 20 epochs, select its shared dev-best checkpoint, then add
VSR and jointly train for 10 epochs. Base LRs in this second stage are scaled
by 0.1; VSR uses 1e-4. The sampler, native losses, input geometry, and target
assignment remain fixed. This is joint refinement of the trainable base,
not frozen-base-only VSR training.

Every epoch reconstructs all dev volumes. Compute Dice per foreground class
and volume, average classes within the volume, then cases within the dataset,
then datasets equally. Empty prediction plus empty target has Dice 1. Select
one shared checkpoint, not separate best epochs per dataset. Final test uses
all declared test cases once the checkpoint is selected.

Non-overlapping supervised cores give each output slice exactly one final
prediction. Halo slices provide context and do not add supervised samples.
The sampler's tensor batch dimension is therefore an ordered slice sequence,
not a batch of unrelated patients.
