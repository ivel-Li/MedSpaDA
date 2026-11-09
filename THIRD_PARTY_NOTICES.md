# Third-party notices

## SAM3

This project builds on Meta's [SAM3](https://github.com/facebookresearch/sam3),
with upstream base commit `8e451d5eb43c817b64ae7577fb7b9ae223db88a9`.
The SAM License is reproduced in [LICENSE](LICENSE). SAM3 weights and the
upstream source checkout are obtained separately and retain their terms.
The released wrapper and target/loss integration include SAM3-derived logic.

## Architectural references

- [Mona](https://github.com/Leiyi-Hu/mona): inspiration for the multi-scale
  spatial branch. This implementation adds a depth convolution, changes
  insertion sites to pre-attention/pre-MLP, uses bottleneck 384, and zero
  initializes the output projection. It is not a reproduction of the authors'
  original Swin training setup.
- [FacT](https://github.com/JieShibo/PETL-ViT): shared factorized adaptation.
  The released adaptation is restricted to visual Q/V projections with
  rank-by-rank per-block cores; it does not adapt every projection or MLP.

No complete third-party repository is vendored here. These references identify
the source architectures; they do not imply endorsement or matching benchmark
results.
