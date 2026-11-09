"""Parallel/serial spatial-depth adapters and shared visual Q/V factors."""
from __future__ import annotations

from dataclasses import dataclass
import weakref
import torch
from torch import Tensor, nn
import torch.nn.functional as F

SPATIAL_KINDS = ("unused", "mona_parallel", "mona_serial")

class _BlockWithIntraSliceAdapters(nn.Module):
    """Insert one adapter before attention and one before the MLP."""

    def __init__(
        self,
        block: nn.Module,
        pre_attention: MonaSliceAdapter,
        pre_mlp: MonaSliceAdapter,
    ) -> None:
        super().__init__()
        self.block = block
        self.pre_attention = pre_attention
        self.pre_mlp = pre_mlp

    def forward(self, tokens: Tensor) -> Tensor:
        from sam3.model.vitdet import window_partition, window_unpartition

        tokens = self.pre_attention(tokens)
        shortcut = tokens
        x = self.block.norm1(tokens)
        if self.block.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.block.window_size)
        x = self.block.ls1(self.block.attn(x))
        if self.block.window_size > 0:
            x = window_unpartition(
                x, self.block.window_size, pad_hw, (height, width)
            )
        x = shortcut + self.block.dropout(self.block.drop_path(x))
        x = self.pre_mlp(x)
        adaptformer = getattr(self.block, "prostate_adaptformer", None)
        adaptation = adaptformer(x) if adaptformer is not None else None
        x = x + self.block.dropout(
            self.block.drop_path(self.block.ls2(self.block.mlp(self.block.norm2(x))))
        )
        if adaptation is not None:
            x = x + adaptation
        return x


class _FacTShared(nn.Module):
    """MA-SAM-style shared input/output factors for vision Q/V updates."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)


class _FacTQKV(nn.Module):
    """Keep the pretrained QKV projection and add trainable Q/V deltas only."""

    def __init__(
        self,
        base_qkv: nn.Linear,
        shared: _FacTShared,
        rank: int,
        *,
        scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError("FacT requires one packed QKV linear projection")
        self.base_qkv = base_qkv
        self.q_factor = nn.Linear(rank, rank, bias=False)
        self.v_factor = nn.Linear(rank, rank, bias=False)
        self.q_dropout = nn.Dropout(dropout)
        self.v_dropout = nn.Dropout(dropout)
        self.scale = float(scale)
        self.dim = int(base_qkv.in_features)
        # Registered once on the root model; avoid 32 duplicate state_dict keys.
        self._shared_ref = weakref.ref(shared)

    @property
    def in_features(self) -> int:
        return self.base_qkv.in_features

    @property
    def out_features(self) -> int:
        return self.base_qkv.out_features

    def forward(self, tokens: Tensor) -> Tensor:
        shared = self._shared_ref()
        if shared is None:
            raise RuntimeError("FacT shared factors were released")
        base = self.base_qkv(tokens)
        latent = shared.down(tokens)
        delta_q = shared.up(self.q_dropout(self.q_factor(latent))) * self.scale
        delta_v = shared.up(self.v_dropout(self.v_factor(latent))) * self.scale
        query, key, value = base.split(self.dim, dim=-1)
        return torch.cat((query + delta_q, key, value + delta_v), dim=-1)

    def fact_parameters(self):
        yield from self.q_factor.parameters()
        yield from self.v_factor.parameters()


@dataclass(frozen=True)
class Encoder3DInstallation:
    arm: str
    adapters: tuple[MonaSliceAdapter, ...]
    zero_based_block_indices: tuple[int, ...]
    adapters_per_block: tuple[int, ...]
    context_mode: str
    halo: int
    topology: str = "none"
    training_strategy: str = "full"

    def parameters(self):
        for adapter in self.adapters:
            yield from adapter.parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class FacTInstallation:
    enabled: bool
    rank: int
    scale: float
    zero_based_block_indices: tuple[int, ...]
    shared: _FacTShared | None
    qkv_wrappers: tuple[_FacTQKV, ...]

    def parameters(self):
        if self.shared is not None:
            yield from self.shared.parameters()
        for wrapper in self.qkv_wrappers:
            yield from wrapper.fact_parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def install_vision_fact(
    model: nn.Module,
    *,
    rank: int = 32,
    scale: float = 1.0,
    dropout: float = 0.1,
) -> FacTInstallation:
    """Install shared FacT factors on Q/V in all vision-attention blocks."""

    if rank < 1:
        raise ValueError("FacT rank must be positive")
    if hasattr(model, "prostate_vision_fact"):
        raise RuntimeError("vision FacT is already installed")
    trunk = model.backbone.vision_backbone.trunk
    dim = int(trunk.channel_list[-1])
    reference = next(trunk.parameters())
    shared = _FacTShared(dim, rank).to(
        device=reference.device, dtype=reference.dtype
    )
    model.add_module("prostate_vision_fact", shared)
    wrappers: list[_FacTQKV] = []
    for wrapped_block in trunk.blocks:
        block = (
            wrapped_block.block
            if isinstance(
                wrapped_block, _BlockWithIntraSliceAdapters
            )
            else wrapped_block
        )
        if isinstance(block.attn.qkv, _FacTQKV):
            raise RuntimeError("vision FacT wrappers are already installed")
        wrapper = _FacTQKV(
            block.attn.qkv,
            shared,
            rank,
            scale=scale,
            dropout=dropout,
        ).to(device=reference.device, dtype=reference.dtype)
        block.attn.qkv = wrapper
        wrappers.append(wrapper)
    return FacTInstallation(
        enabled=True,
        rank=int(rank),
        scale=float(scale),
        zero_based_block_indices=tuple(range(len(trunk.blocks))),
        shared=shared,
        qkv_wrappers=tuple(wrappers),
    )


def no_vision_fact() -> FacTInstallation:
    return FacTInstallation(False, 0, 0.0, (), None, ())

class MonaSliceAdapter(nn.Module):
    """Mona 3/5/7 DW filters plus optional full-channel 3x1x1 Z conv.

    SAM3 tokens are [D,H,W,C]; D must contain one patient's ordered slices.
    Up projection is zero initialized for a common exact-identity start.
    """

    def __init__(self, dim: int, bottleneck: int = 64, *, kind: str, dropout: float = 0.1):
        super().__init__()
        if kind not in SPATIAL_KINDS[1:]:
            raise ValueError(kind)
        self.dim, self.bottleneck, self.kind = dim, bottleneck, kind
        self.context_mode = "none" if kind == "mona2d" else "neighbors"
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))
        self.gamma_x = nn.Parameter(torch.ones(dim))
        self.down = nn.Linear(dim, bottleneck)
        self.filters = nn.ModuleList([
            nn.Conv2d(bottleneck, bottleneck, k, padding=k // 2, groups=bottleneck)
            for k in (3, 5, 7)
        ])
        self.pointwise = nn.Conv2d(bottleneck, bottleneck, 1)
        self.depth_conv = (
            nn.Conv3d(bottleneck, bottleneck, (3, 1, 1))
            if kind != "mona2d" else None
        )
        # Runtime execution policy, deliberately excluded from state_dict.
        # A true singleton image has no anatomical Z neighbour.  The shared
        # model keeps Mona's 2D spatial path but can suppress only this term.
        self.depth_residual_enabled = True
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _z(self, x: Tensor) -> Tensor:
        if not self.depth_residual_enabled:
            return torch.zeros_like(x)
        volume = x.permute(1, 0, 2, 3).unsqueeze(0)
        volume = F.pad(volume, (0, 0, 0, 0, 1, 1), mode="replicate")
        return self.depth_conv(volume).squeeze(0).permute(1, 0, 2, 3)

    def set_depth_residual_enabled(self, enabled: bool) -> None:
        self.depth_residual_enabled = bool(enabled)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(f"expected [D,H,W,{self.dim}], got {tuple(tokens.shape)}")
        u = self.down(self.gamma * self.norm(tokens) + self.gamma_x * tokens)
        u = u.permute(0, 3, 1, 2)
        local = u + sum(layer(u) for layer in self.filters) / 3.0
        spatial = local + self.pointwise(local)
        if self.kind == "mona_serial":
            hidden = spatial + self._z(spatial)
        elif self.kind == "mona_parallel":
            hidden = spatial + self._z(u)
        else:
            hidden = spatial
        hidden = self.dropout(F.gelu(hidden.permute(0, 2, 3, 1)))
        return tokens + self.up(hidden)


def install_spatial_variant(model, arm: str, *, kind: str, bottleneck: int):
    if arm != "M3":
        raise ValueError("Mona variants require M3's 32-block dual-adapter topology")
    trunk = model.backbone.vision_backbone.trunk
    if any(isinstance(b, _BlockWithIntraSliceAdapters) for b in trunk.blocks):
        raise RuntimeError("spatial adapters already installed")
    dim = int(trunk.channel_list[-1])
    modules = []
    for index, block in enumerate(trunk.blocks):
        reference = next(block.parameters())
        pair = [MonaSliceAdapter(dim, bottleneck, kind=kind).to(reference) for _ in range(2)]
        trunk.blocks[index] = _BlockWithIntraSliceAdapters(block, *pair)
        modules.extend(pair)
    return Encoder3DInstallation(
        arm, tuple(modules), tuple(range(len(trunk.blocks))),
        (2,) * len(trunk.blocks), "none" if kind == "mona2d" else "neighbors",
        0 if kind == "mona2d" else 2,
        topology=f"intra_block_pre_attention_and_pre_mlp_{kind}",
        training_strategy="peft",
    )


def install_encoder_adaptation(model, *, encoder_arm, bottleneck=384,
                               spatial_kind="mona_parallel", peft_kind="fact_qv",
                               rank=32, scale=1.0, dropout=0.1):
    if encoder_arm == "E0":
        return Encoder3DInstallation("E0", (), (), (), "none", 0), no_vision_fact()
    if encoder_arm != "M3" or peft_kind != "fact_qv":
        raise ValueError("This release supports PSDA + FacT-QV and the full-FT control")
    peft = install_vision_fact(model, rank=rank, scale=scale, dropout=dropout)
    spatial = install_spatial_variant(model, encoder_arm, kind=spatial_kind, bottleneck=bottleneck)
    return spatial, peft
