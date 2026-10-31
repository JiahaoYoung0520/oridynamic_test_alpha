
"""AdaLN-Zero DiT backbone for the all-atom Stage-B latent generator.

The public API intentionally mirrors :class:`LatentDenoiser`:

* ``forward`` is an EDM clean-latent predictor.
* ``forward_flow`` is an un-preconditioned rectified-flow velocity predictor.
* inputs may be one latent ``[N, C]`` or a sample batch ``[K, N, C]``.

This is a standard DiT-style token stack (pre-norm self attention, MLP, and
AdaLN-Zero residual gates), adapted to protein conditioning in two ways.  The
frozen trunk single representation remains a per-token AdaLN condition, and
the pair conditions (``z_trunk`` and a 64-channel reference slot) are projected
independently by every block into an additive bias for every attention head.
The reference slot is exactly zero when reference conditioning is disabled or
dropped for classifier-free reference conditioning, which preserves
architecture size for a controlled ablation. Padding is removed before
attention and scattered back as exact zeros afterwards.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
DISTO_MIN_BIN = 2.3125
DISTO_MAX_BIN = 21.6875
DISTO_NO_BINS = 64

def _linear(in_features: int, out_features: int, *, zero: bool=False) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    if zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer

def _distogram_bins(ref_coords: torch.Tensor) -> torch.Tensor:
    """The same 64 hard distance bins used by the StructVAE conditioning."""
    device_type = ref_coords.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        distance = torch.cdist(ref_coords.float(), ref_coords.float())
        boundaries = torch.linspace(DISTO_MIN_BIN, DISTO_MAX_BIN, DISTO_NO_BINS - 1, device=distance.device, dtype=distance.dtype)
        index = (distance.unsqueeze(-1) > boundaries).sum(dim=-1)
        return F.one_hot(index, DISTO_NO_BINS).float()

class _FrozenFourierEmbedding(nn.Module):
    """AF3-style fixed random Fourier features without core-model imports."""

    def __init__(self, width: int, seed: int=42) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer('w', torch.randn(width, generator=generator))
        self.register_buffer('b', torch.randn(width, generator=generator))

    def forward(self, level: torch.Tensor) -> torch.Tensor:
        level = level.float()
        return torch.cos(2 * torch.pi * (level.unsqueeze(-1) * self.w + self.b))

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Token-wise AdaLN modulation; leading sample dimensions broadcast."""
    return x * (1.0 + scale) + shift

class LatentDiTBlock(nn.Module):
    """DiT block with token conditioning and per-head protein pair bias."""

    def __init__(self, c_hidden: int, c_condition: int, c_pair: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if c_hidden % n_heads:
            raise ValueError('c_hidden must be divisible by n_heads')
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.head_dim = c_hidden // n_heads
        self.norm1 = nn.LayerNorm(c_hidden, elementwise_affine=False, eps=1e-06)
        self.qkv = _linear(c_hidden, 3 * c_hidden)
        self.attention_out = _linear(c_hidden, c_hidden)
        self.norm2 = nn.LayerNorm(c_hidden, elementwise_affine=False, eps=1e-06)
        c_mlp = int(c_hidden * mlp_ratio)
        self.mlp = nn.Sequential(_linear(c_hidden, c_mlp), nn.GELU(approximate='tanh'), _linear(c_mlp, c_hidden))
        self.adaLN = nn.Sequential(nn.SiLU(), _linear(c_condition, 6 * c_hidden, zero=True))
        self.pair_bias_projection = _linear(c_pair, n_heads)

    def _attention(self, x: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-2]
        n_token = x.shape[-2]
        qkv = self.qkv(x).reshape(*leading_shape, n_token, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=-3)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        pair_bias = self.pair_bias_projection(pair).movedim(-1, -3)
        if x.dim() == 3:
            pair_bias = pair_bias.unsqueeze(0)
        pair_bias = pair_bias.to(q.dtype)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=pair_bias, dropout_p=0.0)
        attended = attended.transpose(-3, -2).reshape(*leading_shape, n_token, self.c_hidden)
        return self.attention_out(attended)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(condition).chunk(6, dim=-1)
        attention_input = _modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a * self._attention(attention_input, pair)
        mlp_input = _modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m * self.mlp(mlp_input)

class _FinalLayer(nn.Module):
    """Official DiT-style AdaLN final layer with a zero-init output head."""

    def __init__(self, c_hidden: int, c_condition: int, c_out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_hidden, elementwise_affine=False, eps=1e-06)
        self.adaLN = nn.Sequential(nn.SiLU(), _linear(c_condition, 2 * c_hidden, zero=True))
        self.linear = _linear(c_hidden, c_out, zero=True)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))

class LatentDiT(nn.Module):
    """Parameter-configurable conditional DiT over normalized VAE latents.

    Constructor names and public calls match ``LatentDenoiser`` so a runner
    can select the backbone without changing its data path.
    """

    def __init__(self, c_s: int=384, c_z: int=128, c_hidden: int=256, c_latent: int=32, n_blocks: int=4, n_heads: int=8, c_noise_embedding: int=256, sigma_data: float=1.0, use_s: bool=True, use_z: bool=True, use_ref: bool=True, mlp_ratio: float=3.0, ref_injection: str='pair') -> None:
        super().__init__()
        if not all((isinstance(value, bool) for value in (use_s, use_z, use_ref))):
            raise TypeError('use_s, use_z, and use_ref must be bool')
        dimensions = (c_s, c_z, c_hidden, c_latent, c_noise_embedding)
        if any((value <= 0 for value in dimensions)):
            raise ValueError('all channel dimensions must be positive')
        if n_blocks <= 0 or n_heads <= 0 or mlp_ratio <= 0:
            raise ValueError('n_blocks, n_heads, and mlp_ratio must be positive')
        if c_hidden % n_heads:
            raise ValueError('c_hidden must be divisible by n_heads')
        if sigma_data <= 0:
            raise ValueError('sigma_data must be positive')
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.c_latent = c_latent
        self.n_heads = n_heads
        self.use_s = use_s
        self.use_z = use_z
        self.use_ref = use_ref
        if ref_injection not in ('pair', 'pair_adaln', 'pair_concat', 'pair_cross_attention'):
            raise ValueError('unknown reference injection')
        self.ref_injection = ref_injection
        self.sigma_data = float(sigma_data)
        self.c_z_pair = (c_z if use_z else 0) + DISTO_NO_BINS
        self.layernorm_s = nn.LayerNorm(c_s)
        self.fourier_embedding = _FrozenFourierEmbedding(c_noise_embedding)
        self.layernorm_n = nn.LayerNorm(c_noise_embedding, bias=False)
        self.linear_no_bias_n = nn.Linear(c_noise_embedding, c_s, bias=False)
        nn.init.xavier_uniform_(self.linear_no_bias_n.weight)
        self.z_in = _linear(c_latent, c_hidden)
        self.pair_norm = nn.LayerNorm(self.c_z_pair)
        self.blocks = nn.ModuleList((LatentDiTBlock(c_hidden=c_hidden, c_condition=c_s, c_pair=self.c_z_pair, n_heads=n_heads, mlp_ratio=mlp_ratio) for _ in range(n_blocks)))
        self.final_layer = _FinalLayer(c_hidden, c_s, c_latent)
        if ref_injection != 'pair':
            self.ref_embed = nn.Sequential(_linear(4, c_hidden), nn.SiLU(), _linear(c_hidden, c_hidden))
            if ref_injection == 'pair_adaln':
                self.ref_to_condition = _linear(c_hidden, c_s)
            elif ref_injection == 'pair_concat':
                self.ref_concat = _linear(2 * c_hidden, c_hidden)
            else:
                self.ref_cross = nn.MultiheadAttention(c_hidden, n_heads, batch_first=True)

    def _reference_tokens(self, ref, index):
        if self.ref_injection == 'pair' or ref is None:
            return None
        ref = ref[index] if index is not None else ref
        ref = ref.float() - ref.float().mean(0, keepdim=True)
        features = torch.cat((ref, ref.norm(dim=-1, keepdim=True)), dim=-1) / 16.0
        return self.ref_embed(features.to(self.ref_embed[0].weight.dtype))

    @staticmethod
    def _valid_index(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        return torch.nonzero(mask.reshape(-1) > 0.5).squeeze(-1)

    @staticmethod
    def _scatter_full(value: torch.Tensor, index: Optional[torch.Tensor], n_token: int) -> torch.Tensor:
        if index is None:
            return value
        full = value.new_zeros(*value.shape[:-2], n_token, value.shape[-1])
        full[..., index, :] = value
        return full

    @staticmethod
    def _canonical_level(level: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(level):
            level = torch.tensor(level, device=x.device, dtype=x.dtype)
        level = level.to(device=x.device, dtype=x.dtype)
        if level.numel() == 1:
            return level.reshape(())
        if x.dim() == 3 and level.numel() == x.shape[0]:
            return level.reshape(x.shape[0])
        expected = 'a scalar' if x.dim() == 2 else f'a scalar or {x.shape[0]} values'
        raise ValueError(f'noise/time level must contain {expected}')

    def _prepare(self, x: torch.Tensor, ref_ca: Optional[torch.Tensor], s_trunk: Optional[torch.Tensor], z_trunk: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() not in (2, 3) or x.shape[-1] != self.c_latent:
            raise ValueError(f'latent input must be [N,{self.c_latent}] or [K,N,{self.c_latent}]')
        n_token = x.shape[-2]
        if self.use_ref:
            if ref_ca is not None and tuple(ref_ca.shape) != (n_token, 3):
                raise ValueError(f'ref_ca must have shape {(n_token, 3)}')
        elif ref_ca is not None:
            raise ValueError('use_ref=False requires ref_ca=None')
        if not self.use_s and s_trunk is not None:
            raise ValueError('use_s=False but s_trunk was passed')
        if s_trunk is not None and tuple(s_trunk.shape) != (n_token, self.c_s):
            raise ValueError(f's_trunk must have shape {(n_token, self.c_s)}')
        if self.use_z:
            if z_trunk is None:
                raise ValueError('use_z=True but z_trunk is None')
            expected_z = (n_token, n_token, self.c_z)
            if tuple(z_trunk.shape) != expected_z:
                raise ValueError(f'z_trunk must have shape {expected_z}')
        elif z_trunk is not None:
            raise ValueError('use_z=False but z_trunk was passed')
        if mask is not None and tuple(mask.shape) != (n_token,):
            raise ValueError(f'mask must have shape {(n_token,)}')
        index = self._valid_index(mask)
        if index is not None:
            if index.numel() == 0:
                raise ValueError('mask must contain at least one valid token')
            x = x[..., index, :]
            if ref_ca is not None:
                ref_ca = ref_ca[index]
            s_trunk = s_trunk[index] if s_trunk is not None else None
            if z_trunk is not None:
                z_trunk = z_trunk[index][:, index]
        if ref_ca is not None:
            ref_bins = _distogram_bins(ref_ca)
        else:
            n_valid = x.shape[-2]
            ref_bins = x.new_zeros(n_valid, n_valid, DISTO_NO_BINS)
        if z_trunk is None:
            pair = ref_bins
        else:
            pair = torch.cat((z_trunk, ref_bins.to(z_trunk.dtype)), dim=-1)
        pair = self.pair_norm(pair)
        if s_trunk is None:
            s_trunk = x.new_zeros(x.shape[-2], self.c_s)
        single = self.layernorm_s(s_trunk)
        return (index, x, single, pair)

    def _condition(self, time_coordinate: torch.Tensor, single: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        time_feature = self.fourier_embedding(time_coordinate)
        time_feature = time_feature.to(self.linear_no_bias_n.weight.dtype)
        time_condition = self.linear_no_bias_n(self.layernorm_n(time_feature))
        time_condition = time_condition.to(single.dtype)
        if time_condition.dim() == 1:
            return single + time_condition.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError('per-sample time levels require [K,N,C] latents')
        return single.unsqueeze(0) + time_condition.unsqueeze(-2)

    def _raw_network(self, x: torch.Tensor, time_coordinate: torch.Tensor, single: torch.Tensor, pair: torch.Tensor, reference: Optional[torch.Tensor]=None) -> torch.Tensor:
        if reference is not None and self.ref_injection == 'pair_adaln':
            single = single + self.ref_to_condition(reference)
        condition = self._condition(time_coordinate, single, x)
        hidden = self.z_in(x)
        if reference is not None and self.ref_injection == 'pair_concat':
            ref = reference.expand(*hidden.shape[:-2], *reference.shape)
            hidden = hidden + self.ref_concat(torch.cat((hidden, ref), dim=-1))
        elif reference is not None and self.ref_injection == 'pair_cross_attention':
            ref = reference.expand(*hidden.shape[:-2], *reference.shape)
            hidden = hidden + self.ref_cross(hidden, ref, ref, need_weights=False)[0]
        for block in self.blocks:
            hidden = block(hidden, condition, pair)
        return self.final_layer(hidden, condition)

    def forward(self, z_noisy: torch.Tensor, sigma: torch.Tensor, ref_ca: Optional[torch.Tensor], s_trunk: Optional[torch.Tensor]=None, z_trunk: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """EDM clean prediction, shape-preserving with zeros at padding."""
        sigma = self._canonical_level(sigma, z_noisy)
        index, z_compressed, single, pair = self._prepare(z_noisy, ref_ca, s_trunk, z_trunk, mask)
        broadcast_sigma = sigma
        while broadcast_sigma.dim() < z_noisy.dim():
            broadcast_sigma = broadcast_sigma.unsqueeze(-1)
        c_in = 1.0 / torch.sqrt(self.sigma_data ** 2 + broadcast_sigma ** 2)
        ratio = broadcast_sigma / self.sigma_data
        tiny = torch.finfo(sigma.dtype).tiny
        time_coordinate = torch.log((sigma / self.sigma_data).clamp_min(tiny)) / 4
        update = self._raw_network(c_in * z_compressed, time_coordinate, single, pair, self._reference_tokens(ref_ca, index))
        clean = z_compressed / (1.0 + ratio ** 2) + broadcast_sigma / torch.sqrt(1.0 + ratio ** 2) * update
        return self._scatter_full(clean, index, z_noisy.shape[-2])

    def forward_flow(self, x_t: torch.Tensor, t: torch.Tensor, ref_ca: Optional[torch.Tensor], s_trunk: Optional[torch.Tensor]=None, z_trunk: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Un-preconditioned velocity prediction with direct finite ``t``."""
        t = self._canonical_level(t, x_t)
        index, x_compressed, single, pair = self._prepare(x_t, ref_ca, s_trunk, z_trunk, mask)
        velocity = self._raw_network(x_compressed, t, single, pair, self._reference_tokens(ref_ca, index))
        return self._scatter_full(velocity, index, x_t.shape[-2])
