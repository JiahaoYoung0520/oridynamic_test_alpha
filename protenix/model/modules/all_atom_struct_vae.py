
# 推理发行版：按符号摘取；来源哈希见runtime_sources.json。
"""All-atom structure VAE built from the native Protenix diffusion blocks.

This module deliberately lives outside DiffusionModule.  It reuses the native
DiffusionConditioning, AtomAttentionEncoder, DiffusionTransformer, and
AtomAttentionDecoder classes with the same widths and block count, then inserts
one continuous token-level bottleneck into the native token transformer stack.

The default base-model layout is:

    AtomAttentionEncoder(3 blocks, c_atom=128, c_atompair=16)
      -> 12 / 24 DiffusionTransformer blocks (c_token=768, 16 heads)
      -> token posterior h [N_token, c_latent]
      -> remaining 12 / 24 DiffusionTransformer blocks
      -> AtomAttentionDecoder(3 blocks)
      -> all-atom displacement

The split preserves the total native transformer depth.  A model can be built
from the actual diffusion config with from_diffusion_config(), and all shared
weights can be initialized from a pretrained DiffusionModule with
load_from_diffusion_module().

The opt-in ``codec_variant="atom_only"`` keeps the same external codec contract
but omits both token-transformer halves. It does not alter the PTX refiner.

Inputs are one unbatched, cropped Protenix example.  Variable-size examples are
handled one at a time, matching the current Protenix/Atlas training path.  The
caller is responsible for mapping and aligning a trajectory frame to the
reference; protenix.data.ensemble_schema provides that boundary.
"""
from __future__ import annotations
import math
from functools import partial
from typing import Any, Mapping, Optional
import torch
import torch.nn as nn
from protenix.model.modules.diffusion import DiffusionConditioning
from protenix.model.modules.primitives import Linear, LinearNoBias
from protenix.model.modules.transformer import AtomAttentionDecoder, AtomAttentionEncoder, DiffusionTransformer
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import checkpoint_blocks
_REQUIRED_ATOM_FEATURES = ('atom_to_token_idx', 'ref_pos', 'ref_charge', 'ref_mask', 'ref_atom_name_chars', 'ref_element', 'd_lm', 'v_lm', 'pad_info', 'relp')

def _plain_dict(value: Any) -> dict[str, Any]:
    """Convert ConfigDict-like objects to a shallow plain dictionary."""
    if value is None:
        return {}
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f'expected a mapping, got {type(value).__name__}')
    return dict(value)

def native_diffusion_architecture(module: nn.Module) -> dict[str, Any]:
    """Return the weight-bearing architecture of a native diffusion stack.

    ``DiffusionModule`` and :class:`AllAtomStructVAE` intentionally expose the
    same native components. Keeping the comparison here prevents a
    base/v2/mini checkpoint from being partially loaded into a different
    topology merely because a permissive ``load_state_dict`` was used.
    Runtime-only choices such as activation checkpointing are excluded;
    widths, attention layouts, and local-attention windows are included.
    """
    atom_encoder = module.atom_attention_encoder.atom_transformer
    token_transformer = module.diffusion_transformer
    atom_decoder = module.atom_attention_decoder.atom_transformer
    return {'sigma_data': float(module.sigma_data), 'c_atom': int(module.c_atom), 'c_atompair': int(module.c_atompair), 'c_token': int(module.c_token), 'c_s': int(module.c_s), 'c_z': int(module.c_z), 'c_s_inputs': int(module.c_s_inputs), 'atom_encoder': {'n_blocks': int(atom_encoder.n_blocks), 'n_heads': int(atom_encoder.n_heads), 'n_queries': int(atom_encoder.n_queries), 'n_keys': int(atom_encoder.n_keys)}, 'transformer': {'n_blocks': int(token_transformer.n_blocks), 'n_heads': int(token_transformer.n_heads)}, 'atom_decoder': {'n_blocks': int(atom_decoder.n_blocks), 'n_heads': int(atom_decoder.n_heads), 'n_queries': int(atom_decoder.n_queries), 'n_keys': int(atom_decoder.n_keys)}, 'frame_latent_enabled': bool(getattr(module, 'frame_latent_enabled', False))}

class AllAtomStructVAE(nn.Module):
    """Reference-conditioned all-atom VAE with native Protenix dimensions.

    The native token transformer is split at bottleneck_block.  The first
    section encodes a reference-relative all-atom displacement, and the second
    decodes h while using a reference-coordinate AtomAttention pass for atom
    skips.  Both AtomAttention passes share the same encoder weights.

    A fixed conditioning noise sets deterministic native EDM input/output
    scales. Target and reference use the same reference-derived origin and
    the same AtomAttentionEncoder plus the same first transformer half. The
    posterior is parameterized from their midpoint activation difference.
    """

    def __init__(self, *, sigma_data: float=16.0, c_atom: int=128, c_atompair: int=16, c_token: int=768, c_s: int=384, c_z: int=128, c_s_inputs: int=449, c_latent: int=32, codec_variant: str='full', atom_encoder: Optional[Mapping[str, Any]]=None, transformer: Optional[Mapping[str, Any]]=None, atom_decoder: Optional[Mapping[str, Any]]=None, bottleneck_block: Optional[int]=None, blocks_per_ckpt: Optional[int]=1, conditioning_sigma: Optional[float]=None, input_coordinate_scale: Optional[float]=None, output_coordinate_scale: Optional[float]=None, zero_init_decoder_output: bool=False) -> None:
        super().__init__()
        atom_encoder_cfg = {'n_blocks': 3, 'n_heads': 4, **_plain_dict(atom_encoder)}
        transformer_cfg = {'n_blocks': 24, 'n_heads': 16, 'drop_path_rate': 0.0, **_plain_dict(transformer)}
        atom_decoder_cfg = {'n_blocks': 3, 'n_heads': 4, **_plain_dict(atom_decoder)}
        if codec_variant not in ('full', 'atom_only'):
            raise ValueError(f'unknown codec_variant: {codec_variant}')
        self.codec_variant = codec_variant
        n_token_blocks = int(transformer_cfg['n_blocks'])
        if codec_variant == 'atom_only':
            if bottleneck_block not in (None, 0):
                raise ValueError('atom_only codec requires bottleneck_block=0 or None')
            bottleneck_block = 0
        elif bottleneck_block is None:
            bottleneck_block = n_token_blocks // 2
        if codec_variant == 'full' and (not 0 < int(bottleneck_block) < n_token_blocks):
            raise ValueError(f'bottleneck_block must split the token transformer into two non-empty parts, got {bottleneck_block} for {n_token_blocks} blocks')
        if c_token % int(transformer_cfg['n_heads']) != 0:
            raise ValueError('c_token must be divisible by transformer n_heads')
        if c_atom % int(atom_encoder_cfg['n_heads']) != 0:
            raise ValueError('c_atom must be divisible by atom_encoder n_heads')
        if c_atom % int(atom_decoder_cfg['n_heads']) != 0:
            raise ValueError('c_atom must be divisible by atom_decoder n_heads')
        self.sigma_data = float(sigma_data)
        self.conditioning_sigma = float(self.sigma_data * math.exp(-1.2) if conditioning_sigma is None else conditioning_sigma)
        if self.sigma_data <= 0 or self.conditioning_sigma <= 0:
            raise ValueError('sigma_data and conditioning_sigma must be positive')
        native_input_scale = math.sqrt(self.sigma_data ** 2 + self.conditioning_sigma ** 2)
        native_output_scale = self.conditioning_sigma / math.sqrt(1.0 + (self.conditioning_sigma / self.sigma_data) ** 2)
        self.input_coordinate_scale = float(native_input_scale if input_coordinate_scale is None else input_coordinate_scale)
        self.output_coordinate_scale = float(native_output_scale if output_coordinate_scale is None else output_coordinate_scale)
        if self.input_coordinate_scale <= 0 or self.output_coordinate_scale <= 0:
            raise ValueError('coordinate scales must be positive')
        self.zero_init_decoder_output = bool(zero_init_decoder_output)
        self.c_atom = int(c_atom)
        self.c_atompair = int(c_atompair)
        self.c_token = int(c_token)
        self.c_s = int(c_s)
        self.c_z = int(c_z)
        self.c_s_inputs = int(c_s_inputs)
        self.c_latent = int(c_latent)
        self.bottleneck_block = int(bottleneck_block)
        self.blocks_per_ckpt = blocks_per_ckpt
        self.atom_encoder_config = atom_encoder_cfg
        self.transformer_config = transformer_cfg
        self.atom_decoder_config = atom_decoder_cfg
        self.diffusion_conditioning = DiffusionConditioning(sigma_data=self.sigma_data, c_z=self.c_z, c_s=self.c_s, c_s_inputs=self.c_s_inputs, frame_latent_enabled=False)
        self.atom_attention_encoder = AtomAttentionEncoder(has_coords=True, c_token=self.c_token, c_atom=self.c_atom, c_atompair=self.c_atompair, c_s=self.c_s, c_z=self.c_z, blocks_per_ckpt=blocks_per_ckpt, **atom_encoder_cfg)
        self.layernorm_s = LayerNorm(self.c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(in_features=self.c_s, out_features=self.c_token, precision=torch.float32, initializer='zeros')
        self.diffusion_transformer = DiffusionTransformer(c_a=self.c_token, c_s=self.c_s, c_z=self.c_z, blocks_per_ckpt=blocks_per_ckpt, **{**transformer_cfg, 'n_blocks': 0 if codec_variant == 'atom_only' else n_token_blocks})
        self.layernorm_a = LayerNorm(self.c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(c_token=self.c_token, c_atom=self.c_atom, c_atompair=self.c_atompair, blocks_per_ckpt=blocks_per_ckpt, **atom_decoder_cfg)
        if self.zero_init_decoder_output:
            self.reset_decoder_output()
        self.posterior_norm = LayerNorm(self.c_token)
        self.mu_head = Linear(self.c_token, self.c_latent, initializer='zeros')
        self.logvar_head = Linear(self.c_token, self.c_latent, initializer='zeros')
        self.latent_in = Linear(self.c_latent, self.c_token, initializer='relu')
        if codec_variant == 'full':
            self.encoder_phase = nn.Parameter(torch.zeros(self.c_s))
            self.decoder_phase = nn.Parameter(torch.zeros(self.c_s))
        else:
            self.register_buffer('encoder_phase', torch.zeros(self.c_s))
            self.register_buffer('decoder_phase', torch.zeros(self.c_s))

    @classmethod
    def from_diffusion_config(cls, diffusion_config: Mapping[str, Any], *, c_latent: int=32, codec_variant: str='full', bottleneck_block: Optional[int]=None, conditioning_sigma: Optional[float]=None, input_coordinate_scale: Optional[float]=None, output_coordinate_scale: Optional[float]=None, zero_init_decoder_output: bool=False) -> 'AllAtomStructVAE':
        """Build from configs.model.diffusion_module without copied constants."""
        cfg = _plain_dict(diffusion_config)
        allowed = {'sigma_data', 'c_atom', 'c_atompair', 'c_token', 'c_s', 'c_z', 'c_s_inputs', 'atom_encoder', 'transformer', 'atom_decoder', 'blocks_per_ckpt'}
        kwargs = {key: cfg[key] for key in allowed if key in cfg}
        kwargs.update(c_latent=c_latent, codec_variant=codec_variant, bottleneck_block=bottleneck_block, conditioning_sigma=conditioning_sigma, input_coordinate_scale=input_coordinate_scale, output_coordinate_scale=output_coordinate_scale, zero_init_decoder_output=zero_init_decoder_output)
        return cls(**kwargs)

    def get_config(self) -> dict[str, Any]:
        """Serializable constructor config stored in checkpoints."""
        return {'sigma_data': self.sigma_data, 'conditioning_sigma': self.conditioning_sigma, 'input_coordinate_scale': self.input_coordinate_scale, 'output_coordinate_scale': self.output_coordinate_scale, 'zero_init_decoder_output': self.zero_init_decoder_output, 'c_atom': self.c_atom, 'c_atompair': self.c_atompair, 'c_token': self.c_token, 'c_s': self.c_s, 'c_z': self.c_z, 'c_s_inputs': self.c_s_inputs, 'c_latent': self.c_latent, 'codec_variant': self.codec_variant, 'atom_encoder': dict(self.atom_encoder_config), 'transformer': dict(self.transformer_config), 'atom_decoder': dict(self.atom_decoder_config), 'bottleneck_block': self.bottleneck_block, 'blocks_per_ckpt': self.blocks_per_ckpt}

    def native_architecture(self) -> dict[str, Any]:
        """Architecture signature used to validate pretrained initialization."""
        return native_diffusion_architecture(self)

    def _validate_inputs(self, reference_coordinate: torch.Tensor, s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, input_feature_dict: Mapping[str, Any], coordinate_mask: Optional[torch.Tensor]) -> None:
        missing = [key for key in _REQUIRED_ATOM_FEATURES if key not in input_feature_dict]
        if missing:
            raise KeyError(f'input_feature_dict missing required keys: {missing}')
        if reference_coordinate.ndim != 2 or reference_coordinate.shape[-1] != 3:
            raise ValueError(f'reference_coordinate must be unbatched [N_atom, 3], got {tuple(reference_coordinate.shape)}')
        atom_to_token = input_feature_dict['atom_to_token_idx']
        if atom_to_token.ndim != 1:
            raise ValueError(f'atom_to_token_idx must be unbatched [N_atom], got {tuple(atom_to_token.shape)}')
        n_atom = reference_coordinate.shape[0]
        if atom_to_token.shape[0] != n_atom:
            raise ValueError('coordinate and atom_to_token_idx atom counts differ')
        if s_inputs.ndim != 2 or s_inputs.shape[-1] != self.c_s_inputs:
            raise ValueError(f's_inputs must be [N_token, {self.c_s_inputs}], got {tuple(s_inputs.shape)}')
        n_token = s_inputs.shape[0]
        if s_trunk.shape != (n_token, self.c_s):
            raise ValueError(f's_trunk must be {(n_token, self.c_s)}, got {tuple(s_trunk.shape)}')
        if z_trunk.shape != (n_token, n_token, self.c_z):
            raise ValueError(f'z_trunk must be {(n_token, n_token, self.c_z)}, got {tuple(z_trunk.shape)}')
        if n_atom and (int(atom_to_token.min()) < 0 or int(atom_to_token.max()) >= n_token):
            raise ValueError('atom_to_token_idx contains an out-of-range token')
        if coordinate_mask is not None and coordinate_mask.shape != (n_atom,):
            raise ValueError(f'coordinate_mask must be {(n_atom,)}, got {tuple(coordinate_mask.shape)}')

    def _coordinate_mask(self, reference_coordinate: torch.Tensor, input_feature_dict: Mapping[str, Any], coordinate_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if coordinate_mask is None:
            coordinate_mask = torch.ones(reference_coordinate.shape[0], device=reference_coordinate.device, dtype=reference_coordinate.dtype)
        else:
            coordinate_mask = coordinate_mask.to(device=reference_coordinate.device, dtype=reference_coordinate.dtype)
        return coordinate_mask

    @staticmethod
    def _masked_center(coordinate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp(min=1.0)
        center = (coordinate * mask[:, None]).sum(dim=-2) / denom
        return (coordinate - center) * mask[:, None]

    @staticmethod
    def _center_on_reference(target_coordinate: torch.Tensor, reference_coordinate: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use one origin so target/reference translations cannot diverge."""
        denom = mask.sum().clamp(min=1.0)
        center = (reference_coordinate * mask[:, None]).sum(dim=-2) / denom
        target = (target_coordinate - center) * mask[:, None]
        reference = (reference_coordinate - center) * mask[:, None]
        return (target, reference)

    @staticmethod
    def _token_mask(atom_to_token_idx: torch.Tensor, atom_mask: torch.Tensor, n_token: int) -> torch.Tensor:
        counts = atom_mask.new_zeros(n_token)
        counts.scatter_add_(0, atom_to_token_idx.long(), atom_mask)
        return (counts > 0).to(atom_mask.dtype)

    def _prepare_conditioning(self, s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, input_feature_dict: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = s_trunk.new_full((1,), self.conditioning_sigma)
        s_single, z_pair = self.diffusion_conditioning(t_hat_noise_level=sigma, relp_feature=input_feature_dict['relp'], s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_z=None, inplace_safe=False, use_conditioning=True, frame_latent=None)
        return (s_single, z_pair, sigma)

    def _atom_encode(self, coordinate_channel: torch.Tensor, s_trunk: torch.Tensor, z_pair: torch.Tensor, input_feature_dict: Mapping[str, Any], *, chunk_size: Optional[int]=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.atom_attention_encoder(atom_to_token_idx=input_feature_dict['atom_to_token_idx'], ref_pos=input_feature_dict['ref_pos'], ref_charge=input_feature_dict['ref_charge'], ref_mask=input_feature_dict['ref_mask'], ref_atom_name_chars=input_feature_dict['ref_atom_name_chars'], ref_element=input_feature_dict['ref_element'], d_lm=input_feature_dict['d_lm'], v_lm=input_feature_dict['v_lm'], pad_info=input_feature_dict['pad_info'], r_l=coordinate_channel.unsqueeze(0), s=s_trunk.unsqueeze(0), z=z_pair.unsqueeze(0), p_lm=None, c_l=None, inplace_safe=False, chunk_size=chunk_size)

    def _run_token_blocks(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor, start: int, end: int, *, chunk_size: Optional[int]=None) -> torch.Tensor:
        if start == end:
            return a
        blocks = [partial(block, inplace_safe=False, chunk_size=chunk_size, enable_efficient_fusion=False) for block in self.diffusion_transformer.blocks[start:end]]
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        if blocks_per_ckpt is not None:
            blocks_per_ckpt = min(int(blocks_per_ckpt), len(blocks))
        a, _, _ = checkpoint_blocks(blocks=blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt)
        return a

    def encode(self, target_coordinate: torch.Tensor, reference_coordinate: torch.Tensor, s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, input_feature_dict: Mapping[str, Any], coordinate_mask: Optional[torch.Tensor]=None, *, chunk_size: Optional[int]=None) -> dict[str, Any]:
        """Encode an aligned target/reference pair into token posterior params."""
        self._validate_inputs(reference_coordinate, s_inputs, s_trunk, z_trunk, input_feature_dict, coordinate_mask)
        if target_coordinate.shape != reference_coordinate.shape:
            raise ValueError('target_coordinate and reference_coordinate must match')
        atom_mask = self._coordinate_mask(reference_coordinate, input_feature_dict, coordinate_mask)
        atom_to_token = input_feature_dict['atom_to_token_idx'].long()
        token_mask = self._token_mask(atom_to_token, atom_mask, s_inputs.shape[0])
        displacement = (target_coordinate - reference_coordinate) * atom_mask[:, None]
        target_centered, reference_centered = self._center_on_reference(target_coordinate, reference_coordinate, atom_mask)
        target_channel = target_centered / self.input_coordinate_scale
        reference_channel = reference_centered / self.input_coordinate_scale
        s_single, z_pair, _ = self._prepare_conditioning(s_inputs, s_trunk, z_trunk, input_feature_dict)
        a_target, _, _, _ = self._atom_encode(target_channel, s_trunk, z_pair, input_feature_dict, chunk_size=chunk_size)
        a_reference, q_ref, c_ref, p_ref = self._atom_encode(reference_channel, s_trunk, z_pair, input_feature_dict, chunk_size=chunk_size)
        condition_token = self.linear_no_bias_s(self.layernorm_s(s_single))
        a_target = a_target.float() + condition_token
        a_reference = a_reference.float() + condition_token
        s_encoder = s_single.float() + self.encoder_phase.view(1, 1, -1)
        z_encoder = z_pair.unsqueeze(0).float()
        u_target = self._run_token_blocks(a_target, s_encoder, z_encoder, 0, self.bottleneck_block, chunk_size=chunk_size)
        u_reference = self._run_token_blocks(a_reference, s_encoder, z_encoder, 0, self.bottleneck_block, chunk_size=chunk_size)
        posterior = self.posterior_norm(u_target - u_reference).squeeze(0)
        mu = self.mu_head(posterior) * token_mask[:, None]
        logvar = self.logvar_head(posterior).clamp(-20.0, 20.0)
        logvar = logvar * token_mask[:, None]
        return {'mu': mu, 'logvar': logvar, 'token_mask': token_mask, 'coordinate_mask': atom_mask, 'target_displacement': displacement, '_reference_cache': {'u_reference': u_reference, 'q_ref': q_ref, 'c_ref': c_ref, 'p_ref': p_ref, 's_single': s_single, 'z_pair': z_pair}}

    def decode(self, latent: torch.Tensor, reference_coordinate: torch.Tensor, s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, input_feature_dict: Mapping[str, Any], coordinate_mask: Optional[torch.Tensor]=None, *, reference_cache: Optional[Mapping[str, torch.Tensor]]=None, chunk_size: Optional[int]=None) -> torch.Tensor:
        """Decode token latent into a masked all-atom displacement."""
        self._validate_inputs(reference_coordinate, s_inputs, s_trunk, z_trunk, input_feature_dict, coordinate_mask)
        n_token = s_inputs.shape[0]
        if latent.shape != (n_token, self.c_latent):
            raise ValueError(f'latent must be {(n_token, self.c_latent)}, got {tuple(latent.shape)}')
        atom_mask = self._coordinate_mask(reference_coordinate, input_feature_dict, coordinate_mask)
        token_mask = self._token_mask(input_feature_dict['atom_to_token_idx'].long(), atom_mask, n_token)
        if reference_cache is None:
            s_single, z_pair, _ = self._prepare_conditioning(s_inputs, s_trunk, z_trunk, input_feature_dict)
            ref_centered = self._masked_center(reference_coordinate, atom_mask) / self.input_coordinate_scale
            a_ref, q_skip, c_skip, p_skip = self._atom_encode(ref_centered, s_trunk, z_pair, input_feature_dict, chunk_size=chunk_size)
            a_ref = a_ref.float() + self.linear_no_bias_s(self.layernorm_s(s_single))
            s_encoder = s_single.float() + self.encoder_phase.view(1, 1, -1)
            u_reference = self._run_token_blocks(a_ref, s_encoder, z_pair.unsqueeze(0).float(), 0, self.bottleneck_block, chunk_size=chunk_size)
        else:
            u_reference = reference_cache['u_reference']
            q_skip = reference_cache['q_ref']
            c_skip = reference_cache['c_ref']
            p_skip = reference_cache['p_ref']
            s_single = reference_cache['s_single']
            z_pair = reference_cache['z_pair']
        latent_token = self.latent_in(latent * token_mask[:, None]).unsqueeze(0)
        a_token = u_reference.float() + latent_token.float()
        s_decoder = s_single.float() + self.decoder_phase.view(1, 1, -1)
        z_decoder = z_pair.unsqueeze(0).float()
        a_token = self._run_token_blocks(a_token, s_decoder, z_decoder, self.bottleneck_block, len(self.diffusion_transformer.blocks), chunk_size=chunk_size)
        a_token = self.layernorm_a(a_token)
        raw_update = self.atom_attention_decoder(atom_to_token_idx=input_feature_dict['atom_to_token_idx'], a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip, inplace_safe=False, chunk_size=chunk_size).squeeze(0)
        c_skip_edm = 1.0 / (1.0 + (self.conditioning_sigma / self.sigma_data) ** 2)
        displacement = (c_skip_edm - 1.0) * self._masked_center(reference_coordinate, atom_mask) + self.output_coordinate_scale * raw_update
        return displacement * atom_mask[:, None]

    @torch.no_grad()
    def reset_decoder_output(self) -> None:
        """Optional ablation: start the native decoder output at exactly zero."""
        nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)
