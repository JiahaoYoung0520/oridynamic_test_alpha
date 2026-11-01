# Copyright 2026 LGLab Frontier AI Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""推理专用Joint；保留既有Heun/SDE、双上下文和anchor位移解码。"""
from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch import nn
from protenix.utils.torch_utils import autocasting_disable_decorator

@dataclass
class JointConfig:
    refiner_precision: str = 'native'
    samples: int = 8
    fm_steps: int = 16
    refiner_steps: int = 20
    sigma_init: float = 16.0
    sigma_min: float = 0.0064
    n_cycle: int = 4
    activation_checkpoint: bool = False
    chunk_size: int | None = None

    def __post_init__(self):
        if self.refiner_precision not in {'native', 'legacy_amp'}:
            raise ValueError('Unknown precision contract')
        if min(self.samples, self.fm_steps, self.refiner_steps, self.n_cycle) < 1:
            raise ValueError('Sample counts must be positive')
        if not 0 < self.sigma_min < self.sigma_init:
            raise ValueError('Invalid sigma range')
        if self.activation_checkpoint:
            raise ValueError('Gradient checkpointing must be disabled for inference')

    @classmethod
    def from_checkpoint(cls, config):
        values = {f.name: config[f.name] for f in fields(cls) if f.name in config}
        values.setdefault('refiner_precision', 'legacy_amp')
        values['activation_checkpoint'] = False
        return cls(**values)

def maybe_checkpoint(fn, *args, enabled):
    if enabled:
        raise ValueError('Gradient checkpointing is unavailable during inference')
    return fn(*args)

def flow_rollout(velocity, initial, steps: int, mask, *, checkpoint_steps=True):
    """Complete Heun noise-to-data rollout, with no detach between steps."""
    if steps < 1:
        raise ValueError('steps must be positive')
    x = initial * mask[..., None]
    dt = 1.0 / steps
    for i in range(steps):
        t = x.new_full((x.shape[0],), i * dt)
        v = maybe_checkpoint(velocity, x, t, enabled=checkpoint_steps)
        euler = (x + dt * v) * mask[..., None]
        v_next = maybe_checkpoint(velocity, euler, t + dt, enabled=checkpoint_steps)
        x = (x + 0.5 * dt * (v + v_next)) * mask[..., None]
    return x

def native_schedule(steps, sigma_init, sigma_min, *, device):
    index = torch.arange(steps + 1, device=device, dtype=torch.float32) / steps
    sigmas = (sigma_init ** (1 / 7) + index * (sigma_min ** (1 / 7) - sigma_init ** (1 / 7))).pow(7)
    return torch.cat((sigmas[:-1], sigmas.new_zeros(1)))

def coordinate_rollout(denoise, proposal, noise, mask, sigmas, *, checkpoint_steps=True, sampler='ode_pure'):
    """Native Euler refinement, gamma0=0, eta=1; one noise per proposal.

    Center restoration is differentiable.  No per-sample Kabsch operation is
    inserted into the sampler.  Tensor coordinate state stays FP32 under AMP.
    """
    if sampler not in {'ode_pure', 'sde_eta10'}:
        raise ValueError(f'unknown coordinate sampler: {sampler}')
    mask = mask.to(proposal.dtype)
    center = (proposal * mask[None, :, None]).sum(-2, keepdim=True) / mask.sum().clamp_min(1)
    x = proposal.float() - center + sigmas[0] * noise.float()
    for i in range(len(sigmas) - 1):
        gamma = 0.8 if sampler == 'sde_eta10' and sigmas[i + 1] > 1.0 else 0.0
        t_hat = sigmas[i] * (1 + gamma)
        if gamma:
            x = x + 1.003 * (t_hat.square() - sigmas[i].square()).sqrt() * torch.randn_like(x)
        sigma = t_hat.expand(x.shape[0])
        clean = maybe_checkpoint(denoise, x, sigma, enabled=checkpoint_steps).float()
        x = x + (sigmas[i + 1] - t_hat) * (x - clean) / t_hat
    output_center = (x * mask[None, :, None]).sum(-2, keepdim=True) / mask.sum().clamp_min(1)
    return x - output_center + center

class JointEnsembleModel(nn.Module):

    def __init__(self, native, generator, codec, mean, std, config: JointConfig):
        super().__init__()
        self.native, self.generator, self.codec = (native, generator, codec)
        self.config = config
        template = getattr(native, 'template_embedder', None)
        self._native_template_blocks = None if template is None else template.n_blocks
        self.disable_online_templates = True
        if template is not None:
            template.n_blocks = 0
        self.reuse_static_conditioning = True
        self.register_buffer('latent_mean', mean.detach().float().clone())
        self.register_buffer('latent_std', std.detach().float().clone())
        if (self.latent_std <= 0).any():
            raise ValueError('latent normalization std must be positive')
        self.codec.requires_grad_(False)

    @torch.no_grad()
    def generate(self, batch, online, *, reference_present=True, noise=None, atom_noise=None, sampler='ode_pure'):
        cfg = self.config
        s_inputs, s, z = online
        ref_token = batch['reference_token'] if reference_present and self.generator.use_ref else None
        mask = batch['token_mask']
        k = cfg.samples if noise is None else noise.shape[0]
        masks = mask[None].expand(k, -1)
        if noise is None:
            noise = torch.randn(k, len(mask), len(self.latent_mean), device=s.device)

        def velocity(x, t):
            return self.generator.forward_flow(x, t, ref_token, s, z, mask)
        latent = flow_rollout(velocity, noise, cfg.fm_steps, masks, checkpoint_steps=cfg.activation_checkpoint)
        mu = latent * self.latent_std + self.latent_mean
        teacher = batch['codec_context']
        reference_cache = None
        if self.reuse_static_conditioning:
            with torch.no_grad():
                reference_cache = self.codec.encode(teacher['reference_coordinate'], teacher['reference_coordinate'], teacher['s_inputs'], teacher['s_trunk'], teacher['z_trunk'], teacher['input_feature_dict'], batch['atom_mask'], chunk_size=cfg.chunk_size)['_reference_cache']

        def decode(h):
            displacement = self.codec.decode(h, teacher['reference_coordinate'], teacher['s_inputs'], teacher['s_trunk'], teacher['z_trunk'], teacher['input_feature_dict'], batch['atom_mask'], reference_cache=reference_cache, chunk_size=cfg.chunk_size)
            return teacher['reference_coordinate'] + displacement
        canonical = torch.stack([maybe_checkpoint(decode, h, enabled=cfg.activation_checkpoint) for h in mu])
        proposal = batch['canonical_frame'].from_canonical(canonical, batch['atom_mask']).float()
        final = self.refine(batch, online, proposal, atom_noise=atom_noise, sampler=sampler)
        return {'final': final, 'proposal': proposal, 'latent': latent}

    def refiner_skip_amp(self):
        if self.config.refiner_precision == 'legacy_amp':
            return False
        skip_amp = getattr(getattr(self.native, 'configs', None), 'skip_amp', None)
        return bool(getattr(skip_amp, 'sample_diffusion', True))

    @torch.no_grad()
    def refine(self, batch, online, proposal, *, atom_noise=None, sampler='ode_pure'):
        """Shared training/inference refiner, also usable on saved proposals."""
        cfg = self.config
        s_inputs, s, z = online
        skip_amp = self.refiner_skip_amp()
        diffusion = self.native.diffusion_module
        pair_cache = autocasting_disable_decorator(skip_amp)(diffusion.diffusion_conditioning.prepare_cache)
        atom_cache = autocasting_disable_decorator(skip_amp)(diffusion.atom_attention_encoder.prepare_cache)
        diffusion_forward = autocasting_disable_decorator(skip_amp)(diffusion)
        pair_z = p_lm = c_l = None
        if self.reuse_static_conditioning:

            def prepare_native_cache(z_condition):
                features = batch['features']
                pair = pair_cache(features['relp'], z_condition, inplace_safe=False)
                atom_pair, atom_single = atom_cache(**{key: features[key] for key in ('ref_pos', 'ref_charge', 'ref_mask', 'ref_element', 'ref_atom_name_chars', 'atom_to_token_idx', 'd_lm', 'v_lm', 'pad_info')}, r_l=True, z=pair, inplace_safe=False)
                return (pair, atom_pair, atom_single)
            pair_z, p_lm, c_l = maybe_checkpoint(prepare_native_cache, z, enabled=cfg.activation_checkpoint)

        def denoise(x, sigma):
            return diffusion_forward(x_noisy=x, t_hat_noise_level=sigma, input_feature_dict=batch['features'], s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_z=pair_z, p_lm=p_lm, c_l=c_l, inplace_safe=False, chunk_size=cfg.chunk_size, enable_efficient_fusion=False)
        sigmas = native_schedule(cfg.refiner_steps, cfg.sigma_init, cfg.sigma_min, device=s.device)
        if atom_noise is None:
            atom_noise = torch.randn_like(proposal)
        final = coordinate_rollout(denoise, proposal, atom_noise, batch['atom_mask'], sigmas, checkpoint_steps=cfg.activation_checkpoint, sampler=sampler)
        return final

    def train(self, mode=True):
        if mode:
            raise RuntimeError('This release provides inference only, without a Joint training interface')
        super().train(False)
        return self

    @torch.no_grad()
    def online_context(self, features):
        if self.training:
            raise RuntimeError('Call eval() before inference')
        if self._native_template_blocks is not None:
            self.native.template_embedder.n_blocks = 0 if self.disable_online_templates else self._native_template_blocks
        return self.native.get_pairformer_output(features, self.config.n_cycle, inplace_safe=False, chunk_size=self.config.chunk_size)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Use generate for inference; training batches and loss requests are unsupported')
