# Copyright 2024 ByteDance and/or its affiliates.
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

from typing import Any, Optional, Union
import torch
import torch.nn as nn
from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import Linear, LinearNoBias, Transition
from protenix.model.modules.transformer import AtomAttentionDecoder, AtomAttentionEncoder, DiffusionTransformer
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import expand_at_dim, get_checkpoint_fn, permute_final_dims

class FrameDistanceLatentEncoder(nn.Module):

    def __init__(self, latent_dim: int=32, distance_bins: int=64, distance_min: float=2.0, distance_max: float=30.0, hidden_dim: int=128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.distance_bins = distance_bins
        self.register_buffer('distance_centers', torch.linspace(distance_min, distance_max, distance_bins), persistent=False)
        width = (distance_max - distance_min) / max(distance_bins - 1, 1)
        self.rbf_width = max(width, 1e-06)
        self.encoder = nn.Sequential(LayerNorm(distance_bins), Linear(distance_bins, hidden_dim, initializer='relu'), nn.SiLU(), Linear(hidden_dim, hidden_dim, initializer='relu'), nn.SiLU())
        self.mu = Linear(hidden_dim, latent_dim)
        self.logvar = Linear(hidden_dim, latent_dim, initializer='zeros')

    def forward(self, coordinate: torch.Tensor, rep_atom_mask: torch.Tensor, coordinate_mask: Optional[torch.Tensor]=None, sample: bool=True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rep_atom_mask = rep_atom_mask.bool()
        rep_coordinate = coordinate[..., rep_atom_mask, :]
        if coordinate_mask is None:
            token_mask = torch.ones(rep_coordinate.shape[:-1], dtype=rep_coordinate.dtype, device=rep_coordinate.device)
        else:
            token_mask = coordinate_mask[..., rep_atom_mask].to(rep_coordinate.dtype)
        pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        distance = torch.cdist(rep_coordinate.float(), rep_coordinate.float()).to(rep_coordinate.dtype)
        centers = self.distance_centers.to(device=distance.device, dtype=distance.dtype)
        rbf = torch.exp(-0.5 * ((distance.unsqueeze(-1) - centers) / self.rbf_width) ** 2)
        pair_mask = pair_mask.unsqueeze(-1)
        pooled = (rbf * pair_mask).sum(dim=(-3, -2)) / pair_mask.sum(dim=(-3, -2)).clamp(min=1.0)
        h = self.encoder(pooled)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-20.0, max=20.0)
        z = mu
        if sample:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()
        return (z, {'frame_latent_mu': mu, 'frame_latent_logvar': logvar, 'frame_latent_kl': kl})

class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3

    Args:
        sigma_data (float, optional): the standard deviation of the data. Defaults to 16.0.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
        c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
    """

    def __init__(self, sigma_data: float=16.0, c_z: int=128, c_s: int=384, c_s_inputs: int=449, c_noise_embedding: int=256, frame_latent_enabled: bool=False, frame_latent_dim: int=32) -> None:
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.frame_latent_enabled = frame_latent_enabled
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32)
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(in_features=self.c_s + self.c_s_inputs, out_features=self.c_s, precision=torch.float32)
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(in_features=c_noise_embedding, out_features=self.c_s, precision=torch.float32)
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)
        self.frame_latent_proj = Linear(frame_latent_dim, self.c_s, initializer='zeros') if self.frame_latent_enabled else None

    def prepare_cache(self, relp_feature: torch.Tensor, z_trunk: torch.Tensor, inplace_safe: bool=False) -> torch.Tensor:
        pair_z = torch.cat(tensors=[z_trunk, self.relpe(relp_feature)], dim=-1)
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def forward(self, t_hat_noise_level: torch.Tensor, relp_feature: torch.Tensor, s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, pair_z: torch.Tensor, inplace_safe: bool=False, use_conditioning: bool=True, frame_latent: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            asym_id (torch.Tensor): asym_id
            residue_index (torch.Tensor): residue_index
            entity_id (torch.Tensor): entity_id
            token_index (torch.Tensor): token_index
            sym_id (torch.Tensor): sym_id
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        elif inplace_safe:
            pair_z_clone = pair_z.clone()
            pair_z = pair_z_clone
        single_s = torch.cat(tensors=[s_trunk, s_inputs], dim=-1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4).to(single_s.dtype)
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(self.layernorm_n(noise_n)).unsqueeze(dim=-2)
        if self.frame_latent_enabled and frame_latent is not None:
            latent_s = self.frame_latent_proj(frame_latent).to(single_s.dtype)
            while latent_s.dim() < single_s.dim() - 1:
                latent_s = latent_s.unsqueeze(dim=-2)
            single_s = single_s + latent_s.unsqueeze(dim=-2)
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        return (single_s, pair_z)

class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        c_token (int, optional): feature channel of token (single a). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
        atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
        atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing is performed.
        use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
            only effective if blocks_per_ckpt is not None.
    """

    def __init__(self, sigma_data: float=16.0, c_atom: int=128, c_atompair: int=16, c_token: int=768, c_s: int=384, c_z: int=128, c_s_inputs: int=449, atom_encoder: dict[str, int]={'n_blocks': 3, 'n_heads': 4}, transformer: dict[str, int]={'n_blocks': 24, 'n_heads': 16, 'drop_path_rate': 0}, atom_decoder: dict[str, int]={'n_blocks': 3, 'n_heads': 4}, drop_path_rate: float=0.0, blocks_per_ckpt: Optional[int]=None, use_fine_grained_checkpoint: bool=False, frame_latent: Optional[dict[str, Any]]=None) -> None:
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint
        frame_latent = frame_latent or {}
        self.frame_latent_enabled = bool(frame_latent.get('enabled', False))
        self.frame_latent_dim = int(frame_latent.get('latent_dim', 32))
        self.frame_latent_encoder = FrameDistanceLatentEncoder(latent_dim=self.frame_latent_dim, distance_bins=int(frame_latent.get('distance_bins', 64)), distance_min=float(frame_latent.get('distance_min', 2.0)), distance_max=float(frame_latent.get('distance_max', 30.0)), hidden_dim=int(frame_latent.get('hidden_dim', 128))) if self.frame_latent_enabled else None
        self.diffusion_conditioning = DiffusionConditioning(sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs, frame_latent_enabled=self.frame_latent_enabled, frame_latent_dim=self.frame_latent_dim)
        self.atom_attention_encoder = AtomAttentionEncoder(**atom_encoder, c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, has_coords=True, c_s=c_s, c_z=c_z, blocks_per_ckpt=blocks_per_ckpt)
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token, precision=torch.float32, initializer='zeros')
        self.diffusion_transformer = DiffusionTransformer(**transformer, c_a=c_token, c_s=c_s, c_z=c_z, blocks_per_ckpt=blocks_per_ckpt)
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(**atom_decoder, c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, blocks_per_ckpt=blocks_per_ckpt)
        self.normalize = LayerNorm(c_z, create_offset=False, create_scale=False)

    def encode_frame_latent(self, coordinate: torch.Tensor, rep_atom_mask: torch.Tensor, coordinate_mask: Optional[torch.Tensor]=None, sample: bool=True) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        if not self.frame_latent_enabled:
            return (None, {})
        return self.frame_latent_encoder(coordinate=coordinate, rep_atom_mask=rep_atom_mask, coordinate_mask=coordinate_mask, sample=sample)

    def sample_frame_latent(self, batch_shape: tuple[int, ...], n_sample: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.frame_latent_enabled:
            return None
        return torch.randn(*batch_shape, n_sample, self.frame_latent_dim, device=device, dtype=dtype)

    def f_forward(self, r_noisy: torch.Tensor, t_hat_noise_level: torch.Tensor, input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]], s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, pair_z: torch.Tensor, p_lm: torch.Tensor, c_l: torch.Tensor, inplace_safe: bool=False, chunk_size: Optional[int]=None, use_conditioning: bool=True, enable_efficient_fusion: bool=False, frame_latent: Optional[torch.Tensor]=None) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(self.diffusion_conditioning, t_hat_noise_level, input_feature_dict['relp'], s_inputs, s_trunk, z_trunk, pair_z, inplace_safe, use_conditioning, frame_latent)
        else:
            s_single, z_pair = self.diffusion_conditioning(t_hat_noise_level, input_feature_dict['relp'], s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning, frame_latent=frame_latent)
        s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)
        z_pair = expand_at_dim(z_pair, dim=-4, n=1)
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(self.atom_attention_encoder, input_feature_dict['atom_to_token_idx'], input_feature_dict['ref_pos'], input_feature_dict['ref_charge'], input_feature_dict['ref_mask'], input_feature_dict['ref_atom_name_chars'], input_feature_dict['ref_element'], input_feature_dict['d_lm'], input_feature_dict['v_lm'], input_feature_dict['pad_info'], r_noisy, s_trunk, z_pair, p_lm, c_l, inplace_safe, chunk_size)
        else:
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(input_feature_dict['atom_to_token_idx'], input_feature_dict['ref_pos'], input_feature_dict['ref_charge'], input_feature_dict['ref_mask'], input_feature_dict['ref_atom_name_chars'], input_feature_dict['ref_element'], input_feature_dict['d_lm'], input_feature_dict['v_lm'], input_feature_dict['pad_info'], r_l=r_noisy, s=s_trunk, z=z_pair, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size)
        a_token = a_token.to(dtype=torch.float32)
        if inplace_safe:
            a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
        else:
            a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))
        if enable_efficient_fusion:
            z = self.normalize(z_pair.to(dtype=torch.float32))
            z = permute_final_dims(z, [2, 0, 1]).contiguous()
        else:
            z = z_pair.to(dtype=torch.float32)
        a_token = self.diffusion_transformer(a=a_token.to(dtype=torch.float32), s=s_single.to(dtype=torch.float32), z=z, inplace_safe=inplace_safe, chunk_size=chunk_size, enable_efficient_fusion=enable_efficient_fusion)
        a_token = self.layernorm_a(a_token)
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(self.atom_attention_decoder, input_feature_dict['atom_to_token_idx'], a_token, q_skip, c_skip, p_skip, inplace_safe, chunk_size)
        else:
            r_update = self.atom_attention_decoder(atom_to_token_idx=input_feature_dict['atom_to_token_idx'], a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size)
        return r_update

    def forward(self, x_noisy: torch.Tensor, t_hat_noise_level: torch.Tensor, input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]], s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, pair_z: torch.Tensor, p_lm: torch.Tensor, c_l: torch.Tensor, inplace_safe: bool=False, chunk_size: Optional[int]=None, use_conditioning: bool=True, enable_efficient_fusion: bool=False, frame_latent: Optional[torch.Tensor]=None) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        r_noisy = x_noisy / torch.sqrt(self.sigma_data ** 2 + t_hat_noise_level ** 2)[..., None, None]
        r_update = self.f_forward(r_noisy=r_noisy, t_hat_noise_level=t_hat_noise_level, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size, use_conditioning=use_conditioning, enable_efficient_fusion=enable_efficient_fusion, frame_latent=frame_latent)
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(r_update.dtype)
        x_denoised = (1 / (1 + s_ratio ** 2) * x_noisy + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * r_update).to(r_update.dtype)
        return x_denoised
