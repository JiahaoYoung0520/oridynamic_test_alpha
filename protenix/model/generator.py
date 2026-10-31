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

from typing import Any, Callable, Optional
import torch
from protenix.model.utils import centre_random_augmentation
from protenix.tfg import parse_tfg_config, TFGEngine
from protenix.utils.logger import get_logger
logger = get_logger(__name__)

class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps).

    Args:
        s_max (float, optional): maximal noise level. Defaults to 160.0.
        s_min (float, optional): minimal noise level. Defaults to 4e-4.
        rho (float, optional): the exponent numerical part. Defaults to 7.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(self, s_max: float=160.0, s_min: float=0.0004, rho: float=7, sigma_data: float=16.0) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f'inference scheduler {self.sigma_data}')

    def __call__(self, N_step: int=200, device: torch.device=torch.device('cpu'), dtype: torch.dtype=torch.float32, sigma_init: Optional[float]=None) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.
            sigma_init (Optional[float], optional): if set, truncate the schedule so
                t_step_list[0] == sigma_init (in scaled units, same as noise_schedule
                values). Used by target-initialized truncated ODE. None -> unchanged
                (top of schedule stays at sigma_data * s_max).

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        s_top = self.s_max if sigma_init is None else sigma_init / self.sigma_data
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = self.sigma_data * (s_top ** (1 / self.rho) + step_indices * step_size * (self.s_min ** (1 / self.rho) - s_top ** (1 / self.rho))) ** self.rho
        t_step_list[..., -1] = 0
        return t_step_list

def _in_temp_stage(step_i: int, num_steps: int, temp_stage: str) -> bool:
    """Whether the temperature shift is active at this sampling step.

    Coarse-to-fine tempering (Li et al., Sec 3.3): the schedule runs from high
    noise (step 0) to low noise (last step). "early" targets the high-noise
    third (cross-mode diversity), "late" the low-noise third (fine detail),
    "middle" the middle third, "full" every step.
    """
    if temp_stage == 'full':
        return True
    third = num_steps / 3.0
    if temp_stage == 'early':
        return step_i < third
    if temp_stage == 'middle':
        return third <= step_i < 2 * third
    if temp_stage == 'late':
        return step_i >= 2 * third
    raise ValueError(f'temp_stage must be one of early/middle/late/full, got {temp_stage!r}')

def sample_diffusion(denoise_net: Callable, input_feature_dict: dict[str, Any], s_inputs: torch.Tensor, s_trunk: torch.Tensor, z_trunk: torch.Tensor, pair_z: torch.Tensor, p_lm: torch.Tensor, c_l: torch.Tensor, noise_schedule: torch.Tensor, N_sample: int=1, gamma0: float=0.8, gamma_min: float=1.0, noise_scale_lambda: float=1.003, step_scale_eta: float=1.5, diffusion_chunk_size: Optional[int]=None, inplace_safe: bool=False, attn_chunk_size: Optional[int]=None, enable_efficient_fusion: bool=False, guidance_configs: Optional[dict[str, Any]]=None, frame_latent: Optional[torch.Tensor]=None, traj_out: Optional[list]=None, x0_out: Optional[list]=None, target_coords: Optional[torch.Tensor]=None, target_coords_mask: Optional[torch.Tensor]=None, sigma_init: Optional[float]=None, disable_augment: Optional[bool]=None, gamma_temp: float=1.0, temp_stage: str='full') -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.
        guidance_configs (Optional[dict[str, Any]]): training free guidance configs. Defaults to None.
        gamma_temp (float): temperature parameter for variance-corrective time shifting
            (Li et al. "Diversify Diffusion with Temperature Sampling", Eq. 12-13).
            0 < gamma_temp < 1 flattens dominant modes / lifts rare modes (more
            ensemble diversity); gamma_temp == 1.0 (default) recovers the exact
            baseline sampler. Implemented by querying the denoiser at the shifted
            noise level sigma_tilde = sqrt(gamma_temp) * sigma while keeping the
            Euler update on the original schedule. Defaults to 1.0.
        temp_stage (str): which coarse-to-fine segment to apply gamma_temp on
            ("early" = high-noise / cross-mode diversity, "middle", "late", or
            "full"). Only active when gamma_temp != 1.0. Defaults to "full".

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict['atom_to_token_idx'].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    tfg_cfg = parse_tfg_config(guidance_configs)
    if tfg_cfg.enable:
        if frame_latent is not None:
            raise NotImplementedError('frame_latent is not supported with TFG yet')
        logger.info('Guidance is enabled.')
        tfg = TFGEngine(tfg_cfg, device=device, dtype=dtype)

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe, frame_latent_chunk=None):
        if target_coords is None:
            x_l = noise_schedule[0] * torch.randn(size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype)
        else:
            target_c = centre_random_augmentation(x_input_coords=target_coords, N_sample=chunk_n_sample, mask=target_coords_mask, centre_only=True).to(dtype)
            x_l = target_c + noise_schedule[0] * torch.randn(size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype)
        if traj_out is not None:
            traj_out.append(x_l.detach().float().cpu())
        if disable_augment is None:
            _disable_rot = target_coords is not None and sigma_init is not None
        else:
            _disable_rot = bool(disable_augment)
        for step_i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            if _disable_rot:
                x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1, centre_only=True).squeeze(dim=-3).to(dtype)
            else:
                x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1).squeeze(dim=-3).to(dtype)
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)
            delta_noise_level = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(size=x_l.shape, device=device, dtype=dtype)
            t_hat = t_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, chunk_n_sample).to(dtype)
            _temp_active = gamma_temp != 1.0 and _in_temp_stage(step_i, len(noise_schedule) - 1, temp_stage)
            if _temp_active:
                sigma_query = t_hat * float(gamma_temp) ** 0.5
            else:
                sigma_query = t_hat
            if tfg_cfg.enable:
                x_l = tfg.step(denoise_net, x=x_noisy, t_hat=t_hat, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion, c_tau=c_tau, step_i=step_i, num_diffusion_steps=len(noise_schedule) - 1, step_scale_eta=step_scale_eta)
            else:
                x_denoised = denoise_net(x_noisy=x_noisy, t_hat_noise_level=sigma_query, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion, frame_latent=frame_latent_chunk)
                delta = (x_noisy - x_denoised) / t_hat[..., None, None]
                dt = c_tau - t_hat
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
                if x0_out is not None:
                    x0_out.append((float(t_hat.flatten()[0]), x_denoised.detach().float().cpu()))
            if traj_out is not None:
                traj_out.append(x_l.detach().float().cpu())
        return x_l
    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe, frame_latent_chunk=frame_latent)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            chunk_n_sample = diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
            frame_latent_chunk = None
            if frame_latent is not None:
                frame_latent_chunk = frame_latent[..., i * diffusion_chunk_size:i * diffusion_chunk_size + chunk_n_sample, :]
            chunk_x_l = _chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe, frame_latent_chunk=frame_latent_chunk)
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)
    return x_l
