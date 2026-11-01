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

import copy
import random
import time
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from protenix.model import sample_confidence
from protenix.model.generator import InferenceNoiseScheduler, sample_diffusion
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import ConstraintEmbedder, InputFeatureEmbedder, RelativePositionEncoding
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import simple_merge_dict_list
from protenix.utils.logger import get_logger
SymmetricPermutation = Any
from protenix.utils.torch_utils import autocasting_disable_decorator
logger = get_logger(__name__)

def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk
    with torch.no_grad():
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(q=[input_feature_dict['ref_pos'], input_feature_dict['ref_space_uid']], k=[input_feature_dict['ref_pos'], input_feature_dict['ref_space_uid']], dim_q=[-2, -1], dim_k=[-2, -1], n_queries=32, n_keys=128, compute_mask=True)
        d_lm = q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        v_lm = (q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()).unsqueeze(dim=-1)
        input_feature_dict['d_lm'] = d_lm
        input_feature_dict['v_lm'] = v_lm
        input_feature_dict['pad_info'] = pad_info
        return input_feature_dict

class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Protenix, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        self.enable_diffusion_shared_vars_cache = self.configs.enable_diffusion_shared_vars_cache
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = False
        self.inference_noise_scheduler = InferenceNoiseScheduler(**configs.inference_noise_scheduler)
        esm_configs = configs.get('esm', {})
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder, esm_configs=esm_configs)
        self.relative_position_encoding = RelativePositionEncoding(**configs.model.relative_position_encoding)
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(**configs.model.msa_module, msa_configs=configs.data.get('msa', {}))
        self.constraint_embedder = ConstraintEmbedder(**configs.model.constraint_embedder)
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)
        self.c_s, self.c_z, self.c_s_inputs = (configs.c_s, configs.c_z, configs.c_s_inputs)
        self.linear_no_bias_sinit = LinearNoBias(in_features=self.c_s_inputs, out_features=self.c_s)
        self.linear_no_bias_zinit1 = LinearNoBias(in_features=self.c_s, out_features=self.c_z)
        self.linear_no_bias_zinit2 = LinearNoBias(in_features=self.c_s, out_features=self.c_z)
        self.linear_no_bias_token_bond = LinearNoBias(in_features=1, out_features=self.c_z)
        self.linear_no_bias_z_cycle = LinearNoBias(in_features=self.c_z, out_features=self.c_z)
        self.linear_no_bias_s = LinearNoBias(in_features=self.c_s, out_features=self.c_s)
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(self, input_feature_dict: dict[str, Any], N_cycle: int, inplace_safe: bool=False, chunk_size: Optional[int]=None, mc_dropout: bool=False, mc_dropout_rate: float=0.4) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()
        s_inputs = self.input_embedder(input_feature_dict, inplace_safe=False, chunk_size=chunk_size)
        z_constraint = None
        if 'constraint_feature' in input_feature_dict:
            z_constraint = self.constraint_embedder(input_feature_dict['constraint_feature'])
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = self.linear_no_bias_zinit1(s_init)[..., None, :] + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict['relp'])
            z_init += self.linear_no_bias_token_bond(input_feature_dict['token_bonds'].unsqueeze(dim=-1))
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict['relp'])
            z_init = z_init + self.linear_no_bias_token_bond(input_feature_dict['token_bonds'].unsqueeze(dim=-1))
            if z_constraint is not None:
                z_init = z_init + z_constraint
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(self.training and (not self.train_confidence_only) and (cycle_no == N_cycle - 1)):
                if mc_dropout:
                    z = z_init + F.dropout(self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)), p=mc_dropout_rate)
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(input_feature_dict, z, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                    z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(input_feature_dict, z, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                    z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(s, z, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()
        return (s_inputs, s, z)

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {key: self.configs.sample_diffusion.get(key) for key in ['gamma0', 'gamma_min', 'noise_scale_lambda', 'step_scale_eta']}
        _configs.update({'attn_chunk_size': self.configs.infer_setting.chunk_size if not self.training else None, 'diffusion_chunk_size': self.configs.infer_setting.sample_diffusion_chunk_size if not self.training else None})
        _configs.update({'guidance_configs': self.configs.sample_diffusion.to_dict().get('guidance')})
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(sample_diffusion)(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(self.confidence_head)(*args, **kwargs)

    def main_inference_loop(self, input_feature_dict: dict[str, Any], label_dict: dict[str, Any], N_cycle: int, mode: str, inplace_safe: bool=True, chunk_size: Optional[int]=4, N_model_seed: int=1, symmetric_permutation: SymmetricPermutation=None, mc_dropout_apply_rate: float=0.4) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        if N_model_seed > 1 and mode in ['inference']:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(input_feature_dict=copy.deepcopy(input_feature_dict) if N_model_seed > 1 and mode == 'inference' else input_feature_dict, label_dict=label_dict, N_cycle=N_cycle, mode=mode, inplace_safe=inplace_safe, chunk_size=chunk_size, symmetric_permutation=symmetric_permutation, mc_dropout=random.random() < mc_dropout_apply_rate)
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])
            all_pred_dict = {'coordinate': _cat(pred_dicts, 'coordinate'), 'summary_confidence': _list_join(pred_dicts, 'summary_confidence'), 'full_data': _list_join(pred_dicts, 'full_data'), 'plddt': _cat(pred_dicts, 'plddt'), 'pae': _cat(pred_dicts, 'pae'), 'pde': _cat(pred_dicts, 'pde'), 'resolved': _cat(pred_dicts, 'resolved')}
            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return (all_pred_dict, all_log_dict, all_time_dict)
        else:
            return self._main_inference_loop(input_feature_dict=input_feature_dict, label_dict=label_dict, N_cycle=N_cycle, mode=mode, inplace_safe=inplace_safe, chunk_size=chunk_size, symmetric_permutation=symmetric_permutation, mc_dropout=random.random() < mc_dropout_apply_rate)

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, 'chunk_size_thresholds'):
            return self.configs.infer_setting.chunk_size
        thresholds = self.configs.infer_setting.chunk_size_thresholds
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size
        return 32

    def _main_inference_loop(self, input_feature_dict: dict[str, Any], label_dict: dict[str, Any], N_cycle: int, mode: str, inplace_safe: bool=True, chunk_size: Optional[int]=4, symmetric_permutation: SymmetricPermutation=None, mc_dropout: bool=False) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict['residue_index'].shape[-1]
        if hasattr(self.configs.infer_setting, 'dynamic_chunk_size') and self.configs.infer_setting.dynamic_chunk_size:
            chunk_size = self._get_dynamic_chunk_size(N_token)
        log_dict = {}
        pred_dict = {}
        time_tracker = {}
        s_inputs, s, z = self.get_pairformer_output(input_feature_dict=input_feature_dict, N_cycle=N_cycle, inplace_safe=inplace_safe, chunk_size=chunk_size, mc_dropout=mc_dropout)
        keys_to_delete = []
        for key in input_feature_dict.keys():
            if 'template_' in key or key in ['msa', 'has_deletion', 'deletion_value', 'profile', 'deletion_mean']:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({'pairformer': step_trunk - step_st})
        N_sample = self.configs.sample_diffusion['N_sample']
        N_step = self.configs.sample_diffusion['N_step']
        sigma_init = self.configs.sample_diffusion.get('sigma_init')
        target_init_kwargs = {}
        if sigma_init is not None and label_dict is not None:
            target_init_kwargs = {'target_coords': label_dict['coordinate'], 'target_coords_mask': label_dict['coordinate_mask'], 'sigma_init': sigma_init}
        else:
            sigma_init = None
        disable_augment = self.configs.sample_diffusion.get('disable_augment')
        if disable_augment is not None:
            target_init_kwargs['disable_augment'] = disable_augment
        gamma_temp = self.configs.sample_diffusion.get('gamma_temp')
        if gamma_temp is not None:
            target_init_kwargs['gamma_temp'] = gamma_temp
            temp_stage = self.configs.sample_diffusion.get('temp_stage')
            if temp_stage is not None:
                target_init_kwargs['temp_stage'] = temp_stage
        noise_schedule = self.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype, sigma_init=sigma_init)
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache['pair_z'] = autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(self.diffusion_module.diffusion_conditioning.prepare_cache)(input_feature_dict['relp'], z, False)
            cache['p_lm/c_l'] = autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(self.diffusion_module.atom_attention_encoder.prepare_cache)(ref_pos=input_feature_dict['ref_pos'], ref_charge=input_feature_dict['ref_charge'], ref_mask=input_feature_dict['ref_mask'], ref_element=input_feature_dict['ref_element'], ref_atom_name_chars=input_feature_dict['ref_atom_name_chars'], atom_to_token_idx=input_feature_dict['atom_to_token_idx'], d_lm=input_feature_dict['d_lm'], v_lm=input_feature_dict['v_lm'], pad_info=input_feature_dict['pad_info'], r_l=True, z=cache['pair_z'], inplace_safe=False)
        else:
            cache['pair_z'] = None
            cache['p_lm/c_l'] = [None, None]
        frame_latent = self.diffusion_module.sample_frame_latent(batch_shape=s_inputs.shape[:-2], n_sample=N_sample, device=s_inputs.device, dtype=s_inputs.dtype)
        pred_dict['coordinate'] = self.sample_diffusion(denoise_net=self.diffusion_module, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s, z_trunk=None if cache['pair_z'] is not None else z, pair_z=cache['pair_z'], p_lm=cache['p_lm/c_l'][0], c_l=cache['p_lm/c_l'][1], N_sample=N_sample, noise_schedule=noise_schedule, inplace_safe=inplace_safe, enable_efficient_fusion=self.enable_efficient_fusion, frame_latent=frame_latent, **target_init_kwargs)
        step_diffusion = time.time()
        time_tracker.update({'diffusion': step_diffusion - step_trunk})
        pred_dict['contact_probs'] = autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(distogram_logits=self.distogram_head(z), **sample_confidence.get_bin_params(self.configs.confidence_bins.distogram))
        pred_dict['plddt'], pred_dict['pae'], pred_dict['pde'], pred_dict['resolved'] = self.run_confidence_head(input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None, x_pred_coords=pred_dict['coordinate'], triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        step_confidence = time.time()
        time_tracker.update({'confidence': step_confidence - step_diffusion})
        time_tracker.update({'model_forward': time.time() - step_st})
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(input_feature_dict=input_feature_dict, pred_dict=pred_dict, label_dict=label_dict, permute_by_pocket='pocket_mask' in label_dict and 'interested_ligand_mask' in label_dict)
            last_step_seconds = step_confidence
            time_tracker.update({'permutation': time.time() - last_step_seconds})
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get('interested_ligand_mask', None)
        pred_dict['summary_confidence'], pred_dict['full_data'] = autocasting_disable_decorator(True)(sample_confidence.compute_full_data_and_summary)(configs=self.configs, pae_logits=pred_dict['pae'], plddt_logits=pred_dict['plddt'], pde_logits=pred_dict['pde'], contact_probs=pred_dict.get('per_sample_contact_probs', pred_dict['contact_probs']), token_asym_id=input_feature_dict['asym_id'], token_has_frame=input_feature_dict['has_frame'], atom_coordinate=pred_dict['coordinate'], atom_to_token_idx=input_feature_dict['atom_to_token_idx'], atom_is_polymer=1 - input_feature_dict['is_ligand'], N_recycle=N_cycle, interested_atom_mask=interested_atom_mask, return_full_data=True, mol_id=input_feature_dict['mol_id'] if mode != 'inference' else None, elements_one_hot=input_feature_dict['ref_element'] if mode != 'inference' else None)
        return (pred_dict, log_dict, time_tracker)

    @torch.no_grad()
    def forward(self, input_feature_dict, label_full_dict=None, label_dict=None, mode='inference', disable_inplace=False, mc_dropout_apply_rate=0.0):
        if mode != 'inference' or label_full_dict is not None or label_dict is not None or self.training:
            raise ValueError('This release supports label-free inference in eval mode only')
        input_feature_dict = update_input_feature_dict(self.relative_position_encoding.generate_relp(input_feature_dict))
        pred, log, times = self.main_inference_loop(input_feature_dict=input_feature_dict, label_dict=None, N_cycle=self.N_cycle, mode='inference', inplace_safe=not disable_inplace, chunk_size=self.configs.infer_setting.chunk_size, N_model_seed=self.N_model_seed, symmetric_permutation=None, mc_dropout_apply_rate=mc_dropout_apply_rate)
        log.update(time=times)
        return (pred, None, log)
