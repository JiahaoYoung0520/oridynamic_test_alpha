"""CPU微型推理fixture；真实codec/DiT/diffusion，不含训练步骤。"""
import torch
from torch import nn
from protenix.model.modules.all_atom_struct_vae import AllAtomStructVAE
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.latent_dit import LatentDiT
from protenix.model.modules.primitives import rearrange_qk_to_dense_trunk
from protenix.model.lora import inject_lora
from protenix.model.coordinate_frame import canonicalize_batch
from protenix.model.modules.joint_ensemble import JointConfig, JointEnsembleModel
def _tiny_config():
    return {'sigma_data': 16.0, 'c_atom': 8, 'c_atompair': 4, 'c_token': 16, 'c_s': 8, 'c_z': 4, 'c_s_inputs': 5, 'atom_encoder': {'n_blocks': 1, 'n_heads': 2}, 'transformer': {'n_blocks': 2, 'n_heads': 2, 'drop_path_rate': 0.0}, 'atom_decoder': {'n_blocks': 1, 'n_heads': 2}, 'blocks_per_ckpt': None}

def _example():
    torch.manual_seed(3)
    n_atom, n_token = (6, 3)
    ref_pos = torch.randn(n_atom, 3)
    ref_space_uid = torch.zeros(n_atom, dtype=torch.long)
    q_trunked, k_trunked, pad_info = rearrange_qk_to_dense_trunk(q=[ref_pos, ref_space_uid], k=[ref_pos, ref_space_uid], dim_q=[-2, -1], dim_k=[-2, -1], n_queries=32, n_keys=128, compute_mask=True)
    d_lm = q_trunked[0][..., None, :] - k_trunked[0][..., None, :, :]
    v_lm = (q_trunked[1][..., None].int() == k_trunked[1][..., None, :].int()).unsqueeze(-1)
    atom_to_token = torch.tensor([0, 0, 1, 1, 2, 2])
    atom_features = {'atom_to_token_idx': atom_to_token, 'ref_pos': ref_pos, 'ref_charge': torch.zeros(n_atom), 'ref_mask': torch.ones(n_atom), 'ref_atom_name_chars': torch.zeros(n_atom, 4, 64), 'ref_element': torch.zeros(n_atom, 128), 'd_lm': d_lm, 'v_lm': v_lm, 'pad_info': pad_info, 'relp': torch.zeros(n_token, n_token, 139)}
    reference = torch.randn(n_atom, 3)
    target = reference + 0.2 * torch.randn_like(reference)
    coordinate_mask = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    target[-1] = 123.0
    return {'reference_coordinate': reference, 'target_coordinate': target, 'coordinate_mask': coordinate_mask, 's_inputs': torch.randn(n_token, 5), 's_trunk': torch.randn(n_token, 8), 'z_trunk': torch.randn(n_token, n_token, 4), 'input_feature_dict': atom_features}
class TinyTrunk(nn.Module):
    """Only trunk computation is replaced; diffusion/codec/DiT are real."""

    def __init__(self):
        super().__init__()
        self.single = nn.Linear(8, 8)
        self.pair = nn.Linear(4, 4)
        self.diffusion_module = DiffusionModule(**_tiny_config())
        trainable = {n for n, p in self.diffusion_module.named_parameters() if p.requires_grad}
        self.requires_grad_(False)
        for name, p in self.diffusion_module.named_parameters():
            p.requires_grad_(name in trainable)
        self.trunk = nn.ModuleList([self.single, self.pair])
        inject_lora(self.trunk, r=2, alpha=2, verbose=False)

    def get_pairformer_output(self, features, *args, **kwargs):
        return (features['test_si'], self.trunk[0](features['test_s']), self.trunk[1](features['test_z']))

def fixture(checkpoint=True):
    torch.manual_seed(91)
    example = _example()
    example['coordinate_mask'] = torch.ones(6, dtype=torch.bool)
    example['target_coordinate'] = example['reference_coordinate'] + torch.randn(6, 3) * 0.2
    teacher, frame = canonicalize_batch(example, 'reference_canonical')
    codec = AllAtomStructVAE.from_diffusion_config(_tiny_config(), c_latent=3, bottleneck_block=1)
    generator = LatentDiT(c_s=8, c_z=4, c_hidden=16, c_latent=3, n_blocks=2, n_heads=2, c_noise_embedding=12, use_ref=False)
    with torch.no_grad():
        for module in (codec, generator):
            for p in module.parameters():
                if p.ndim >= 2 and (not p.any()):
                    p.normal_(std=0.03)
    native = TinyTrunk()
    cfg = JointConfig(samples=2, fm_steps=2, refiner_steps=2, sigma_init=1.0, activation_checkpoint=False)
    model = JointEnsembleModel(native, generator, codec, torch.zeros(3), torch.ones(3), cfg)
    model.condition_contract = 'fixed_base_codec__online_lora_generator_refiner_v1'
    features = dict(example['input_feature_dict'])
    features.update(test_si=example['s_inputs'], test_s=example['s_trunk'], test_z=example['z_trunk'])
    target = example['reference_coordinate'][None] + 0.3 * torch.randn(4, 6, 3)
    batch = dict(features=features, codec_context=teacher, canonical_frame=frame, token_mask=torch.ones(3, dtype=torch.bool), atom_mask=torch.ones(6, dtype=torch.bool), loss_atom_mask=torch.ones(6, dtype=torch.bool), representative_mask=torch.ones(6, dtype=torch.bool), reference_token=torch.randn(3, 3), target=target, posterior_mu=torch.randn(2, 3, 3), posterior_mask=torch.ones(2, 3, dtype=torch.bool), bonds=torch.tensor([[0, 1], [2, 3], [4, 5]]), excluded_pairs=torch.tensor([1, 15, 29]))
    for key in ('target', 'posterior_mu', 'posterior_mask', 'bonds', 'excluded_pairs', 'loss_atom_mask', 'representative_mask'):
        batch.pop(key, None)
    return model.eval().requires_grad_(False), batch
