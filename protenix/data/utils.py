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

from typing import Mapping, Sequence
import numpy as np
import torch
from biotite.structure import AtomArray

def get_atom_mask_by_name(atom_array: AtomArray, entity_id: int=None, position: int=None, atom_name: str=None, copy_id: int=None) -> np.ndarray:
    """
    Get the atom mask of atoms with specific identifiers.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        entity_id (int): Entity id.
        position (int): Residue index of the atom.
        atom_name (str): Atom name.
        copy_id (copy_id): A asym chain id in N copies of an entity.

    Returns:
        np.ndarray: Array of a bool mask.
    """
    mask = np.ones(atom_array.shape, dtype=np.bool_)
    if entity_id is not None:
        mask &= atom_array.label_entity_id == str(entity_id)
    if position is not None:
        mask &= atom_array.res_id == int(position)
    if atom_name is not None:
        mask &= atom_array.atom_name == str(atom_name)
    if copy_id is not None:
        mask &= atom_array.copy_id == int(copy_id)
    return mask

def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result

def get_ligand_polymer_bond_mask(atom_array: AtomArray, lig_include_ions=False) -> np.ndarray:
    """
    Ref AlphaFold3 SI Chapter 3.7.1.
    Get bonds between the bonded ligand and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        lig_include_ions (bool): whether to include ions in the ligand.

    Returns:
        np.ndarray: bond records between the bonded ligand and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        unique_chain_id, counts = np.unique(atom_array.label_asym_id, return_counts=True)
        chain_id_to_count_map = dict(zip(unique_chain_id, counts))
        ions_mask = np.array([chain_id_to_count_map[label_asym_id] == 1 for label_asym_id in atom_array.label_asym_id])
        lig_mask = (atom_array.mol_type == 'ligand') & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == 'ligand'
    polymer_mask = np.isin(atom_array.mol_type, ['protein', 'rna', 'dna'])
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]
    lig_polymer_bond_indices = np.where(lig_mask[idx_i] & polymer_mask[idx_j] | lig_mask[idx_j] & polymer_mask[idx_i])[0]
    if lig_polymer_bond_indices.size == 0:
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[lig_polymer_bond_indices]
    return lig_polymer_bonds

def make_dummy_feature(features_dict: Mapping[str, torch.Tensor], dummy_feats: Sequence=['msa']) -> dict[str, torch.Tensor]:
    num_token = features_dict['token_index'].shape[0]
    num_atom = features_dict['atom_to_token_idx'].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = get_data_shape_dict(num_token=num_token, num_atom=num_atom, num_msa=num_msa, num_templ=num_templ, num_pocket=num_pockets)
    for feat_name in dummy_feats:
        if feat_name not in ['msa', 'template']:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if 'msa' in dummy_feats:
        features_dict['msa'] = torch.nonzero(features_dict['restype'])[:, 1].unsqueeze(0)
        assert features_dict['msa'].shape == feat_shape['msa']
        features_dict['has_deletion'] = torch.zeros(feat_shape['has_deletion'])
        features_dict['deletion_value'] = torch.zeros(feat_shape['deletion_value'])
        features_dict['profile'] = features_dict['restype'].clone()
        assert features_dict['profile'].shape == feat_shape['profile']
        features_dict['deletion_mean'] = torch.zeros(feat_shape['deletion_mean'])
        for key in ['prot_pair_num_alignments', 'prot_unpair_num_alignments', 'rna_pair_num_alignments', 'rna_unpair_num_alignments']:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)
    if 'template' in dummy_feats:
        features_dict['template_restype'] = torch.ones(feat_shape['template_restype']) * 31
        features_dict['template_all_atom_mask'] = torch.zeros(feat_shape['template_all_atom_mask'])
        features_dict['template_all_atom_positions'] = torch.zeros(feat_shape['template_all_atom_positions'])
    if features_dict['msa'].dim() < 2:
        raise ValueError(f"msa must be 2D, get shape: {features_dict['msa'].shape}")
    return features_dict

def data_type_transform(feat_or_label_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], AtomArray]:
    for key, value in feat_or_label_dict.items():
        if key in IntDataList:
            feat_or_label_dict[key] = value.to(torch.long)
    return feat_or_label_dict
IntDataList = ['residue_index', 'token_index', 'asym_id', 'entity_id', 'sym_id', 'ref_space_uid', 'template_restype', 'atom_to_token_idx', 'atom_to_tokatom_idx', 'frame_atom_index', 'msa', 'entity_mol_id', 'mol_id', 'mol_atom_index']

def get_data_shape_dict(num_token: int, num_atom: int, num_msa: int, num_templ: int, num_pocket: int) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """
    Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        tuple[dict[str, tuple], dict[str, tuple]]: A tuple containing feat and label shape dictionaries.
    """
    feat = {'residue_index': (num_token,), 'token_index': (num_token,), 'asym_id': (num_token,), 'entity_id': (num_token,), 'sym_id': (num_token,), 'restype': (num_token, 32), 'entity_mol_id': (num_atom,), 'mol_id': (num_atom,), 'mol_atom_index': (num_atom,), 'ref_pos': (num_atom, 3), 'ref_mask': (num_atom,), 'ref_element': (num_atom, 128), 'ref_charge': (num_atom,), 'ref_atom_name_chars': (num_atom, 4, 64), 'ref_space_uid': (num_atom,), 'msa': (num_msa, num_token), 'has_deletion': (num_msa, num_token), 'deletion_value': (num_msa, num_token), 'profile': (num_token, 32), 'deletion_mean': (num_token,), 'template_restype': (num_templ, num_token), 'template_all_atom_mask': (num_templ, num_token, 37), 'template_all_atom_positions': (num_templ, num_token, 37, 3), 'template_pseudo_beta_mask': (num_templ, num_token), 'template_backbone_frame_mask': (num_templ, num_token), 'template_distogram': (num_templ, num_token, num_token, 39), 'template_unit_vector': (num_templ, num_token, num_token, 3), 'token_bonds': (num_token, num_token), 'is_protein': (num_atom,), 'is_rna': (num_atom,), 'is_dna': (num_atom,), 'is_ligand': (num_atom,), 'distogram_rep_atom_mask': (num_atom,), 'pae_rep_atom_mask': (num_atom,), 'plddt_m_rep_atom_mask': (num_atom,), 'modified_res_mask': (num_atom,), 'bond_mask': (num_atom, num_atom), 'resolution': (1,)}
    extra_feat = {'atom_to_token_idx': (num_atom,), 'atom_to_tokatom_idx': (num_atom,), 'pae_rep_atom_mask': (num_atom,), 'is_distillation': (1,)}
    label = {'coordinate': (num_atom, 3), 'coordinate_mask': (num_atom,), 'has_frame': (num_token,), 'frame_atom_index': (num_token, 3), 'interested_ligand_mask': (num_pocket, num_atom), 'pocket_mask': (num_pocket, num_atom)}
    all_feat = {**feat, **extra_feat}
    return (all_feat, label)

def get_atom_level_token_mask(token_array, atom_array) -> np.ndarray:
    """
    Create a boolean mask indicating whether each atom in the atom array
    corresponds to an atom-level token (token containing only one atom).

    Returns:
        np.ndarray: Boolean tensor of shape [N_atom] where True indicates
                     the atom belongs to an atom-level token
    """
    atom_level_mask = np.zeros(len(atom_array), dtype=bool)
    for token in token_array:
        if len(token.atom_indices) == 1:
            atom_level_mask[token.atom_indices[0]] = True
    return atom_level_mask

def pad_to(arr: np.ndarray, shape: tuple, **kwargs) -> np.ndarray:
    """Pads an array to a given shape. Wrapper around np.pad().

    Args:
      arr: numpy array to pad
      shape: target shape, use None for axes that should stay the same
      **kwargs: additional args for np.pad, e.g. constant_values=-1

    Returns:
      the padded array

    Raises:
      ValueError if arr and shape have a different number of axes.
    """
    if arr.ndim != len(shape):
        raise ValueError(f'arr and shape have different number of axes. arr.shape={arr.shape!r}, shape={shape!r}')
    num_pad = []
    for axis, width in enumerate(shape):
        if width is None:
            num_pad.append((0, 0))
        elif width >= arr.shape[axis]:
            num_pad.append((0, width - arr.shape[axis]))
        else:
            raise ValueError(f'Can not pad to a smaller shape. arr.shape={arr.shape!r}, shape={shape!r}')
    padded_arr = np.pad(arr, pad_width=num_pad, **kwargs)
    return padded_arr
