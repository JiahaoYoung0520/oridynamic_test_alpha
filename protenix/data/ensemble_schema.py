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

"""Entity-agnostic data contract for reference-conditioned ensembles.

The two important invariants in this module are:

* trajectory atoms are matched to reference atoms by stable identities, never
  merely by array length; and
* the atoms used to fit a rigid transform are explicit.  In particular, a
  ligand or a non-anchor protein chain is transformed with the complex but is
  not independently fitted, so its relative motion remains observable.

The module intentionally depends only on NumPy, PyTorch and Protenix's existing
AF3 weighted rigid alignment implementation.  It is suitable for CPU-side data
preparation and does not require a CUDA runtime.
"""
from __future__ import annotations
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np
import torch
ArrayLike = Union[np.ndarray, torch.Tensor]
MaskLike = Union[Sequence[bool], np.ndarray, torch.Tensor]
IdentityLike = Union['AtomIdentity', Hashable, Mapping[str, Any]]

def _python_scalar(value: Any) -> Any:
    """Convert NumPy/zero-dimensional torch scalars into hashable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError('atom identity fields must be scalar')
        return value.detach().cpu().item()
    return value

@dataclass(frozen=True)
class AtomIdentity:
    """Stable identity of one atom across a reference and trajectory.

    ``asym_id`` distinguishes physical chain copies, while ``entity_id`` and
    ``sym_id`` retain Protenix's entity/symmetry semantics.  ``component_id``
    and ``insertion_code`` make the key safe for ligands and PDB numbering.
    Loaders should normalise these fields once, before constructing identities.
    """
    asym_id: Hashable
    residue_index: Hashable
    atom_name: str
    entity_id: Optional[Hashable] = None
    sym_id: Optional[Hashable] = None
    component_id: str = ''
    insertion_code: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.atom_name, str) or not self.atom_name.strip():
            raise ValueError('atom_name must be a non-empty string')
        for name in ('asym_id', 'residue_index', 'entity_id', 'sym_id'):
            value = _python_scalar(getattr(self, name))
            if not isinstance(value, Hashable):
                raise TypeError(f'AtomIdentity.{name} must be hashable')
            object.__setattr__(self, name, value)

def _identity_key(identity: IdentityLike) -> Hashable:
    if isinstance(identity, Mapping):
        identity = AtomIdentity(**identity)
    if isinstance(identity, list):
        identity = tuple((_python_scalar(value) for value in identity))
    elif isinstance(identity, tuple):
        identity = tuple((_python_scalar(value) for value in identity))
    else:
        identity = _python_scalar(identity)
    if not isinstance(identity, Hashable):
        raise TypeError(f'atom identity must be hashable, got {type(identity).__name__}')
    return identity

def _unique_index(identities: Sequence[IdentityLike], side: str) -> dict[Hashable, int]:
    index: dict[Hashable, int] = {}
    duplicates: list[Hashable] = []
    for atom_index, identity in enumerate(identities):
        key = _identity_key(identity)
        if key in index:
            duplicates.append(key)
        else:
            index[key] = atom_index
    if duplicates:
        preview = ', '.join((repr(key) for key in duplicates[:3]))
        raise ValueError(f'duplicate {side} atom identities: {preview}')
    return index

@dataclass(frozen=True)
class AtomIdentityMapping:
    """Bidirectional atom-index mapping between target and reference order.

    Missing atoms are represented by ``-1``.  Mapping a target tensor produces
    a tensor in reference order plus an explicit validity mask.
    """
    reference_to_target: torch.Tensor
    target_to_reference: torch.Tensor

    def __post_init__(self) -> None:
        ref_to_target = torch.as_tensor(self.reference_to_target, dtype=torch.long).cpu()
        target_to_ref = torch.as_tensor(self.target_to_reference, dtype=torch.long).cpu()
        if ref_to_target.ndim != 1 or target_to_ref.ndim != 1:
            raise ValueError('atom identity mappings must be one-dimensional')
        object.__setattr__(self, 'reference_to_target', ref_to_target)
        object.__setattr__(self, 'target_to_reference', target_to_ref)

    @property
    def num_reference_atoms(self) -> int:
        return int(self.reference_to_target.numel())

    @property
    def num_target_atoms(self) -> int:
        return int(self.target_to_reference.numel())

    @property
    def reference_matched_mask(self) -> torch.Tensor:
        return self.reference_to_target >= 0

    @property
    def target_matched_mask(self) -> torch.Tensor:
        return self.target_to_reference >= 0

    @property
    def num_matched(self) -> int:
        return int(self.reference_matched_mask.sum().item())

    def gather_target(self, values: ArrayLike, fill_value: Union[int, float]=0) -> ArrayLike:
        """Gather an atom-leading target array into reference atom order."""
        if values.shape[0] != self.num_target_atoms:
            raise ValueError(f'target array has {values.shape[0]} atoms; expected {self.num_target_atoms}')
        output_shape = (self.num_reference_atoms, *values.shape[1:])
        matched = self.reference_matched_mask
        if isinstance(values, torch.Tensor):
            output = torch.full(output_shape, fill_value=fill_value, dtype=values.dtype, device=values.device)
            output[matched.to(values.device)] = values[self.reference_to_target[matched].to(values.device)]
            return output
        values_np = np.asarray(values)
        output_np = np.full(output_shape, fill_value=fill_value, dtype=values_np.dtype)
        matched_np = matched.numpy()
        output_np[matched_np] = values_np[self.reference_to_target[matched].numpy()]
        return output_np

    def gather_target_mask(self, target_mask: Optional[MaskLike]=None) -> torch.Tensor:
        """Return a boolean target-valid mask in reference atom order."""
        matched = self.reference_matched_mask.clone()
        if target_mask is None:
            return matched
        target_mask_t = validate_mask(target_mask, self.num_target_atoms, 'target_mask').cpu()
        matched_indices = self.reference_to_target[matched]
        matched[matched.clone()] = target_mask_t[matched_indices]
        return matched

def build_atom_identity_mapping(reference_identities: Sequence[IdentityLike], target_identities: Sequence[IdentityLike], *, require_all_reference: bool=False, require_all_target: bool=False) -> AtomIdentityMapping:
    """Match atoms by stable identities and return target/reference index maps.

    Array length and order never participate in the match.  Missing atoms are
    allowed by default because experimental structures and trajectories can
    have different resolved masks; callers must propagate the returned mask.
    Duplicate identities are always rejected because their mapping is
    ambiguous.
    """
    reference_index = _unique_index(reference_identities, 'reference')
    target_index = _unique_index(target_identities, 'target')
    reference_to_target = torch.full((len(reference_identities),), -1, dtype=torch.long)
    target_to_reference = torch.full((len(target_identities),), -1, dtype=torch.long)
    for identity, ref_index in reference_index.items():
        target_atom_index = target_index.get(identity)
        if target_atom_index is not None:
            reference_to_target[ref_index] = target_atom_index
            target_to_reference[target_atom_index] = ref_index
    if require_all_reference and torch.any(reference_to_target < 0):
        missing = [reference_identities[index] for index in torch.where(reference_to_target < 0)[0].tolist()[:3]]
        raise ValueError(f'target is missing reference atoms: {missing!r}')
    if require_all_target and torch.any(target_to_reference < 0):
        extra = [target_identities[index] for index in torch.where(target_to_reference < 0)[0].tolist()[:3]]
        raise ValueError(f'target contains atoms absent from reference: {extra!r}')
    return AtomIdentityMapping(reference_to_target, target_to_reference)

def validate_mask(mask: MaskLike, length: int, name: str='mask') -> torch.Tensor:
    """Validate and convert a one-dimensional binary mask to ``torch.bool``."""
    mask_t = torch.as_tensor(mask)
    if mask_t.ndim != 1 or mask_t.shape[0] != length:
        raise ValueError(f'{name} must have shape ({length},), got {tuple(mask_t.shape)}')
    if mask_t.dtype != torch.bool:
        if torch.is_floating_point(mask_t) and (not bool(torch.isfinite(mask_t).all())):
            raise ValueError(f'{name} contains non-finite values')
        is_binary = torch.logical_or(mask_t == 0, mask_t == 1)
        if not bool(is_binary.all()):
            raise ValueError(f'{name} must contain only boolean/0/1 values')
    return mask_t.bool()
