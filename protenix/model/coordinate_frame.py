
"""Deterministic reference-defined Cartesian frame for all-atom latent codecs.

Only world-space target/reference coordinates are transformed. Chemical-local
``ref_pos`` and scalar/vector trunk features are deliberately left untouched.
The anchor rule is O(N_atom): farthest valid atom from the masked centroid,
farthest valid atom from that anchor, then farthest valid atom from their axis.
Index order resolves near-ties. Geometry is computed in float64 so a joint
proper rotation and translation selects the same frame whenever the reference
is well conditioned.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
RAW_COORDINATE_FRAME_VERSION = 'raw_v1'
REFERENCE_CANONICAL_FRAME_VERSION = 'reference_anchor_frame_v1'
RAW_ANCHOR_SELECTION_VERSION = 'none'
REFERENCE_CANONICAL_ANCHOR_SELECTION_VERSION = 'masked_centroid_farthest_pair_axis_v1'
_TIE_RELATIVE_TOLERANCE = 1e-06
_COLLINEAR_RELATIVE_TOLERANCE = 1e-06

def frame_version_for_mode(mode: str) -> str:
    if mode == 'raw':
        return RAW_COORDINATE_FRAME_VERSION
    if mode == 'reference_canonical':
        return REFERENCE_CANONICAL_FRAME_VERSION
    raise ValueError(f'unknown coordinate frame mode: {mode!r}')

def anchor_selection_version_for_mode(mode: str) -> str:
    if mode == 'raw':
        return RAW_ANCHOR_SELECTION_VERSION
    if mode == 'reference_canonical':
        return REFERENCE_CANONICAL_ANCHOR_SELECTION_VERSION
    raise ValueError(f'unknown coordinate frame mode: {mode!r}')

def _stable_argmax(scores: torch.Tensor) -> int:
    """Choose the lowest original index within a small relative max-score tie."""
    maximum = scores.max()
    tolerance = torch.clamp(maximum.abs() * _TIE_RELATIVE_TOLERANCE, min=1e-24)
    return int(torch.nonzero(scores >= maximum - tolerance, as_tuple=False)[0, 0])

@dataclass(frozen=True)
class ReferenceCanonicalFrame:
    """Reference-local frame; ``basis`` columns are right-handed world axes.

    Row-vector convention: ``local = (world - origin) @ basis`` and
    ``world = local @ basis.T + origin``. The stored mask is used by default,
    keeping invalid/padded atoms zero even though the origin is nonzero.
    """
    origin: torch.Tensor
    basis: torch.Tensor
    coordinate_mask: torch.Tensor
    anchor_indices: tuple[int, int, int]

    def _mask_for(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor | None) -> torch.Tensor:
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError('coordinates must be a torch.Tensor')
        if not coordinates.is_floating_point():
            raise TypeError('coordinates must use a floating-point dtype')
        if coordinates.ndim < 2 or coordinates.shape[-1] != 3:
            raise ValueError('coordinates must have shape [..., N_atom, 3]')
        mask = self.coordinate_mask if coordinate_mask is None else coordinate_mask
        mask = torch.as_tensor(mask, device=coordinates.device).bool()
        if mask.ndim != 1 or int(mask.shape[0]) != int(coordinates.shape[-2]):
            raise ValueError('coordinate_mask must have shape [N_atom]')
        return mask.view(*[1] * (coordinates.ndim - 2), -1, 1)

    def to_canonical(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor | None=None) -> torch.Tensor:
        """Transform world coordinates to reference-local coordinates."""
        mask = self._mask_for(coordinates, coordinate_mask)
        origin = self.origin.to(coordinates.device)
        basis = self.basis.to(coordinates.device)
        local = (coordinates.to(torch.float64) - origin) @ basis
        result = local.to(coordinates.dtype)
        return torch.where(mask, result, torch.zeros_like(result))

    def from_canonical(self, coordinates: torch.Tensor, coordinate_mask: torch.Tensor | None=None) -> torch.Tensor:
        """Transform reference-local coordinates back to world coordinates."""
        mask = self._mask_for(coordinates, coordinate_mask)
        origin = self.origin.to(coordinates.device)
        basis = self.basis.to(coordinates.device)
        world = coordinates.to(torch.float64) @ basis.T + origin
        result = world.to(coordinates.dtype)
        return torch.where(mask, result, torch.zeros_like(result))

def build_reference_canonical_frame(reference_coordinate: torch.Tensor, coordinate_mask: torch.Tensor) -> ReferenceCanonicalFrame:
    """Build a proper frame from valid reference atoms, independent of target.

    At least three non-collinear valid atoms are required. Collinearity is
    tested relative to the reference extent; no arbitrary global axis is used
    because such an axis would destroy rotation consistency.
    """
    if not isinstance(reference_coordinate, torch.Tensor):
        raise TypeError('reference_coordinate must be a torch.Tensor')
    if reference_coordinate.ndim != 2 or reference_coordinate.shape[-1] != 3:
        raise ValueError('reference_coordinate must have shape [N_atom, 3]')
    mask = torch.as_tensor(coordinate_mask, device=reference_coordinate.device).bool()
    if mask.ndim != 1 or mask.shape[0] != reference_coordinate.shape[0]:
        raise ValueError('coordinate_mask must have shape [N_atom]')
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if int(valid_indices.numel()) < 3:
        raise ValueError('reference needs at least three non-collinear valid atoms')
    valid_xyz = reference_coordinate[valid_indices].to(torch.float64)
    if not bool(torch.isfinite(valid_xyz).all()):
        raise ValueError('valid reference coordinates must be finite')
    origin = valid_xyz.mean(dim=0)
    centered = valid_xyz - origin
    radius_squared = centered.square().sum(dim=-1)
    first = _stable_argmax(radius_squared)
    distance_squared = (valid_xyz - valid_xyz[first]).square().sum(dim=-1)
    second = _stable_argmax(distance_squared)
    axis = valid_xyz[second] - valid_xyz[first]
    axis_length = torch.linalg.vector_norm(axis)
    extent = torch.sqrt(radius_squared.max())
    if bool(axis_length <= extent * _COLLINEAR_RELATIVE_TOLERANCE):
        raise ValueError('reference needs at least three non-collinear valid atoms')
    e1 = axis / axis_length
    from_first = valid_xyz - valid_xyz[first]
    perpendicular = from_first - (from_first @ e1).unsqueeze(-1) * e1
    perpendicular_squared = perpendicular.square().sum(dim=-1)
    third = _stable_argmax(perpendicular_squared)
    perpendicular_length = torch.sqrt(perpendicular_squared[third])
    if bool(perpendicular_length <= extent * _COLLINEAR_RELATIVE_TOLERANCE):
        raise ValueError('reference needs at least three non-collinear valid atoms')
    e2 = perpendicular[third] / perpendicular_length
    e3 = torch.linalg.cross(e1, e2)
    basis = torch.stack((e1, e2, e3), dim=-1)
    anchor_indices = tuple((int(valid_indices[index]) for index in (first, second, third)))
    return ReferenceCanonicalFrame(origin, basis, mask, anchor_indices)

def canonicalize_batch(batch: Mapping[str, object], mode: str) -> tuple[Mapping[str, object], ReferenceCanonicalFrame | None]:
    """Canonicalize world coordinates without altering chemical/trunk inputs.

    ``raw`` returns the exact original mapping. ``reference_canonical`` returns
    a shallow copy, replacing only reference and (if present) target coordinates.
    The reference defines the frame for both; invalid atoms are zeroed.
    """
    frame_version_for_mode(mode)
    if mode == 'raw':
        return (batch, None)
    reference = batch['reference_coordinate']
    mask = batch['coordinate_mask']
    if not isinstance(reference, torch.Tensor):
        raise TypeError('batch reference_coordinate must be a torch.Tensor')
    frame_mask = batch.get('reference_coordinate_mask', mask)
    frame = build_reference_canonical_frame(reference, torch.as_tensor(frame_mask))
    canonical = dict(batch)
    canonical['reference_coordinate'] = frame.to_canonical(reference)
    if 'target_coordinate' in batch:
        target = batch['target_coordinate']
        if not isinstance(target, torch.Tensor):
            raise TypeError('batch target_coordinate must be a torch.Tensor')
        canonical['target_coordinate'] = frame.to_canonical(target, coordinate_mask=torch.as_tensor(mask))
    return (canonical, frame)
