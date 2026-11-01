"""仅权重加载、特征检查、采样与CIF写出；无训练入口。"""
from __future__ import annotations
import copy, hashlib, json, os, random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import torch
from protenix.data.ensemble_schema import build_atom_identity_mapping
from protenix.model.coordinate_frame import frame_version_for_mode, anchor_selection_version_for_mode
from protenix.model.modules.all_atom_struct_vae import AllAtomStructVAE

def require(condition, message):
    if not condition:
        raise ValueError(message)

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    os.replace(temp, path)

def _build_denoiser(backbone, config):
    from protenix.model.modules.latent_dit import LatentDiT
    require(backbone == 'dit' and config.get('ref_injection') != 'native_tokens',
            'This release requires the validated Joint DiT architecture')
    return LatentDiT(**dict(config))

def configure_sample_template(model, args, contract):
    require(contract == 'joint_online_native_null_template_v1', 'Unsupported template contract')
    native_null_template_settings(model)
    model.disable_online_templates = False
    model.require_null_templates = True


def _torch_load(path: Path) -> Any:
    from oridynamic.weights_io import load_weights
    return load_weights(path, 'codec')

def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == 'cuda')
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device, non_blocking=device.type == 'cuda')
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple((_move_to_device(item, device) for item in value))
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value

def find_diffusion_config(payload: Any) -> Optional[dict[str, Any]]:
    """Extract a resolved diffusion config from common checkpoint/config layouts."""
    if hasattr(payload, 'to_dict'):
        payload = payload.to_dict()
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get('diffusion_config')
    if isinstance(direct, Mapping):
        return _plain_tree(direct)
    direct = payload.get('diffusion_module')
    if isinstance(direct, Mapping) and (not any((isinstance(value, torch.Tensor) for value in direct.values()))):
        return _plain_tree(direct)
    model = payload.get('model')
    if isinstance(model, Mapping) and isinstance(model.get('diffusion_module'), Mapping):
        return _plain_tree(model['diffusion_module'])
    for key in ('model_config', 'config', 'configs'):
        if key in payload:
            found = find_diffusion_config(payload[key])
            if found is not None:
                return found
    return None

def find_vae_options(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    model_config = payload.get('model_config', payload)
    if isinstance(model_config, Mapping) and isinstance(model_config.get('vae'), Mapping):
        return _plain_tree(model_config['vae'])
    return {}

def validate_architecture(diffusion_config: dict[str, Any], summary: Mapping[str, Any], *, allow_infer_c_z: bool) -> dict[str, Any]:
    """Infer only c_z when allowed, then check every record-visible width."""
    config = copy.deepcopy(diffusion_config)
    record_c_z = int(summary['c_z'])
    if 'c_z' not in config or (allow_infer_c_z and int(config['c_z']) != record_c_z):
        config['c_z'] = record_c_z
    expected = {'c_s_inputs': int(summary['c_s_inputs']), 'c_s': int(summary['c_s']), 'c_z': record_c_z}
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) is None or int(config[key]) != value}
    if mismatches:
        raise ValueError(f'record/model diffusion width mismatch (config, record): {mismatches}. Regenerate context with the same Protenix model.')
    if 'c_token' not in config:
        raise ValueError('diffusion config has no local c_token. Do not substitute the global Protenix c_token; provide model.diffusion_module config explicitly.')
    if not isinstance(config.get('transformer'), Mapping):
        raise ValueError('diffusion config must contain its native transformer block config')
    return config

def _extract_model_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ('model', 'state_dict'):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if payload and all((isinstance(value, torch.Tensor) for value in payload.values())):
            return payload
    raise ValueError("checkpoint has no tensor state dict under 'model' or 'state_dict'")

def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(state)
    while result and all((key.startswith('module.') for key in result)):
        result = {key[len('module.'):]: value for key, value in result.items()}
    return result

_FLAT_CONFIG_REQUIRED = frozenset({'c_atom', 'c_atompair', 'c_token', 'c_s', 'c_z', 'c_s_inputs', 'c_latent', 'atom_encoder', 'transformer', 'atom_decoder'})

_FLAT_CONFIG_ALLOWED = frozenset({'codec_variant', 'sigma_data', 'c_atom', 'c_atompair', 'c_token', 'c_s', 'c_z', 'c_s_inputs', 'c_latent', 'atom_encoder', 'transformer', 'atom_decoder', 'bottleneck_block', 'blocks_per_ckpt', 'conditioning_sigma', 'input_coordinate_scale', 'output_coordinate_scale', 'zero_init_decoder_output'})

_VAE_FACTORY_ALLOWED = frozenset({'codec_variant', 'c_latent', 'bottleneck_block', 'conditioning_sigma', 'input_coordinate_scale', 'output_coordinate_scale', 'zero_init_decoder_output'})

def _plain_tree(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_tree(item) for item in value]
    if isinstance(value, list):
        return [_plain_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def _mapping_nodes(payload: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """Return likely config nodes without walking tensor state dictionaries."""
    if not isinstance(payload, Mapping):
        return []
    queue: list[tuple[str, Mapping[str, Any]]] = [('checkpoint', payload)]
    result: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[int] = set()
    while queue:
        label, node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        result.append((label, node))
        for key in ('model_config', 'config', 'configs', 'vae_config'):
            child = node.get(key)
            if isinstance(child, Mapping):
                queue.append((f'{label}.{key}', child))
    return result

def _find_flat_vae_config(payload: Any) -> tuple[Optional[dict[str, Any]], str]:
    for label, node in _mapping_nodes(payload):
        if _FLAT_CONFIG_REQUIRED.issubset(node):
            return ({key: _plain_tree(node[key]) for key in _FLAT_CONFIG_ALLOWED if key in node}, label)
    return (None, '')

def _find_nested_vae_options(payload: Any) -> dict[str, Any]:
    options = find_vae_options(payload)
    if options:
        return options
    for _, node in _mapping_nodes(payload):
        candidate = node.get('vae')
        if isinstance(candidate, Mapping):
            return _plain_tree(candidate)
    return {}

def _validate_flat_widths(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    expected = {'c_s_inputs': int(summary['c_s_inputs']), 'c_s': int(summary['c_s']), 'c_z': int(summary['c_z'])}
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) is None or int(config[key]) != value}
    if mismatch:
        raise ValueError(f'flat VAE config/record width mismatch (checkpoint, record): {mismatch}. The context and checkpoint must use the same Protenix base.')

def load_vae_checkpoint(checkpoint_path: Path, summary: Mapping[str, Any], device: torch.device, *, allow_anchor_free: bool=False) -> tuple[AllAtomStructVAE, dict[str, Any]]:
    """Load strict VAE weights from either supported config layout."""
    payload = _torch_load(checkpoint_path)
    carrier = payload.get('train_config', {}).get('carrier_protocol', {})
    if carrier.get('protocol') == 'anchor_free_latent_or_product_v1':
        raise ValueError('Anchor-free/dual-space checkpoints are incompatible with this release codec')
    if carrier.get('protocol') == 'zero_global_anchor_codec_v1' and (not allow_anchor_free):
        raise ValueError('Experimental codecs are incompatible with this release anchor/cache contract')
    state = _strip_module_prefix(_extract_model_state(payload))
    diffusion_config = find_diffusion_config(payload)
    if diffusion_config is not None:
        diffusion_config = validate_architecture(diffusion_config, summary, allow_infer_c_z=False)
        vae_options = {key: value for key, value in _find_nested_vae_options(payload).items() if key in _VAE_FACTORY_ALLOWED}
        model = AllAtomStructVAE.from_diffusion_config(diffusion_config, **vae_options)
        layout = 'nested'
        config_source = 'diffusion_config + vae'
    else:
        flat_config, config_source = _find_flat_vae_config(payload)
        if flat_config is None:
            raise ValueError('checkpoint has weights but no reconstructable VAE config. Expected model_config.diffusion_config + model_config.vae, or a flat AllAtomStructVAE.get_config() mapping under config/model_config.')
        _validate_flat_widths(flat_config, summary)
        model = AllAtomStructVAE(**flat_config)
        layout = 'flat'
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError('checkpoint weights do not exactly match the reconstructed all-atom VAE; do not validate with guessed widths or a different Protenix base') from exc
    model = model.to(device)
    model.eval()
    train_config = payload.get('train_config', {}) if isinstance(payload, Mapping) else {}
    if not isinstance(train_config, Mapping):
        train_config = {}
    training_coordinate_frame = payload.get('coordinate_frame', train_config.get('coordinate_frame', 'raw')) if isinstance(payload, Mapping) else 'raw'
    training_frame_version = payload.get('frame_version', train_config.get('frame_version')) if isinstance(payload, Mapping) else None
    if training_frame_version is None and training_coordinate_frame == 'raw':
        training_frame_version = frame_version_for_mode('raw')
    training_anchor_version = payload.get('anchor_selection_version', train_config.get('anchor_selection_version')) if isinstance(payload, Mapping) else None
    if training_anchor_version is None and training_coordinate_frame == 'raw':
        training_anchor_version = anchor_selection_version_for_mode('raw')
    metadata = {'path': str(checkpoint_path), 'step': payload.get('step') if isinstance(payload, Mapping) else None, 'config_layout': layout, 'config_source': config_source, 'architecture': _plain_tree(model.get_config()), 'parameters': int(sum((parameter.numel() for parameter in model.parameters()))), 'training_coordinate_frame': training_coordinate_frame, 'training_frame_version': training_frame_version, 'training_anchor_selection_version': training_anchor_version, 'joint_rotation_augmentation': train_config.get('joint_rotation_augmentation'), 'rotation_augmentation_probability': train_config.get('rotation_augmentation_probability')}
    return (model, metadata)

def native_null_template_settings(model):
    """只修正原生模板stack的checkpoint分组；权重与其余stack保持不变。"""
    template = getattr(model.native, 'template_embedder', None)
    blocks = getattr(model, '_native_template_blocks', None)
    if template is None or blocks != 2 or len(template.pairformer_stack.blocks) != blocks:
        raise ValueError('The native_null contract requires the complete two-layer v1.0.0 template module')
    if any((p.requires_grad for p in template.parameters())):
        raise ValueError('Template parameters must be frozen under this checkpoint contract')
    stack = template.pairformer_stack
    requested = getattr(model, '_template_requested_blocks_per_ckpt', stack.blocks_per_ckpt)
    if requested is not None and (not isinstance(requested, int) or requested < 1):
        raise ValueError('Native template checkpoint grouping must be a positive integer or None')
    model._template_requested_blocks_per_ckpt = requested
    effective = None if requested is None else min(requested, blocks)
    stack.blocks_per_ckpt = effective
    return dict(mode='native_null', real_templates=False, n_blocks=blocks, requested_blocks_per_ckpt=requested, effective_blocks_per_ckpt=effective, checkpoint_policy='template_stack_clamp_only_v1')

def validate_null_templates(features):
    """空模板仍是网络输入；禁止将真实结构或缺失特征误算为此对照。"""
    required = ('template_distogram', 'template_pseudo_beta_mask', 'template_unit_vector', 'template_backbone_frame_mask')
    if 'template_aatype' not in features or any((k not in features for k in required)):
        raise ValueError('Null-template input is missing native template features')
    keys = set(required) | {k for k in features if k.startswith('template_') and 'mask' in k}
    counts = {}
    for key in sorted(keys):
        value = features[key]
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or value.count_nonzero():
            raise ValueError(f'Null-template inputs must have finite, zero template geometry/masks: {key}')
        counts[key] = 0
    return counts

def reset_template_probe_rng(seed):
    """独立配对随机流；只供显式模板对照，不更改旧采样随机数。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def checkpoint_state_names(model):
    trained = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name for name in model.state_dict() if name in trained or name.startswith('generator.') or name in {'latent_mean', 'latent_std'}}

def generate_samples_chunked(model, batch, online, *, reference_present, samples, chunk_size, sampler='ode_pure'):
    """Generate one seeded sample set in bounded-memory chunks.

    Draw the complete latent and atom-noise tensors before chunking.  Hence a
    cell is still one sampling seed with exactly ``samples`` draws; chunking is
    only an execution detail and does not turn 1x250 into multiple seeds.
    """
    if samples < 1 or chunk_size < 1:
        raise ValueError('samples and sample chunk size must be positive')
    device = online[1].device
    latent_noise = torch.randn(samples, len(batch['token_mask']), len(model.latent_mean), device=device)
    atom_noise = torch.randn(samples, len(batch['atom_mask']), 3, device=device)
    pieces = {}
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        output = model.generate(batch, online, reference_present=reference_present, noise=latent_noise[start:stop], atom_noise=atom_noise[start:stop], **{'sampler': sampler} if sampler != 'ode_pure' else {})
        for key, value in output.items():
            pieces.setdefault(key, []).append(value)
    return {key: torch.cat(value, dim=0) for key, value in pieces.items()}

def _annotation(atom_array: Any, names: Sequence[str], default: Any=None) -> np.ndarray:
    """Read the first available AtomArray annotation, with a safe fallback."""
    annotations = getattr(atom_array, '_annot', {})
    for name in names:
        if name in annotations or hasattr(atom_array, name):
            return np.asarray(getattr(atom_array, name))
    if default is None:
        raise KeyError(f'atom array has none of annotations {tuple(names)}')
    n_atom = len(atom_array)
    if np.isscalar(default):
        return np.full(n_atom, default)
    result = np.asarray(default)
    if result.shape[0] != n_atom:
        raise ValueError('fallback atom annotation has the wrong length')
    return result

def build_reference_identities(atom_array: Any) -> list[dict[str, Any]]:
    """Build serializable, entity-aware identities in Protenix atom order."""
    asym = _annotation(atom_array, ('label_asym_id', 'chain_id'), '')
    residue = _annotation(atom_array, ('res_id', 'label_seq_id'), -1)
    atom_name = _annotation(atom_array, ('atom_name',), '')
    entity = _annotation(atom_array, ('entity_id_int', 'label_entity_id'), -1)
    symmetry = _annotation(atom_array, ('sym_id_int', 'sym_id'), -1)
    component = _annotation(atom_array, ('cano_seq_resname', 'res_name'), 'UNK')
    insertion = _annotation(atom_array, ('ins_code',), '')
    identities: list[dict[str, Any]] = []
    for index in range(len(atom_array)):
        try:
            residue_index: Any = int(residue[index])
        except (TypeError, ValueError):
            residue_index = str(residue[index]).strip()
        try:
            entity_id: Any = int(entity[index])
        except (TypeError, ValueError):
            entity_id = str(entity[index]).strip()
        try:
            sym_id: Any = int(symmetry[index])
        except (TypeError, ValueError):
            sym_id = str(symmetry[index]).strip()
        identities.append({'asym_id': str(asym[index]).strip(), 'residue_index': residue_index, 'atom_name': str(atom_name[index]).strip(), 'entity_id': entity_id, 'sym_id': sym_id, 'component_id': str(component[index]).strip().upper(), 'insertion_code': str(insertion[index]).strip()})
    build_atom_identity_mapping(identities, identities, require_all_reference=True, require_all_target=True)
    return identities

def _stable_json_sha256(value: Any) -> str:
    serialized = json.dumps(_plain_tree(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def reference_token_centroids(reference_coordinate: torch.Tensor, reference_coordinate_mask: torch.Tensor, atom_to_token_idx: torch.Tensor, n_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool canonical reference atoms to one masked coordinate per token."""
    coordinate = reference_coordinate.detach().float()
    mask = reference_coordinate_mask.detach().bool()
    atom_to_token = atom_to_token_idx.detach().long()
    if coordinate.ndim != 2 or coordinate.shape[-1] != 3:
        raise ValueError('reference_coordinate must have shape [N_atom,3]')
    if mask.shape != (coordinate.shape[0],):
        raise ValueError('reference_coordinate_mask must have shape [N_atom]')
    if atom_to_token.shape != (coordinate.shape[0],):
        raise ValueError('atom_to_token_idx must have shape [N_atom]')
    sums = coordinate.new_zeros((n_token, 3))
    counts = coordinate.new_zeros((n_token,))
    weights = mask.to(coordinate.dtype)
    sums.index_add_(0, atom_to_token, coordinate * weights[:, None])
    counts.index_add_(0, atom_to_token, weights)
    token_mask = counts > 0
    centroids = sums / counts.clamp(min=1.0)[:, None]
    centroids = torch.where(token_mask[:, None], centroids, torch.zeros_like(centroids))
    return (centroids.cpu(), token_mask.cpu())

def write_cif(path, stack, ids):
    """不对齐、不重排；写出后重新解析全部模型以验证身份和坐标。"""
    import numpy as np
    from biotite.structure.io import pdbx
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, stack, data_block='oridynamic_legacy', include_bonds=False)
    entities = np.array([int(i['entity_id']) + 1 for i in ids])
    cif.block['atom_site']['label_entity_id'] = np.tile(entities, len(stack))
    require(not path.exists(), f'Refusing to overwrite CIF: {path}')
    temp = path.with_suffix('.partial.cif')
    require(not temp.exists(), f'An incomplete export exists; inspect it first: {temp}')
    cif.write(temp)
    back_file = pdbx.CIFFile.read(temp)
    back = pdbx.get_structure(back_file, model=None, include_bonds=False, use_author_fields=False)
    require(back.coord.shape == stack.coord.shape, 'CIF model/atom counts changed')
    for key in ('chain_id', 'res_id', 'res_name', 'atom_name', 'ins_code', 'element', 'hetero'):
        require(np.array_equal(back.get_annotation(key), stack.get_annotation(key)), f'CIF identity changed: {key}')
    error = float(np.max(np.abs(back.coord - stack.coord)))
    require(error <= 0.00051, f'CIF coordinate error exceeds tolerance: {error}')
    require(np.array_equal(back_file.block['atom_site']['label_entity_id'].as_array(int), np.tile(entities, len(stack))), 'CIF entity identities changed')
    os.replace(temp, path)
    os.chmod(path, 420)
    return error
