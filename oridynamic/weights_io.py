"""推理资源的受限加载；不反序列化训练状态或自定义Python对象。"""
import hashlib
from pathlib import Path
import torch

SCHEMAS = {k: f'oridynamic_{k}_inference_v1' for k in ('native', 'codec', 'joint')}
TOP_KEYS = {
    'native': {'schema_version', 'source_sha256', 'model'},
    'codec': {'schema_version', 'source_sha256', 'step', 'model', 'model_config',
              'coordinate_frame', 'frame_version', 'anchor_selection_version'},
    'joint': {'schema_version', 'source_sha256', 'step', 'metadata', 'joint_config', 'trainable_state'},
}
META_KEYS = {'condition_contract', 'codec_identity', 'stageb_model_config', 'stageb_backbone',
             'lora', 'online_template_execution_contract', 'online_template_settings', 'native_inference_profile'}
JOINT_KEYS = {'refiner_precision', 'samples', 'fm_steps', 'refiner_steps', 'sigma_init',
              'sigma_min', 'n_cycle', 'activation_checkpoint', 'chunk_size'}
PROFILE = 'ptx_base_v1_msa512_native_null_v1'
OFFICIAL_NATIVE_SHA256 = '2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04'
LEGACY_NATIVE_SHA256 = 'b0d59aa3f6d8bb6b051486a04af8d1198af2b236f05e438c5bab593f0161f107'


def native_compatibility_hash(file_sha256):
    # 精确核实过参数逐张量相同的两种序列化；不接受任意来源声明。
    return LEGACY_NATIVE_SHA256 if file_sha256 == OFFICIAL_NATIVE_SHA256 else file_sha256



def validate_payload(payload, kind):
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS[kind] or payload.get('schema_version') != SCHEMAS[kind]:
        raise ValueError(f'{kind} is not a supported inference checkpoint; use the matching release exports or the exact official Protenix base file')
    state = payload['trainable_state' if kind == 'joint' else 'model']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f'{kind} weights must be a nonempty name-to-Tensor mapping')
    if kind == 'joint':
        if set(payload['metadata']) != META_KEYS or set(payload['joint_config']) != JOINT_KEYS:
            raise ValueError('Joint metadata contains unknown fields or missing inference settings')
        if payload['metadata']['native_inference_profile'] != PROFILE:
            raise ValueError('Unknown native inference profile')
        if payload['joint_config']['activation_checkpoint']:
            raise ValueError('Gradient checkpointing must be disabled for inference')
    return payload


def load_weights(path, kind):
    # 必须显式 weights_only=True；不自动回退到不受限 pickle。
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if kind == 'native' else None
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=True, mmap=False)
    if kind == 'native' and digest == OFFICIAL_NATIVE_SHA256:
        if not isinstance(payload, dict) or set(payload) != {'model'}:
            raise ValueError('Unexpected official Protenix v1.0.0 checkpoint structure')
        payload = dict(schema_version=SCHEMAS['native'],
                       source_sha256=OFFICIAL_NATIVE_SHA256, model=payload['model'])
    return validate_payload(payload, kind)
