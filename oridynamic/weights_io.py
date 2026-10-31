"""推理资源的受限加载；不反序列化训练状态或自定义Python对象。"""
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


def validate_payload(payload, kind):
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS[kind] or payload.get('schema_version') != SCHEMAS[kind]:
        raise ValueError(f'{kind}不是本版推理专用权重；请使用已导出的资源包，不能直接传训练checkpoint')
    state = payload['trainable_state' if kind == 'joint' else 'model']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f'{kind}权重必须是非空的名称→Tensor映射')
    if kind == 'joint':
        if set(payload['metadata']) != META_KEYS or set(payload['joint_config']) != JOINT_KEYS:
            raise ValueError('Joint元数据包含未知字段或缺少推理设置')
        if payload['metadata']['native_inference_profile'] != PROFILE:
            raise ValueError('未知原生推理配置')
        if payload['joint_config']['activation_checkpoint']:
            raise ValueError('推理不启用梯度checkpoint')
    return payload


def load_weights(path, kind):
    # 必须显式weights_only=True；不自动回退到不受限pickle。
    return validate_payload(torch.load(Path(path), map_location='cpu', weights_only=True, mmap=False), kind)
