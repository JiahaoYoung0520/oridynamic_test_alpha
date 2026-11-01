"""按权重合同选择codec上下文；只适配推理，不包含标签重编码或训练入口。"""
from protenix.model.modules.joint_ensemble import JointEnsembleModel

FIXED_CONTEXT = 'fixed_base_codec__online_lora_generator_refiner_v1'
ONLINE_CONTEXT = 'shared_online_lora_codec_generator_refiner__stopgrad_md_labels_v1'


class OnlineCodecJoint(JointEnsembleModel):
    def generate(self, batch, online, **kwargs):
        # 复用当前在线trunk的三组表征；保持原anchor、特征、掩码和调用者batch。
        # 父类在当前调用内构造reference cache，不跨上下文复用缓存。
        s_inputs, single, pair = online
        teacher = dict(batch['codec_context'], s_inputs=s_inputs,
                       s_trunk=single, z_trunk=pair)
        return super().generate(dict(batch, codec_context=teacher), online, **kwargs)


def joint_model_class(contract):
    if contract == FIXED_CONTEXT:
        return JointEnsembleModel
    if contract == ONLINE_CONTEXT:
        return OnlineCodecJoint
    raise ValueError(f'Unknown codec conditioning contract: {contract}')
