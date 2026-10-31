# 推理模型说明与适用边界

## 当前默认模型

发布候选继续使用已经导出的三份`oridynamic_*_inference_v1`资源：PTX base v1.0.0、
full/c32 AE100k、Joint11500。Joint中包含Stage-B参数，不需独立训练或单独初始化Stage-B。
新做的预测anchor在线LoRA适配500步尚未替换这里的默认模型，也不能把训练checkpoint直接传入。
推理包受限加载、严格核对基础权重SHA和架构；不包含optimizer、scheduler、训练随机状态或数据路径。

## 原有科学协议不变

1. 原生序列/MSA/CCD特征；不给结构时，PTX默认4候选、SDE200/10cycle、原生ranking top1。
2. 固定base上下文C0用于codec；在线LoRA上下文用于DiT和refiner，二者当前仍有区别。
3. anchor定义canonical坐标系；AE解码输出位移，由调用者加回anchor。
4. Stage-B Heun32产生潜变量；PTX refiner SDE20/sigma16输出精修结构。
5. trunk/codec/DiT默认BF16；扩散遵守原生FP32边界，TF32关闭。

本轮只改善打包、入口、输入路由和文档，未调整这些模型、噪声、坐标、采样或筛选设置。
共享代码含原生化学参考`ref_pos`，不等于用户提供的全局anchor。
不给初始结构时依然会预测anchor，不是新无anchor AE；显式DiT-reference仍关闭。
无GT挑选、替换或修补输出，不从MD轨迹初始化。四个anchor候选不代表四个最终构象簇。

## 能力与未验收事项

- 用户入口：FASTA单体或PTX JSON的单体/PL/PP；FASTA不是复合物的多链描述。
- PL/PP复用原生featurizer、化学conformer、MSA与整复合物采样，不删除配体或拆成独立链。
  CPU特征/映射/输出身份与完整GPU质量是不同验收项；新五案例的GPU结果尚未运行。
  旧Joint11500的PP为域外应用，不声称经过四来源重新训练。格式见complex_inputs.md。
- 小模型CPU回归不代表全尺寸GPU推理、真实MSA搜索或低显存通过。
- 保守chunk与阶段卸载已实现；模型分片、多GPU协同、CPU-only生成、任意长度24GB运行未承诺。
- 生成构象不是MD时间轨迹，不声称真实动力学或热力学basin占比。
- CIF B-factor为占位，不应作为pLDDT/RMSF解释。
- 当前没有对外质量保证或新性能结论。用户输入、公开权重及数据库均须有适当使用权限。
