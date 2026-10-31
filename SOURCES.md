# 推理源码来源与裁剪范围

2026-10-31：借鉴[BioEmu2](https://github.com/microsoft/bioemu2)和
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)的Python包/CLI、教程、资源与容器分层。
新增打包与薄CLI调度，单文件输入复用已有batch目标路由，doctor不加载权重；模型/采样源码未改。
没有复制两者源码、模型、安装版本或许可证。原始来源指纹保留，当前文件SHA以code_manifest.json为准。

新增`batch_predict.py`/`batch_runtime.py`/`oridynamic/__main__.py`为常驻批量调度：
复用本目录`inference_runtime.py`、`local_msa.py`及`runtime_utils.py`；不导入研究仓库。
顺序为全部base anchor/C0→一次Joint加载→全部系综，模型/采样算子未改。
参考[BioEmu2发布说明](https://github.com/microsoft/bioemu2)中的模型复用和文档组织，
没有复制BioEmu2源码、权重或自动采用其MIT授权。

2026-10-30：从已核对的主项目快照机械摘取必要实现，保存在本目录；不在运行时
调用主项目或training/analysis worktree。具体原文件SHA和发布文件SHA见
`runtime_sources.json`；原路径只用于追溯，复制来源文件不等于声明重新授权。

| 发布位置 | 保留内容 | 不包含内容 |
|---|---|---|
| `protenix/model/protenix.py` | 原生网络与inference分支，保留严格权重键 | 训练loop/forward、训练噪声采样器 |
| `protenix/model/modules/all_atom_struct_vae.py` | encode参考缓存、位移decode、严格权重键 | 训练forward、KL loss、训练初始化辅助 |
| `protenix/model/modules/latent_dit.py` | 既有DiT网络和FM速度预测 | FM训练loop/目标/优化器 |
| `protenix/model/modules/joint_ensemble.py` | 原generate/refine数学、Heun/SDE、双上下文 | replay/native loss、ensemble loss、训练forward |
| `runtime_utils.py` | 从旧runner/scripts摘取严格加载、身份、分块、CIF写出 | 原main、MD读取、训练状态保存 |
| `protenix/data/inference/`等 | 原生JSON/CCD/MSA特征依赖 | 训练dataset、mixture/轨迹适配器 |
| `runner/msa_search.py`等 | 原生在线搜索依赖；本地搜索仍用`local_msa.py` | 训练runner、批量研究实验入口 |
| `configs/inference_defaults.json` | 已验证smoke的推理数值、confidence分箱 | 数据集绝对路径、优化器/scheduler/loss权重 |
| `benchmark_batch.py`、`compare_runs.py` | 通用已有MSA批量测速、同anchor结果差异检查 | 研究数据、搜索或新的模型实现 |

抽取保留必要参数、随机特征buffer和state_dict键，严格恢复base/codec/Joint。
Joint权重自带DiT，不需要独立Stage-B初始化权重。`requires_grad`在加载时仅用于核验
历史checkpoint保存集合，加载完成全模型冻结；`Joint.train(True)`和训练forward拒绝调用。
这不是防止他人自行再训练的保护措施。

默认Torch LayerNorm与历史smoke一致；不携带fused LayerNorm编译链。
原生底层算子可能仍含training状态、autograd或checkpoint通用实现，这是网络计算
依赖，不是AE/Stage-B/Joint训练配方。公开入口接通单体/PL/PP的原生特征与整复合物采样；
CPU特征验收不等于新PL/PP完整GPU质量已验收，边界见docs/model_card.md。

本地搜索调用ColabFold/MMseqs2 CLI，没有复制同事批量调度代码；MSA数据库、CCD、
权重另行提供且只读。PyTorch与各化学/搜索依赖按原许可安装。保留上游`LICENSE-PTX`
和来源文件版权头；本地分支代码、权重、CCD/数据库的正式再分发许可仍需维护者确认。

不读取GT/MD、不按评价结果选择anchor。自动top1、用户结构和历史多anchor实验应分别记录。
