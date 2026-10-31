# OriDynamic

**从蛋白序列、可选配体/伙伴链及初始结构生成全原子构象系综。仅推理。**

[安装](INSTALL.md) · [快速开始](#快速开始) · [批量输入](docs/batch_inference.md) · [模型与限制](docs/model_card.md) · [发布检查](docs/releasing.md)

用户准备序列，程序调用本地MSA搜索（或读取已有A3M），自动预测结构anchor，再生成系综。
也可以直接提供初始结构。无需研究数据、MD轨迹、GT/frame0或其他代码仓库。

## 功能与边界

- 单体、PL（蛋白—配体）、PP（蛋白—蛋白）及混合批量输入；常驻模型先准备全部anchor，再生成全部系综。
- 已有MSA或本地ColabFold/MMseqs2；在线搜索仅提供[教程](docs/msa_online.md)，新入口不隐式联网。
- 可选PDB/mmCIF初始结构；不提供时用原生PTX预测候选，按置信度选top1，无GT选优。
- 用户控制seed、每seed样本数及分块；输出anchor、proposal、refined ensemble CIF与计时。
- 默认保守分块和分阶段CPU卸载；尚无任意长度/24GB显存保证。
- PL/PP使用原生PTX JSON；多条FASTA仍表示独立目标，不是一个复合物。复合物CPU特征/身份检查与完整GPU质量验收分开。
- 不包含训练入口、研究数据或内部评价结果；模型与采样协议见[模型说明](docs/model_card.md)。

## 安装

当前验证环境为Linux、Python3.11、PyTorch2.7.1+CUDA12.6、NVIDIA GPU。
不要在正在训练的环境中安装，也不要安装另一份Protenix。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
oridynamic --help
```

源码目录运行也可用`python -m oridynamic`；安装后可从任意工作目录使用`oridynamic`。
权重及CCD资源独立于Git，**需要另行准备**。当前未提供公开下载地址，不自动下载或要求训练数据。
按[安装说明](INSTALL.md)准备推理专用资源，配置：

```bash
export ORIDYNAMIC_PATHS=/absolute/path/paths.json
export ORIDYNAMIC_ASSETS=/absolute/path/ptx_assets
oridynamic doctor
```

`paths.json`格式见`configs/portable_paths.example.json`；相对权重路径基于该JSON文件。
`doctor`默认不加载Torch/权重、不写文件；显式`--check-device`才查询CUDA。
`preflight_only`不代表已完成模型前向或完整权重验证。

## 快速开始

已有MSA，自动预测anchor；2个seed各8个样本，总共16个构象：

```bash
oridynamic sample --fasta protein.fasta --msa-a3m protein.a3m \
  --seeds 0,1 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/protein
```

提供初始结构，跳过PTX anchor预测：

```bash
oridynamic sample --fasta protein.fasta --msa-a3m protein.a3m \
  --initial-structure initial.cif --structure-chain A \
  --seeds 0,1 --samples-per-seed 8 --out runs/with_structure
```

初始结构须匹配序列和完整重原子；它是decoder anchor，不是trunk模板或DiT-reference开关。
默认候选4个，每个独立seed；选择过程可在代码和实际运行记录中查看。

## 只提供序列：本地MSA

先按[本地搜索安装说明](INSTALL.md#4-本地msa搜索复用colabfoldmmseqs2)安装搜索器和公共数据库：

```bash
oridynamic sample --fasta protein.fasta \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0,1 --samples-per-seed 8 --out runs/from_sequence
```

这复用现有ColabFold/MMseqs2，不另造搜索器。可指定搜索可执行程序路径；支持独立搜索：

```bash
oridynamic msa --fasta protein.fasta --msa-local-db /data/colabfold_db --out msas/protein
```

搜索结果、临时缓存只写用户的新输出目录；数据库只读。缺工具/缺库明确失败，不静默联网或降级。
`--single-sequence`仅为query-only离线smoke，**不是正常MSA搜索的替代质量协议**。

## PL、PP与五个真实小用例

使用PTX原生`proteinChain`/`ligand` JSON，同一任务的全部链/配体一起生成；每条蛋白entity
可提供MSA，或逐entity本地搜索。配体CCD/SMILES/MOL解析直接复用PTX，不另造化学图。

```bash
oridynamic sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0 --samples-per-seed 4 --sample-chunk-size 1 --out runs/five_cases
```

五个示例为3单体＋1PL＋1PP，按小规模/历史表现挑选，**不是无偏性能基准**。
见[用例及离线烟测](examples/README.md)、[复合物/逐链MSA/用户anchor格式](docs/complex_inputs.md)。
仅提供序列与CCD，不发布MD轨迹/GT坐标；没有anchor时预测完整复合物。

## 批量输入

```bash
oridynamic sample --fasta sequences.fasta --msa-dir msas --structure-dir anchors \
  --seeds 0,1 --samples-per-seed 8 --out runs/batch
```

文件名对应FASTA标题首词，例如`msas/protein_a.a3m`、`anchors/protein_a.cif`。
没有anchor的目标自动预测；可加本地库只搜索缺失MSA。复杂路径用通用JSON输入，不需要数据集manifest。
模型一次加载、每目标仍独立前向。更多见[输入、内存与生命周期](docs/batch_inference.md)。

## 输出与计时

```text
runs/batch/
  summary.json              # 总览；推理墙钟不含MSA搜索
  timing.json               # 全阶段墙钟，搜索单列
  progress.json
  targets/<name>/
    anchor.cif
    proposals.cif
    ensemble.cif
    run_metadata.json
    completion.json
```

每个CIF含全部seed合并的构象；总数 = seed数量 × samples-per-seed。
chunk改变一次处理量，不改变总数；不同chunk不承诺SDE逐位相同。
只允许新输出目录，不自动覆盖/恢复失败推理。运行细节、缓存和计时定义见批量文档。

## Docker与检查

```bash
docker build -f docker/Dockerfile.inference -t oridynamic-inference:local .
docker run --rm oridynamic-inference:local --help
python -B -m unittest discover -s tests -v
python -B tools/release_bundle.py check
```

GPU及权重/CCD/用户输入挂载见[INSTALL.md](INSTALL.md)。
CPU测试、模型镜像build、真实GPU、本地大库、小显存是不同验收项，不相互代替。
本轮打包/CLI验证状态见[发布清单](docs/releasing.md)。

## 来源与许可

模型实现复用Protenix，保留上游版权及[LICENSE-PTX](LICENSE-PTX)，详见[SOURCES.md](SOURCES.md)。
仓库组织借鉴[BioEmu2](https://github.com/microsoft/bioemu2)与[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)，
不复制它们的模型、权重、训练代码或许可证。

**当前是本地发布候选，非已发布的开源授权。** 本地新增代码和权重的最终许可、权重托管地址、
正式仓库URL及论文引用信息须由维护者确认；不会自动上传代码、权重或数据。
