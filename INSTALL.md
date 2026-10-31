# 安装与Docker运行

当前支持`python -m pip install -e .`或构建wheel后安装，公开命令为
`oridynamic sample`、`oridynamic msa`、`oridynamic doctor`；完整快速用法见[README](README.md)。
Python3.11/CUDA12.6配方不随参考项目升级。依赖安装在独立环境，未发布PyPI或权重下载服务。
用`ORIDYNAMIC_PATHS`和`ORIDYNAMIC_ASSETS`设置用户资源，或显式传`--paths/--assets-root`。

只需要本目录代码、推理专用权重、公共CCD及MSA资源，不需要研究数据集或manifest。
FASTA用于独立单体；PL/PP用[原生JSON入口](docs/complex_inputs.md)，一次可提交多种目标。
不要安装另一个Protenix仓库或将其他worktree加入PYTHONPATH。

## 常驻批量入口

依赖沿用本文件下方已验证环境，不照搬BioEmu2的Python/Torch版本。
从本仓库执行`python -m oridynamic sample`，或安装后任意目录执行`oridynamic sample`；无需其他仓库或训练代码。
已有MSA、本地搜索、可选结构用法见[批量推理](docs/batch_inference.md)；
在线只提供[用户操作教程](docs/msa_online.md)。

新构建的模型Docker默认使用统一入口；不改变已经存在的镜像或容器。批量运行：

```bash
mkdir batch_output
docker run --rm --network none --gpus device=0 \
  -v /absolute/path/weights:/weights:ro -v /absolute/path/assets:/assets:ro \
  -v /absolute/path/user_inputs:/input:ro -v "$PWD/batch_output:/output" \
  oridynamic-inference:local sample \
  --paths /opt/oridynamic/configs/container_paths.example.json --assets-root /assets \
  --fasta /input/sequences.fasta --msa-dir /input/msas --out /output/run
```

模型镜像不内置大库或搜索器。可在独立搜索Docker先生成MSA；若使用一条命令自动本地搜索，
推理环境中须能执行ColabFold/MMseqs2，并挂载已有数据库。新镜像构建仍待实际验收。

## 1. 推理环境

参考验收环境：Linux x86_64、Python3.11、PyTorch2.7.1+cu126。
下面安装到独立新环境，不升级正在训练的环境：

```bash
conda create -y -p "$PWD/.envs/ptx" python=3.11
conda run -p "$PWD/.envs/ptx" python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
conda run -p "$PWD/.envs/ptx" python -m pip install -r requirements-inference.txt
conda run -p "$PWD/.envs/ptx" python -m pip install --no-deps -e .
```

或使用提供的模型Docker配方，仅COPY本目录源码，不克隆研究仓库：

```bash
docker build -f docker/Dockerfile.inference -t oridynamic-inference:local .
```

默认Torch后端不编译fused LayerNorm；cuequivariance仅在已安装兼容扩展时显式启用。
新镜像build/干净安装尚待实际验收，不能把现有PTX镜像验证当作新镜像验证。

## 2. 权重与CCD

资源目录：

```text
weights/
  protenix_base_default_v1.0.0.pt
  all_atom_struct_vae_step0100000.pt
  joint_step0011500.pt
  weights_manifest.json
assets/common/
  components.cif
  components.cif.rdkit_mol.pkl
```

三份文件须为配套的推理专用导出包，不是同名旧训练checkpoint；程序严格核对schema、
base/codec文件SHA、架构及Joint参数集合，不提供不受限pickle回退。
权重只保留必要网络、推理配置及latent归一化，没有训练状态或研究数据。

公共CCD按[Protenix官方项目](https://github.com/bytedance/Protenix)对应版本准备，
包括配套RDKit资源；仅使用可信的pickle。代码不自动下载。
若资源不在默认位置，用`--paths my_paths.json --assets-root /path/to/assets`；
路径JSON格式参考`configs/portable_paths.example.json`，相对路径基于JSON所在目录。
权重和各公共资源的许可需单独确认，当前未上传公开权重或镜像。

## 3. 快速检查与GPU测试

```bash
python -B -m unittest discover -s tests -v
python -B predict.py --fasta examples/monomer.fasta --single-sequence \
  --check --out checks/new_install
CUDA_VISIBLE_DEVICES=0 python -B -u predict.py \
  --fasta examples/monomer.fasta --single-sequence --anchor-candidates 1 \
  --seeds 0,1 --samples-per-seed 2 --sample-chunk-size 1 --out runs/quick_smoke_v1
```

在新模型镜像里可以离线试跑（选择空闲GPU、创建新的输出目录）：

```bash
mkdir new_ensemble_output
docker run --rm --network none --gpus device=0 \
  -v /absolute/path/weights:/weights:ro -v /absolute/path/assets:/assets:ro \
  -v "$PWD/new_ensemble_output:/output" oridynamic-inference:local \
    --paths /opt/oridynamic/configs/container_paths.example.json --assets-root /assets \
    --fasta /opt/oridynamic/examples/monomer.fasta --single-sequence --anchor-candidates 1 \
    --seeds 0,1 --samples-per-seed 2 --sample-chunk-size 1 --out /output/run
```

正式任务使用自己的FASTA/MSA挂载，不用query-only smoke代表质量。
默认分块与CPU卸载降低显存压力，但尚无24/48GB卡长度保证。

## 4. 本地MSA搜索：复用ColabFold/MMseqs2

MSA工具/数据库由用户自行准备，本项目不另行开发或提供常驻搜索服务。可以直接使用
公开ColabFold/MMseqs2，或选择现有在线服务；已有A3M直接传`predict.py --msa-a3m`。
下方Docker/薄适配均为可选方法，不要求必须经由我们的搜索包装器。

建议将搜索器与推理环境隔离，避免依赖冲突。宿主机可使用搜索Docker；不要在容器里套Docker：

```bash
docker pull ghcr.io/sokrypton/colabfold:1.6.3-cuda12
MSA_IMAGE=ghcr.io/sokrypton/colabfold:1.6.3-cuda12 MSA_DEVICE=gpu MSA_GPUS=0 \
  bash tools/search_msa_docker.sh input.fasta /absolute/path/colabfold_db new_search_run \
    --msa-threads 16
```

包装器不自动pull；源码/输入/库只读挂载、网络关闭、仅写新输出，记录实际image ID。
CPU搜索改`MSA_DEVICE=cpu`。完成后用`--msa-a3m new_search_run/msa/unpaired.a3m`
传入predict；此路径在模型记录中为provided，搜索记录在邻接的msa/search.json中。

也提供`docker/Dockerfile.msa`，或在新的独立环境安装公开工具：

```bash
python3.11 -m venv .envs/msa
.envs/msa/bin/pip install 'colabfold==1.6.3'
mkdir -p vendor/mmseqs18
curl -fL https://mmseqs.com/archive/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/mmseqs-linux-gpu.tar.gz \
  -o vendor/mmseqs18/mmseqs.tar.gz
tar -xzf vendor/mmseqs18/mmseqs.tar.gz -C vendor/mmseqs18
vendor/mmseqs18/mmseqs/bin/mmseqs version
.envs/msa/bin/colabfold_search --help
```

系统/架构兼容性以[MMseqs2官方说明](https://github.com/soedinglab/MMseqs2#installation)为准。
一次准备公共数据库需大量磁盘/RAM，不是轻量操作；参照
[ColabFold官方本地库说明](https://github.com/sokrypton/ColabFold#generating-msas-for-large-scale-structurecomplex-predictions)，
不要对共享生产库原地重建或运行makepaddedseqdb。现有库可只读复用。
默认前缀uniref30_2302_db、colabfold_envdb_202108_db，可分别通过
`--msa-uniref-db`/`--msa-env-db`覆盖。db2是模板库，本入口不用。

搜索预检（不搜索，不写文件）：

```bash
python -B local_msa.py --fasta examples/monomer.fasta --out runs/msa_check \
  --msa-local-db /absolute/path/colabfold_db \
  --msa-search-executable "$PWD/.envs/msa/bin/colabfold_search" \
  --msa-mmseqs "$PWD/vendor/mmseqs18/mmseqs/bin/mmseqs" --check
```

一条命令完成本地搜索→模型的方式见README。默认CPU搜索/16线程，GPU须显式指定；
搜索结束后才加载Torch模型，不启用常驻GPU server、不搜索模板，不修改数据库。
数据库只核验dbtype标记，不伪称全库哈希；MSA首序列和全部比对列验收后才交给模型。
查询失败不会静默改用在线服务或query-only。完整真实大库搜索仍待端到端验收。

四种输入组合的可选计时命令见README第6节；MSA搜索时间与模型推理时间分开报告。
本地和在线结果不能混为同一搜索性能基准，在线包括网络/排队和服务端缓存影响。

## 5. 来源与边界

复用公开Protenix网络/特征、ColabFold/MMseqs2 CLI；未复制其他人的私有搜索调度代码。
保留LICENSE-PTX、SOURCES和源码指纹；各依赖、权重和公共数据库的许可分别遵守。
代码公开不附带研究数据或训练器；正式发布文件见PUBLIC_RELEASE。
用户自己的输入、搜索结果与生成输出只属于其新任务目录，不随代码交付。
