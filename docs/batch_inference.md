# 常驻批量推理

FASTA/旧manifest接受N条**相互独立的单蛋白序列**，不是把N条链拼成一个复合物。
新增`--input-json`接受PTX原生单体/PL/PP任务列表；一个任务内的全部entity共同生成。
见[复合物格式](complex_inputs.md)，下文FASTA用法保持不变。
复用现有模型、严格权重加载、原子身份映射和采样算子，没有引入训练代码。

## 输入与运行

在本发行目录、已安装依赖的环境执行；无需其他代码仓库：

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-dir msas \
  --seeds 0,1 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/batch_v1
```

FASTA标题首词为唯一目标名；MSA对应`msas/<目标名>.a3m`。
可加`--structure-dir anchors`读取同名`.cif/.mmcif/.pdb`，没有结构的目标由PTX预测。
有结构也仍需MSA；结构作为decoder anchor，不是trunk模板，也不替换化学参考`ref_pos`。
初始结构须含同序列、完整重原子，缺原子/残基会明确报错，不静默补GT。

若路径/链名不规则，改用`--manifest request.json`，JSON相对路径基于该JSON目录：

```json
{
  "schema": "oridynamic_batch_request_v1",
  "targets": [
    {"name": "protein_a", "sequence": "ACDEFG", "msa_a3m": "msas/a.a3m"},
    {"name": "protein_b", "sequence": "ACDEFGH", "initial_structure": "anchors/b.cif", "structure_chain": "A"}
  ]
}
```

第二项未提供MSA，需同时指定下述本地数据库；不是使用零MSA。公共接口只读这些输入字段，
不读取轨迹、MD、GT或研究缓存。兼容已有`oridynamic_offline_batch_v1`序列/MSA清单。
`--limit 2`先检查前两项；`--plan`只读检查输入/资源路径，不搜索、不导入Torch、不写文件。

## 本地MSA与在线教程

提供已有数据库和ColabFold/MMseqs2可执行程序即可一条命令搜索后推理：

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-local-db /data/colabfold_db \
  --msa-search-executable /opt/colabfold/bin/colabfold_search \
  --msa-mmseqs /opt/mmseqs/bin/mmseqs --msa-threads 32 \
  --seeds 0,1 --samples-per-seed 8 --out runs/local_batch_v1
```

允许同时提供`--msa-dir`：已提供的MSA直接使用，只搜索缺失项。复用`local_msa.py`，
本轮仍逐目标调用本地搜索器，没有声称数据库只加载一次。搜索完成后才加载模型；
搜索临时文件位于新任务目录，公共数据库只读；失败不会回退在线或query-only。
本地安装、CPU/GPU搜索及独立搜索Docker见[INSTALL](../INSTALL.md)。
在线搜索不集成到新批量入口，见[在线MSA教程](msa_online.md)，自行取得A3M后传入。
`--single-sequence`只是显式query-only smoke选项，不代表真实MSA搜索或同等质量。

## 两阶段生命周期与显存

1. 校验输入，准备所有MSA；不加载模型。
2. 一次加载base PTX，为每个目标计算冻结上下文C0，并预测/映射anchor。
3. 全部anchor/C0落盘后，一次加载AE、DiT及Joint状态，向base对象注入LoRA。
4. 遍历目标，计算该目标在线LoRA上下文，用同一anchor生成各seed/sample。

必须先完成第2步：`build_joint()`会修改native对象，不能随后拿它冒充base PTX预测anchor。
每个目标仍需要自己的网络前向；“一次加载”不是把不同序列共享一个结构，也不是一次forward。
默认每个目标预测4个候选，按原生ranking score选top1，无GT筛选；所有最终样本复用它。
设置`--anchor-candidates 1`才是每目标只预测一个候选，改变候选预算，不是本次提速默认。

模型保留在进程中，按阶段CPU/GPU卸载；每个目标结束释放大中间张量，不将N个蛋白同时塞入显存。
`--sample-chunk-size 1`和`--attention-chunk-size 32`为保守默认，不承诺任意长蛋白都可在小显存卡运行。
冻结上下文为O(长度²)磁盘缓存，需预留输出盘空间；不新增跨任务隐式共享缓存。

## 产物与计时

- `targets/<name>/anchor.cif`：原始精度坐标另在`prepared.pt`，不从三位小数CIF回读用于本次采样。
- `prepared.pt`及`prepared.json`：本次输入、base C0、原子身份、anchor；受限张量加载和SHA/合同检查。
- `ensemble.cif`、`proposals.cif`、`seed_*.pt`、`run_metadata.json`、`completion.json`：每目标最终产物。
- `progress.json`：正在执行的anchor/ensemble阶段和目标序号；终端实时阶段计时。
- `summary.json`：一次性初始化耗时、各目标anchor/生成阶段耗时和模型加载计数。
- `timing.json`：完整墙钟含搜索；`inference_wall_seconds_excluding_msa`从搜索结束后开始，包含加载/前向/写盘，**不含MSA搜索时间**。
- `anchor_phase/timing.json`与`ensemble_phase/timing.json`：各自从目标开始处理计时，不把等其他目标的间隔算作该目标运行时间。

根计时中`ensemble_seed`是`ensemble_target`的子阶段，汇总时不能两者相加。
当前失败即停止，保留既有成功产物；拒绝覆盖/自动续跑。已有anchor可通过新manifest显式复用。
缓存写入会将NumPy标量元数据转换为Python基本类型，读取继续使用`weights_only=True`。
旧版若因`numpy._core.multiarray.scalar`报错，修复后使用新的`--out`目录重跑；
原失败目录保留，不自动迁移缓存，也不关闭受限加载。此前生成的普通字符串/张量缓存格式不变。
旧`benchmark_batch.py`保持逐目标独立进程，用作冷启动对照；正在执行的旧任务不自动切换。

## 验证边界

CPU测试覆盖生命周期、混合MSA路由、身份/缓存、真实微型codec/DiT/diffusion重复生成；
正式GPU批量速度、同输入单目标/批量数值比较、真实大库搜索及小显存验收仍需执行。
不将CPU模拟调度或模型加载计数称为已获得GPU加速倍数。

接口组织借鉴[BioEmu2](https://github.com/microsoft/bioemu2)的模型复用、命令行与教程分离，
但没有复制其实现、训练流程或采样协议，也不沿用其许可证为本项目重新授权。
