# 五个小用例：3个单体、1个PL、1个PP

输入在[five_cases.json](five_cases.json)，只包含序列、链组成及一个公共CCD标识。
不包含MD轨迹、frame0、GT坐标、内部MSA/缓存或研究绝对路径。

| 用例 | 类型 | 选择依据 |
|---|---|---|
| [6jv8_A](https://www.rcsb.org/structure/6JV8) | 单体 | 较短；历史生成的逐体系RMSF形状相关较好 |
| [7n0j_E](https://www.rcsb.org/structure/7N0J) | 单体 | 较短；历史生成的逐体系RMSF形状相关较好 |
| [6tly_A](https://www.rcsb.org/structure/6TLY) | 单体 | 较短；历史幅度比更接近1，避免只按相关性选例 |
| [5T3N](https://www.rcsb.org/structure/5T3N) | PL | 153 aa蛋白＋CCD_75G；历史配体位姿误差较小 |
| A00C3（DynaPPI公开体系编号） | PP | 两条蛋白链，总317 aa；历史PP中较小且RMSF形状相关较好 |

这些是**挑选的演示案例，不是随机测试集或性能基准**。序列采用之前处理的体系区段，
不一定等于PDB完整SEQRES，标签/末端也保留；不能仅按PDB编号替换成不同的全长序列。
5T3N改用公共CCD_75G化学参考，不能把旧MD化学特征下的成绩当作本例复现结果。
历史筛选参考的online-codec/32候选配置不同于release固定C0/4候选默认；新完整GPU输出尚待验收。
PP例是公共DynaPPI的序列组成，不要求用户下载DynaPPI轨迹。

## 正常推理（本地MSA）

先按INSTALL配置`ORIDYNAMIC_PATHS`、`ORIDYNAMIC_ASSETS`及本地搜索器/公共数据库：

```bash
oridynamic sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0 --samples-per-seed 4 --sample-chunk-size 1 \
  --out runs/five_cases
```

已有MSA时，在每个`proteinChain`中填写`unpairedMsaPath`，去掉`--msa-local-db`。
每体系4个最终构象，共20个；默认4个PTX候选选一个anchor，候选数不计入最终ensemble样本数。

## 离线快速烟测

没有搜索库且不能联网时，可显式使用query-only：

```bash
oridynamic sample --input-json examples/five_cases.json --single-sequence \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_smoke
```

这是接口/完整输出烟测，不代表正常MSA条件下的质量。可加`--plan`只读检查；
该命令不做模型前向，也不能替代CCD真实特征检查。推理请使用空闲GPU，不占用在跑训练的卡。
给定anchor的PL/PP格式及逐链MSA说明见[复合物输入](../docs/complex_inputs.md)。
