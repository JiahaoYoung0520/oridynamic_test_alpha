# 用户自行准备在线MSA

新批量入口不请求远端MSA服务。在线搜索是用户独立操作，会把序列发送给第三方；
敏感序列应使用本地数据库。机器不能联网时不要运行以下步骤。

按[ColabFold官方说明](https://github.com/sokrypton/ColabFold#generating-msas-for-small-scale-local-structurecomplex-predictions-using-the-msa-server)
安装兼容的ColabFold环境，在可联网机器执行：

```bash
colabfold_batch sequences.fasta online_msas --msa-only
```

这里只取得MSA，不调用OriDynamic、不预测本项目anchor。遵守服务条款与限流，
不要对公共服务运行大规模并行请求。也可使用官方Colab界面取得A3M。

把产生的A3M复制到推理机器，核对每条query和FASTA完全一致。若输出文件名不是FASTA
标题首词，使用`request.json`逐项明确`msa_a3m`路径，而不是猜测编号。
本地推理：

```bash
python -B -u -m oridynamic --manifest request.json \
  --seeds 0,1 --samples-per-seed 8 --out runs/from_online_msas_v1
```

在线准备时间不计入本地推理计时。当前目录保留旧`predict.py`的在线兼容入口以免影响
既有任务；推荐新入口仅支持已有MSA、本地搜索和显式query-only，不接受在线搜索开关。
