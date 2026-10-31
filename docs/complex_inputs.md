# 单体、PL与PP：同一原生PTX特征入口

使用`oridynamic sample --input-json complex.json ...`。JSON为PTX的任务数组，
`sequences`中每个对象是一个entity，支持`proteinChain`与`ligand`。
同一个任务内的全部链共同进入trunk/anchor预测/Joint/CIF，不拆成多个单体。
FASTA的多记录仍表示多个独立单体，不能用多FASTA假装PP。

## 输入格式

```json
[
  {
    "name": "my_complex",
    "sequences": [
      {"proteinChain": {"id": ["A"], "count": 1, "sequence": "ACDEFGHIK", "unpairedMsaPath": "msas/A.a3m"}},
      {"proteinChain": {"id": ["B"], "count": 1, "sequence": "LMNPQRSTV", "unpairedMsaPath": "msas/B.a3m"}},
      {"ligand": {"id": ["L"], "count": 1, "ligand": "CCD_ATP"}}
    ]
  }
]
```

这里的短序列只演示格式；可运行的真实序列见`examples/five_cases.json`。
每个entity显式给出唯一链ID列表；同源多拷贝可用`id: ["A", "B"], count: 2`。
相对MSA、配体文件和anchor路径基于输入JSON，不基于启动目录。
仅支持标准20种氨基酸；不支持的修饰、核酸、离子、共价跨entity约束明确拒绝，不能静默忽略。

## MSA与化学参考

- 每个蛋白entity可提供`unpairedMsaPath`，以及可选`pairedMsaPath`。二者分别验证query。
- 未给unpaired MSA时，用`--msa-local-db`逐entity搜索；同一entity的多个拷贝共用MSA。
- 本地搜索只产生逐链unpaired结果；不伪造paired MSA。提供paired A3M时直接交给原生PTX配对逻辑。
- `--single-sequence`只允许显式query-only烟测，不是正常输入质量的替代。默认不会隐式联网。
- 配体`CCD_75G`等直接查本地公共CCD；SMILES和`FILE_relative/path.mol`也使用PTX原生解析与conformer。
  本项目不另造CCD检索器、不从MD复制配体构象、不依据GT选化学组分。
- 化学参考`ref_pos`和整复合物decoder anchor是不同输入。化学图、原子名、元素、电荷、ref_mask、
  bond及atom-to-token来自原生featurizer，不因为加入Joint而删去配体。

## 用户提供初始结构

任务对象可增加：

```json
"initial_structure": "anchors/full_complex.cif",
"structure_chains": {"A": "X", "B": "Y", "L": "Z"}
```

映射左侧为输入JSON链ID，右侧为anchor的author链ID。不提供映射时要求同名链。
anchor必须包含所有输入链和全部所需重原子；逐链残基顺序/组分名、原子名和元素严格对应。
CCD配体必须使用同CCD原子名；SMILES/MOL的PTX命名可能不同，不自动猜测同构映射或补原子。
可将本入口生成的`anchor.cif`原样回灌。PP同源链不自动置换；PL不单独拟合/居中配体。
映射完成后统一建立全复合物canonical坐标系，再复用原Joint计算。

不提供初始结构时，PTX从完整蛋白/配体输入共同预测整个复合物，按原生ranking_score选anchor。
最终`anchor.cif`、`proposals.cif`和`ensemble.cif`包含全部链和配体；会做身份/坐标回读核验。

## 验证边界

新增的是复合物输入接线与严格映射，不是重新训练或提高PL/PP质量的新模型。
旧Joint11500主要在三来源训练，PP仍有域外泛化限制。
CPU原生特征与回归结果见本轮维护记录；完整五案例GPU生成与真实本地MSA搜索须另行运行。
不能用CPU检查宣称配体结合位姿、PP界面或动态分布已经验证。
