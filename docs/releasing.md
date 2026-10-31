# 面向Git仓库的发布检查

## 借鉴的形式

[BioEmu2](https://github.com/microsoft/bioemu2)使用可安装Python包、统一命令、
模型资源与代码分离及分层教程；[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)
另提供明确的安装、配置示例、容器和工作流入口。本项目借鉴交付方式，不移植训练或新采样算法。

## 当前目录结构

```text
pyproject.toml               # pip安装、oridynamic命令、Python/依赖版本
oridynamic/                 # sample / msa / doctor公开入口
batch_predict.py            # 已验收常驻两阶段调度；为兼容旧入口暂不搬动
batch_runtime.py
protenix/ runner/ configs/  # 本包内裁剪的必要PTX推理依赖，无训练入口
examples/                   # 通用单目标/批量FASTA
docs/                       # 输入、MSA、模型边界和发布说明
docker/                     # 模型/MSA分开的镜像配方
tests/ .github/workflows/   # CPU接口回归；不自动启动GPU或下载权重
tools/release_bundle.py     # 白名单、SHA检查与干净导出
```

保留flat模块以避免重写已验证的原生imports和权重键；这是实际可安装包，不依赖研究仓库的PYTHONPATH。
请使用专用环境，避免同时安装另一份Protenix或其他同名runner/configs包。
`complex_inputs.py`为新增PTX JSON薄适配；examples还含3单体/1PL/1PP真实序列和公共CCD标识，
均经明确白名单纳入Git/源码包，不含MD/GT坐标或内部路径。完整复合物输入说明见complex_inputs.md。

## 导出

维护者审核代码与许可后，更新代码清单，再导出到一个不存在的新目录：

```bash
python -B tools/release_bundle.py refresh
python -B tools/release_bundle.py check
python -B tools/release_bundle.py export --out /absolute/path/new_repo
```

只复制白名单文件，不复制weights/assets/inputs/runs/cache、PLAN、内部数据、训练代码或Git历史。
输入源目录含有运行产物也不会整体带走；拒绝软链接和覆盖已有输出。
refresh只是重建维护者清单，不能证明许可证、数值正确性或代码安全；提交前仍需审查。
不要直接git add整个原运行目录。导出的源码目录可另行git init，远端URL与发布权限由维护者决定。

## 发布前必须完成

- [ ] 确认本地新增代码的许可证、第三方通知和权重/CCD/MSA库再分发条件。不能照搬参考仓库的MIT/BSD授权。
- [ ] 准备获授权的权重托管与SHA清单、公开下载说明；当前只支持用户提供本地资源。
- [ ] 配置真实仓库URL、维护者信息及论文引用；没有填入虚构作者或链接。
- [ ] 干净安装/模型Docker build、常驻批量GPU烟测、正常MSA输入与用户anchor路径、真实本地搜索。
- [ ] 小显存/长序列运行边界；只有完成硬件实测后才公布具体保证。
- [ ] 指定另一模型独立审计；GitHub CI首次运行及实际权重验收。

CPU回归和离线wheel安装由本轮本地检查验证；不代表以上GPU/搜索验收。
pyproject暂有`Private :: Do Not Upload`以防误发PyPI；正式许可审核后由维护者移除。
新Docker入口统一为oridynamic；需要旧单目标脚本时显式覆盖entrypoint，不能混用两套参数。
