# 公开交付范围：序列到系综

正式交付使用`tools/release_bundle.py export`白名单导出的新目录，而不是整个工作目录。
已增加pyproject、oridynamic CLI、doctor、单文件anchor/MSA、CPU CI及Docker统一入口；
安装不会附带weights/assets/inputs/runs，公共资源须另行准备。
当前仅为本地发布候选，未选择新增代码的开源许可、未上传远端，见[发布检查](docs/releasing.md)。

推荐`python -m oridynamic`多目标入口；其源码含完整的简单anchor选择逻辑（原生置信度top1），
不依赖隐藏服务。新入口只集成本地MSA，在线搜索放在教程。`predict.py`保留兼容旧任务，
其原生在线依赖仍在发行目录中，但新入口不调用。`runs/inputs/prepared`等用户产物不发布。

用户提供自己的蛋白序列，可加配体CCD/SMILES及伙伴链，安装运行资源后得到ensemble；可选初始结构及seed/sample
接口保留。不需要研究数据、MD、GT/frame0、latent/trunk缓存或数据集manifest。

## 代码与资源分开

- 代码：推理入口、网络/特征、MSA适配、CIF输出、安装/Docker、通用FASTA及CPU测试。
- 权重：独立weights目录，三份oridynamic_*_inference_v1文件及SHA清单。
  逐张量与原权重相同，只保留网络/架构/归一化/采样信息；不含优化器、训练配置、
  RNG或内部路径。受限加载，不直接接受旧训练checkpoint。
- 公共资源：CCD和MSA库由用户单独安装，推理只读；不是本项目研究数据。
- 许可：保留来源与上游许可；权重/本地分支/数据库再分发许可仍须维护者确认。
  本地准备完成不等于已远端公开上传。

## 不交付

研究训练/验证/测试数据、完整目标列表/划分、轨迹/GT/frame0、预计算缓存、内部评测结果、
训练入口/loss/优化器/数据准备流程，以及本地用户输入、日志和运行产物。
唯一明确批准的演示输入是examples/five_cases.json的五个序列/组分用例（3单体/1PL/1PP）；
选例规则公开，不能当作完整无偏benchmark，不带坐标/MD/MSA或研究路径。

code_manifest.json记录维护者白名单导出的实际代码文件及SHA，不含weights/assets。
.dockerignore阻止资源/运行产物进入镜像，.gitignore防止误加入Git；
忽略规则不等于tar/zip白名单，也不能清除已跟踪文件。

原准备目录保留历史记录，不要整体上传。干净导出目录也可能在用户测试后产生runs；
正式发布应按code_manifest.json列出的文件审查，再单独处理获授权的权重。
不隐瞒anchor：默认使用PTX预测top1，用户可提供完整结构；没有用GT选择或修复输出。
