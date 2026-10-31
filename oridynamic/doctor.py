"""只读部署检查：默认不导入Torch、不联网、不加载权重或创建缓存。"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

from oridynamic import __version__


def inspect(paths=None, assets_root=None, check_device=False):
    errors = []
    report = dict(version=__version__, python=sys.version.split()[0], platform=platform.system(),
                  packages={}, resources={}, device="未检查；可显式--check-device")
    if sys.version_info[:2] != (3, 11): errors.append("当前部署配方仅验收Python3.11")
    if platform.system() != "Linux": errors.append("当前推理仅支持Linux/NVIDIA CUDA")
    for package in ("torch", "numpy", "scipy", "pandas", "biotite", "rdkit", "ml-collections"):
        try: report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None; errors.append(f"缺少依赖：{package}")
    spec = importlib.util.find_spec('batch_predict')
    root = Path(spec.origin).resolve().parent if spec and spec.origin else None
    report['runtime_root'] = str(root) if root else None
    if root is None: errors.append("找不到本包batch_predict入口")
    if paths:
        try:
            path = Path(paths).resolve(strict=True)
            config = json.loads(path.read_text())['checkpoints']
            for kind in ('native', 'codec', 'joint'):
                weight = Path(config[kind])
                weight = (weight if weight.is_absolute() else path.parent / weight).resolve()
                report['resources'][kind] = dict(path=str(weight), exists=weight.is_file())
                if not weight.is_file(): errors.append(f"缺少{kind}权重：{weight}")
        except (OSError, ValueError, KeyError, TypeError) as exc: errors.append(f"权重路径配置无效：{exc}")
    else: report['resources']['weights'] = '未指定--paths/ORIDYNAMIC_PATHS，未验证'
    if assets_root:
        for filename in ('common/components.cif', 'common/components.cif.rdkit_mol.pkl'):
            asset = Path(assets_root) / filename
            report['resources'][filename] = dict(path=str(asset.resolve()), exists=asset.is_file())
            if not asset.is_file(): errors.append(f"缺少CCD资源：{asset}")
    else: report['resources']['ccd'] = '未指定--assets-root/ORIDYNAMIC_ASSETS，未验证'
    if check_device:
        try:
            import torch
            available = torch.cuda.is_available()
            report['device'] = dict(cuda_available=available, cuda_version=torch.version.cuda,
                                    count=torch.cuda.device_count() if available else 0)
            if not available: errors.append("当前进程没有可用CUDA设备")
        except ImportError as exc: errors.append(f"Torch导入失败：{exc}")
    report.update(errors=errors, status='failed' if errors else 'preflight_only',
                  limitations='只检查安装/路径；未验证权重schema/SHA、模型数值、显存容量或真实MSA搜索')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog='oridynamic doctor', description=__doc__)
    p.add_argument('--paths', type=Path, default=os.environ.get('ORIDYNAMIC_PATHS'))
    p.add_argument('--assets-root', type=Path, default=os.environ.get('ORIDYNAMIC_ASSETS'))
    p.add_argument('--check-device', action='store_true', help='显式导入Torch并查询CUDA；仍不加载模型')
    a = p.parse_args(argv)
    result = inspect(a.paths, a.assets_root, a.check_device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['errors'] else 0
