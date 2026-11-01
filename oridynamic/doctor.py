"""Read-only deployment checks; no model loading or network access."""
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
                  packages={}, resources={}, device="Not checked; use --check-device explicitly")
    if sys.version_info[:2] != (3, 11): errors.append("This deployment recipe supports Python 3.11")
    if platform.system() != "Linux": errors.append("Inference currently supports Linux/NVIDIA CUDA")
    for package in ("torch", "numpy", "scipy", "pandas", "biotite", "rdkit", "ml-collections"):
        try: report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None; errors.append(f"Missing dependency: {package}")
    spec = importlib.util.find_spec('oridynamic.batch_predict')
    root = Path(spec.origin).resolve().parents[1] if spec and spec.origin else None
    report['runtime_root'] = str(root) if root else None
    if root is None: errors.append("Cannot locate the installed batch_predict entry point")
    if paths:
        try:
            path = Path(paths).resolve(strict=True)
            config = json.loads(path.read_text())['checkpoints']
            for kind in ('native', 'codec', 'joint'):
                weight = Path(config[kind])
                weight = (weight if weight.is_absolute() else path.parent / weight).resolve()
                report['resources'][kind] = dict(path=str(weight), exists=weight.is_file())
                if not weight.is_file(): errors.append(f"Missing {kind} checkpoint: {weight}")
        except (OSError, ValueError, KeyError, TypeError) as exc: errors.append(f"Invalid checkpoint path configuration: {exc}")
    else: report['resources']['weights'] = 'Not checked: specify --paths or ORIDYNAMIC_PATHS'
    if assets_root:
        for filename in ('common/components.cif', 'common/components.cif.rdkit_mol.pkl'):
            asset = Path(assets_root) / filename
            report['resources'][filename] = dict(path=str(asset.resolve()), exists=asset.is_file())
            if not asset.is_file(): errors.append(f"Missing CCD resource: {asset}")
    else: report['resources']['ccd'] = 'Not checked: specify --assets-root or ORIDYNAMIC_ASSETS'
    if check_device:
        try:
            import torch
            available = torch.cuda.is_available()
            report['device'] = dict(cuda_available=available, cuda_version=torch.version.cuda,
                                    count=torch.cuda.device_count() if available else 0)
            if not available: errors.append("No CUDA device is available to this process")
        except ImportError as exc: errors.append(f"Torch import failed: {exc}")
    report.update(errors=errors, status='failed' if errors else 'preflight_only',
                  limitations='Installation/path checks only; checkpoint schema/SHA, model numerics, GPU capacity, and MSA search are not validated')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog='oridynamic doctor', description=__doc__)
    p.add_argument('--paths', type=Path, default=os.environ.get('ORIDYNAMIC_PATHS'))
    p.add_argument('--assets-root', type=Path, default=os.environ.get('ORIDYNAMIC_ASSETS'))
    p.add_argument('--check-device', action='store_true', help='Import Torch and query CUDA explicitly; no model loading')
    a = p.parse_args(argv)
    result = inspect(a.paths, a.assets_root, a.check_device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['errors'] else 0
