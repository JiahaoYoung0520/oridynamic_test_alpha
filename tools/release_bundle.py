#!/usr/bin/env python3
"""按明确白名单核验/导出Git源码目录；不包含权重、用户输入或运行产物。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    '.gitignore', '.dockerignore', 'README.md', 'INSTALL.md', 'PUBLIC_RELEASE.md', 'SOURCES.md',
    'LICENSE-PTX', 'pyproject.toml', 'MANIFEST.in', 'Makefile', 'requirements-inference.txt',
    'runtime_sources.json', 'predict.py', 'prepare.py', 'local_msa.py', 'weights_io.py',
    'batch_predict.py', 'batch_runtime.py', 'inference_runtime.py', 'inference_timing.py',
    'runtime_utils.py', 'benchmark_inputs.py', 'benchmark_batch.py', 'compare_runs.py',
    'examples/monomer.fasta', 'examples/batch.fasta', 'examples/five_cases.json', 'examples/README.md',
    'complex_inputs.py',
    'configs/__init__.py', 'configs/configs_data.py', 'configs/inference_defaults.json',
    'configs/base_diffusion_signature.json', 'configs/portable_paths.example.json',
    'configs/container_paths.example.json', 'tools/release_bundle.py', 'tools/search_msa_docker.sh',
    'docker/Dockerfile.inference', 'docker/Dockerfile.msa', '.github/workflows/ci.yml',
}
TREES = {'protenix': {'.py'}, 'runner': {'.py'}, 'oridynamic': {'.py'},
         'tests': {'.py', '.json'}, 'docs': {'.md'}}


def inventory(root=ROOT):
    names = set(ROOT_FILES)
    for folder, suffixes in TREES.items():
        for path in (root / folder).rglob('*'):
            if '__pycache__' not in path.parts and path.is_file() and path.suffix in suffixes:
                names.add(path.relative_to(root).as_posix())
    result = {}
    for name in sorted(names):
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError(f'白名单文件缺失或包含符号链接：{path}')
        data = path.read_bytes()
        result[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return dict(schema='oridynamic_code_bundle_v1', files=result)


def verify(root=ROOT):
    expected = json.loads((root / 'code_manifest.json').read_text())
    actual = inventory(root)
    if expected != actual:
        raise ValueError('源码白名单或SHA已改变；维护者审核后执行refresh，不自动接受未知代码')
    return actual


def export(out, root=ROOT):
    record = verify(root)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink() or out.resolve().is_relative_to(root.resolve()):
        raise ValueError('导出必须是源码目录外的新目录，拒绝覆盖或递归复制')
    out.mkdir(parents=True, exist_ok=False)
    for name in (*record['files'], 'code_manifest.json'):
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dst)
    verify(out)
    return len(record['files'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'refresh', 'export'))
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'refresh':
        record = inventory()
        (ROOT / 'code_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(f"[清单更新] {len(record['files'])}个源码文件；权重/输入/运行数据不纳入")
    elif a.action == 'check':
        print(f"[发布检查通过] {len(verify()['files'])}个白名单文件，SHA一致")
    else:
        if a.out is None: p.error('export需要--out')
        print(f'[干净导出完成] {export(a.out)}个源码文件 → {a.out}；未提交/推送Git')


if __name__ == '__main__': main()
