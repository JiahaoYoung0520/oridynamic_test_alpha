#!/usr/bin/env python3
"""已有MSA的离线单卡批量测速：只调用本发行版predict，不搜索、不读取轨迹。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from local_msa import validate_a3m, write_json
from prepare import AMINO_ACIDS, file_info, seed_list

HERE = Path(__file__).resolve().parent


def load_manifest(path):
    path = Path(path).resolve(strict=True)
    doc = json.loads(path.read_text())
    if doc.get('schema') != 'oridynamic_offline_batch_v1' or not doc.get('targets'):
        raise ValueError('需要非空oridynamic_offline_batch_v1清单')
    rows, names = [], set()
    for item in doc['targets']:
        name, seq = item['name'], item['sequence']
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name)
                or name in names or not seq or set(seq) - AMINO_ACIDS):
            raise ValueError(f'名称重复/不安全或序列无效：{name}')
        names.add(name)
        raw = Path(item['msa_a3m'])
        msa = (raw if raw.is_absolute() else path.parent / raw).resolve(strict=True)
        _, count = validate_a3m(msa, seq)
        row = dict(name=name, sequence=seq, tokens=len(seq), msa_a3m=str(msa),
                   msa_sha256=file_info(msa)['sha256'], msa_records=count, msa_query_only=(count == 1))
        if item.get('initial_structure'):
            raw = Path(item['initial_structure'])
            anchor = (raw if raw.is_absolute() else path.parent / raw).resolve(strict=True)
            row.update(initial_structure=str(anchor), anchor_sha256=file_info(anchor)['sha256'])
            if item.get('structure_chain'):
                row['structure_chain'] = item['structure_chain']
        rows.append(row)
    return rows


def command(a, row, folder):
    cmd = [sys.executable, '-B', '-u', str(HERE / 'predict.py'), '--fasta',
           str(a.out / 'inputs' / (row['name'] + '.fasta')), '--name', row['name'],
           '--msa-a3m', row['msa_a3m'], '--out', str(folder)]
    for key in ('assets_root', 'paths', 'seeds', 'samples_per_seed', 'sample_chunk_size',
                'anchor_candidates', 'anchor_seed', 'attention_chunk_size', 'attention_backend', 'dtype', 'device'):
        cmd.extend(('--' + key.replace('_', '-'), str(getattr(a, key))))
    for key in ('initial_structure', 'structure_chain'):
        if key in row:
            cmd.extend(('--' + key.replace('_', '-'), row[key]))
    if a.trace_inputs:
        cmd.append('--trace-inputs')
    return cmd


def summarize(folder, row, seconds, expected_samples):
    done = json.loads((folder / 'completion.json').read_text())
    timing = json.loads((folder / 'timing.json').read_text())
    if ((folder / 'failure.json').exists() or done.get('status') != 'inference_complete'
            or timing.get('status') != 'inference_complete' or done['msa_mode'] != 'provided'
            or done['total_samples'] != expected_samples or timing['total_samples'] != expected_samples
            or done['n_tokens'] != row['tokens']):
        raise ValueError('实际任务未完整完成/输入或预算不符')
    stages = timing['stages']
    if any(s['stage'].startswith('msa_') or s['status'] != 'complete'
           or not math.isfinite(s['seconds']) or s['seconds'] < 0 for s in stages):
        raise ValueError('离线测速不允许搜索阶段或无效计时')
    def total(*names):
        return sum(s['seconds'] for s in stages if s['stage'] in names)
    return dict(name=row['name'], status='complete', tokens=row['tokens'], atoms=done['n_atoms'],
        msa_records=row.get('msa_records'), msa_query_only=row.get('msa_query_only'),
        samples=expected_samples, process_seconds=seconds, wall_seconds=timing['wall_seconds'],
        load_seconds=total('joint_load_and_weight_hashes', 'native_model_load', 'codec_and_joint_load'),
        feature_seconds=total('native_features'), context_seconds=total('base_context', 'online_context'),
        anchor_seconds=total('ptx_anchor'), ensemble_seconds=timing['ensemble_seconds'],
        seconds_per_sample=timing['ensemble_seconds_per_sample'],
        max_cuda_allocated_gib=timing.get('max_cuda_allocated_gib'),
        max_cuda_reserved_gib=timing.get('max_cuda_reserved_gib'),
        gpu_name=timing.get('gpu_name'), output=str(folder), error='')


def save_progress(out, rows, total, elapsed):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    good = [r for r in rows if r['status'] == 'complete']
    write_json(out / 'summary.json', dict(targets=total, completed=len(good),
        failed=sum(r['status'] == 'failed' for r in rows), elapsed_seconds=elapsed,
        timing_scope='separate_process_per_target_includes_loading_no_msa_search',
        totals={k: sum(r[k] for r in good) for k in
                ('process_seconds', 'load_seconds', 'anchor_seconds', 'ensemble_seconds', 'samples')},
        rows=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='新输出目录；旧结果不覆盖')
    p.add_argument('--assets-root', type=Path, default=HERE / 'assets')
    p.add_argument('--paths', type=Path, default=HERE / 'configs/portable_paths.example.json')
    p.add_argument('--seeds', default='0,1')
    p.add_argument('--samples-per-seed', type=int, default=8)
    p.add_argument('--sample-chunk-size', type=int, default=1)
    p.add_argument('--anchor-candidates', type=int, default=4)
    p.add_argument('--anchor-seed', type=int, default=101)
    p.add_argument('--attention-chunk-size', type=int, default=32)
    p.add_argument('--attention-backend', choices=('torch', 'cuequivariance'), default='torch')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--trace-inputs', action='store_true')
    p.add_argument('--only', help='可选逗号分隔目标名；失败后在新目录仅重跑这些目标')
    p.add_argument('--plan', action='store_true', help='检查全部序列/MSA并打印计划，不写文件或加载模型')
    a = p.parse_args()
    seed_list(a.seeds)
    if (min(a.samples_per_seed, a.sample_chunk_size, a.attention_chunk_size) < 1
            or not 1 <= a.anchor_candidates <= 32
            or not 0 <= a.anchor_seed <= 2**32 - a.anchor_candidates):
        p.error('样本数、分块或anchor配置无效')
    if a.out.exists() or a.out.is_symlink():
        p.error('输出已存在，请换新目录；不混合历史计时')
    for key in ('manifest', 'out', 'paths', 'assets_root'):
        setattr(a, key, getattr(a, key).resolve())
    rows = load_manifest(a.manifest)
    if a.only:
        selected = a.only.split(',')
        if len(set(selected)) != len(selected) or not set(selected) <= {r['name'] for r in rows}:
            p.error('--only目标重复或不在清单中')
        rows = [r for r in rows if r['name'] in selected]
    print(f'[离线计划] {len(rows)}个目标；每目标{len(seed_list(a.seeds))*a.samples_per_seed}构象；'
          f'单卡{a.device}顺序运行；已有MSA，不搜索；每目标独立加载权重', flush=True)
    if a.plan:
        for row in rows:
            print(f"{row['name']} tokens={row['tokens']} MSA={row['msa_records']} anchor="
                  + ('provided' if row.get('initial_structure') else 'ptx_predict'))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'inputs').mkdir()
    (a.out / 'targets').mkdir()
    (a.out / 'logs').mkdir()
    write_json(a.out / 'batch.json', dict(schema='oridynamic_offline_batch_v1',
        manifest=file_info(a.manifest), targets=rows,
        settings={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        code_sha256={name:file_info(HERE/name)['sha256'] for name in ('predict.py', 'benchmark_batch.py')}))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    results = []; start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        name = row['name']; folder = a.out / 'targets' / name
        (a.out / 'inputs' / f'{name}.fasta').write_text(f'>{name}\n{row["sequence"]}\n')
        cmd = command(a, row, folder)
        print(f'[目标 {i}/{len(rows)}] {name} tokens={row["tokens"]}；日志={a.out / "logs" / (name+".log")}', flush=True)
        started = time.perf_counter()
        with (a.out / 'logs' / f'{name}.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = proc.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        print(f'[进行中 {i}/{len(rows)}] {name} 已等待{time.perf_counter()-started:.0f}s', flush=True)
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                raise
        elapsed = time.perf_counter()-started
        try:
            if code:
                raise ValueError(f'predict退出码{code}，详见日志')
            result = summarize(folder, row, elapsed, len(seed_list(a.seeds))*a.samples_per_seed)
            print(f'[完成 {i}/{len(rows)}] {name} 总计={elapsed:.1f}s '
                  f'anchor={result["anchor_seconds"]:.1f}s ensemble={result["ensemble_seconds"]:.1f}s', flush=True)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(name=name, status='failed', tokens=row['tokens'], process_seconds=elapsed, error=str(exc))
            print(f'[失败 {i}/{len(rows)}] {name}: {exc}；保留现场并继续下一目标', flush=True)
        results.append(result)
        save_progress(a.out, results, len(rows), time.perf_counter()-start)
    failed = [r['name'] for r in results if r['status'] != 'complete']
    print(f'[批量结束] 成功={len(rows)-len(failed)}/{len(rows)}，汇总={a.out / "summary.csv"}', flush=True)
    if failed:
        print('失败目标：'+','.join(failed), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
