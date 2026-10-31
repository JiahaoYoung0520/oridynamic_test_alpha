#!/usr/bin/env python3
"""批量单体/PL/PP：全部anchor就绪后，再复用一次加载的Joint生成系综。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import time

from inference_timing import TimingLog
from local_msa import add_arguments, executable, run_search, validate_a3m, write_json
from prepare import AMINO_ACIDS, fasta_records, file_info, seed_list
from complex_inputs import load_native_targets, proteins

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--fasta', type=Path, help='多条FASTA；每条作为独立单蛋白，不是复合物')
    source.add_argument('--manifest', type=Path, help='目标JSON，支持各自MSA和初始结构；相对路径基于JSON')
    source.add_argument('--input-json', type=Path, help='PTX原生任务数组：proteinChain/ligand，支持单体、PL、PP')
    p.add_argument('--msa-dir', type=Path, help='FASTA模式下读取<目标名>.a3m；缺失项可本地搜索')
    p.add_argument('--structure-dir', type=Path, help='FASTA模式下可选<目标名>.cif/.mmcif/.pdb，缺失项由PTX预测')
    p.add_argument('--msa-a3m', type=Path, help='单条FASTA的已有MSA；不与--msa-dir混用')
    p.add_argument('--initial-structure', type=Path, help='单条FASTA的可选PDB/mmCIF anchor')
    p.add_argument('--structure-chain', help='单条FASTA初始结构的author链ID')
    p.add_argument('--single-sequence', action='store_true', help='明确允许无MSA目标使用query-only；仅用于smoke')
    add_arguments(p)
    p.add_argument('--out', type=Path, required=True, help='新的输出目录；拒绝覆盖，不隐式续跑')
    p.add_argument('--paths', type=Path, default=ROOT / 'configs/portable_paths.example.json')
    p.add_argument('--assets-root', type=Path, default=ROOT / 'assets', help='只读公共CCD资源根')
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
    p.add_argument('--limit', type=int, help='仅处理输入中的前N条，便于快速检查')
    p.add_argument('--plan', action='store_true', help='只读预检；不搜索、不导入Torch、不创建目录')
    return p


def load_targets(a):
    if getattr(a, 'input_json', None):
        if any(getattr(a, k, None) for k in ('msa_dir', 'structure_dir', 'msa_a3m', 'initial_structure', 'structure_chain')):
            raise ValueError('PTX JSON模式请在各entity/任务内填写MSA/anchor，不混用FASTA快捷参数')
        return load_native_targets(a.input_json, a)
    if a.fasta:
        rows = [dict(name=n.split()[0], sequence=s.upper()) for n, s in fasta_records(a.fasta)]
        base = a.fasta.resolve().parent
        single_msa, single_structure = getattr(a, 'msa_a3m', None), getattr(a, 'initial_structure', None)
        chain = getattr(a, 'structure_chain', None)
        if (single_msa or single_structure or chain) and len(rows) != 1:
            raise ValueError('单文件MSA/初始结构选项只适用于一条FASTA；多目标请使用目录或manifest')
        if (single_msa and a.msa_dir) or (single_structure and a.structure_dir):
            raise ValueError('单文件选项不能与对应目录选项混用')
        if single_msa: rows[0]['msa_a3m'] = str(single_msa.resolve(strict=True))
        if single_structure: rows[0]['initial_structure'] = str(single_structure.resolve(strict=True))
        if chain: rows[0]['structure_chain'] = chain
        for row in rows:
            # 先验证标识，再拼接用户目录，避免路径穿越。
            validate_name(row['name'])
            if a.msa_dir:
                path = a.msa_dir / (row['name'] + '.a3m')
                if path.is_file():
                    row['msa_a3m'] = str(path.resolve())
            if a.structure_dir:
                candidates = [a.structure_dir / (row['name'] + suffix) for suffix in ('.cif', '.mmcif', '.pdb')]
                candidates = [path for path in candidates if path.is_file()]
                if len(candidates) > 1:
                    raise ValueError(f"{row['name']}有多个初始结构；请用manifest明确选择")
                if candidates:
                    row['initial_structure'] = str(candidates[0].resolve())
    else:
        if a.msa_dir or a.structure_dir or any(getattr(a, k, None) for k in ('msa_a3m', 'initial_structure', 'structure_chain')):
            raise ValueError('manifest模式请在各目标内填写路径，不混用目录参数')
        data = json.loads(a.manifest.read_text())
        if data.get('schema') not in ('oridynamic_offline_batch_v1', 'oridynamic_batch_request_v1'):
            raise ValueError('未知manifest schema')
        rows = data['targets']
        base = a.manifest.resolve().parent
    if not isinstance(rows, list) or not rows:
        raise ValueError('至少需要一条序列')
    names, result = set(), []
    for item in rows:
        row = dict(item)
        validate_name(row['name'])
        if row['name'] in names:
            raise ValueError(f"重复目标名：{row['name']}")
        names.add(row['name'])
        sequence = row['sequence']
        if not sequence or set(sequence) - (AMINO_ACIDS - {'X'}):
            raise ValueError(f"{row['name']}需要明确的20种氨基酸大写序列")
        # 丢弃无关输入字段；公共入口不读取轨迹/GT/frame0字段。
        row = {k: row[k] for k in ('name', 'sequence', 'msa_a3m', 'initial_structure', 'structure_chain') if k in row}
        for key in ('msa_a3m', 'initial_structure'):
            if row.get(key):
                path = Path(row[key])
                path = (path if path.is_absolute() else base / path).resolve(strict=True)
                row[key] = str(path)
                row[key + '_identity'] = file_info(path)
        if row.get('initial_structure') and Path(row['initial_structure']).suffix.lower() not in ('.cif', '.mmcif', '.pdb'):
            raise ValueError('初始结构须为PDB/mmCIF')
        if row.get('structure_chain') is not None and not row.get('initial_structure'):
            raise ValueError('structure_chain需要initial_structure')
        if row.get('msa_a3m'):
            _, row['msa_records'] = validate_a3m(Path(row['msa_a3m']), sequence)
            row['msa_mode'] = 'provided'
        elif a.msa_local_db:
            row['msa_mode'] = 'local'
        elif a.single_sequence:
            row['msa_mode'] = 'single_sequence'
        else:
            raise ValueError(f"{row['name']}缺少MSA：提供A3M或--msa-local-db；不会自动调用在线服务")
        result.append(row)
    if a.limit is not None:
        if a.limit < 1:
            raise ValueError('--limit必须为正')
        result = result[:a.limit]
    return result


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name):
        raise ValueError('目标名需1–120个安全字母/数字/下划线/点/短横线')


def execute_batch(engine, targets, progress):
    """两阶段屏障：base完成所有目标后才能注入LoRA，禁止逐目标重复加载。"""
    try:
        engine.load_base()
        for i, target in enumerate(targets, 1):
            progress('anchor', i, target)
            engine.prepare(target)
        engine.load_joint()
        for i, target in enumerate(targets, 1):
            progress('ensemble', i, target)
            engine.sample(target)
    finally:
        engine.close()


def main(argv=None):
    a = parser().parse_args(argv)
    seeds = seed_list(a.seeds)
    if min(a.samples_per_seed, a.sample_chunk_size, a.attention_chunk_size) < 1:
        raise ValueError('样本数与chunk必须为正')
    if not 1 <= a.anchor_candidates <= 32 or not 0 <= a.anchor_seed <= 2**32-a.anchor_candidates:
        raise ValueError('anchor预算/seed范围无效')
    if a.single_sequence and a.msa_local_db:
        raise ValueError('query-only与本地搜索不能同时指定')
    targets = load_targets(a)
    a.out = a.out.absolute()
    if a.out.exists() or a.out.is_symlink():
        raise ValueError(f'输出已存在，拒绝覆盖：{a.out}')
    a.assets_root = a.assets_root.resolve(strict=True)
    for name in ('common/components.cif', 'common/components.cif.rdkit_mol.pkl'):
        if not (a.assets_root / name).is_file():
            raise ValueError(f'缺少只读CCD资源：{name}')
    paths = json.loads(a.paths.read_text())
    weights = {}
    for key in ('native', 'codec', 'joint'):
        path = Path(paths['checkpoints'][key])
        weights[key] = (path if path.is_absolute() else a.paths.resolve().parent / path).resolve(strict=True)
        if not weights[key].is_file():
            raise ValueError(f'权重不是文件：{weights[key]}')
    from predict import select_runtime
    a.runtime_root = select_runtime(ROOT)
    if paths.get('runtime_source_root'):
        select_runtime(Path(paths['runtime_source_root']))
    if a.msa_local_db:
        a.msa_local_db = a.msa_local_db.resolve(strict=True)
        if a.out.resolve().is_relative_to(a.msa_local_db):
            raise ValueError('输出不能放在MSA数据库内')
    plan = dict(schema='oridynamic_resident_batch_v1', targets=targets,
        seeds=seeds, samples_per_seed=a.samples_per_seed, sample_chunk_size=a.sample_chunk_size,
        anchor_candidates=a.anchor_candidates, phase_order=['all_base_anchors_and_contexts', 'all_joint_ensembles'],
        online_search=False, weights={k:str(v) for k,v in weights.items()})
    if a.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print('[只读预检完成] 未搜索、未加载模型、未写文件；本地工具可用性在正式搜索预检确认')
        return
    if any(p['msa_mode'] == 'local' for t in targets for p in proteins(t)):
        # 在切换到输出目录之前解析相对可执行路径。
        a.msa_search_executable = executable(a.msa_search_executable)
        a.msa_mmseqs = executable(a.msa_mmseqs)
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a.timing = TimingLog(a.out, started)
    a.check = False
    os.environ.update(PROTENIX_ROOT_DIR=str(a.assets_root), LAYERNORM_TYPE='torch',
        PROTENIX_DISABLE_FUSED_LN='1', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
        TORCH_EXTENSIONS_DIR=str(a.out / '.cache/torch_extensions'),
        XDG_CACHE_HOME=str(a.out / '.cache'), HF_HOME=str(a.out / '.cache/huggingface'),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.chdir(a.out)
    write_json(a.out / 'request.json', plan)
    try:
        # 搜索与模型生命周期分开，优先复用用户A3M；这里不导入Torch。
        for target in targets:
            folder = a.out / 'targets' / target['name']
            folder.mkdir(parents=True, exist_ok=False)
            for protein in proteins(target):
                chain_folder = folder / 'chains' / protein['name'] if 'native_input' in target else folder
                chain_folder.mkdir(parents=True, exist_ok=True)
                fasta = chain_folder / 'query.fasta'
                fasta.write_text('>' + protein['name'] + '\n' + protein['sequence'] + '\n')
                if protein['msa_mode'] == 'local':
                    search = copy.copy(a)
                    search.fasta, search.out = fasta, chain_folder / 'msa'
                    with a.timing.stage('msa_local', target=target['name'], chain=protein['name']):
                        msa_path, _ = run_search(search, search.out)
                    protein['msa_a3m'] = str(msa_path)
                for key, output in (('msa_a3m', 'provided.a3m'), ('paired_msa_a3m', 'paired.a3m')):
                    if protein.get(key + '_identity'):
                        if file_info(Path(protein[key])) != protein[key+'_identity']:
                            raise ValueError(f"{target['name']}/{protein['name']}的MSA在输入检查后改变")
                        text, _ = validate_a3m(Path(protein[key]), protein['sequence'])
                        path = chain_folder / output
                        path.write_text(text)
                        protein[key] = str(path)
                if protein['msa_mode'] == 'single_sequence' and not protein.get('msa_a3m'):
                    path = chain_folder / 'query.a3m'
                    path.write_text('>query\n' + protein['sequence'] + '\n')
                    protein['msa_a3m'] = str(path)
                if protein.get('msa_a3m'):
                    protein['effective_msa_identity'] = file_info(Path(protein['msa_a3m']))
        inference_started = time.perf_counter()
        with a.timing.stage('runtime_import'):
            from batch_runtime import ResidentRuntime
        engine = ResidentRuntime(a, weights)
        def progress(phase, index, target):
            print(f"[{phase} {index}/{len(targets)}] {target['name']}", flush=True)
            write_json(a.out / 'progress.json', dict(phase=phase, index=index, total=len(targets), target=target['name']))
        execute_batch(engine, targets, progress)
        result = dict(schema='oridynamic_resident_summary_v1', status='inference_complete',
            targets=engine.results, n_targets=len(targets), model_loads=engine.load_counts,
            inference_wall_seconds_excluding_msa=time.perf_counter()-inference_started,
            shared_initialization_seconds=sum(x['seconds'] for x in a.timing.data['stages']
                if x['stage'] in ('runtime_import', 'joint_load_and_weight_hashes', 'native_model_load', 'codec_and_joint_load')),
            msa_seconds=sum(x['seconds'] for x in a.timing.data['stages'] if x['stage']=='msa_local'))
        a.timing.finish('inference_complete', total_samples=len(targets)*len(seeds)*a.samples_per_seed,
                        inference_wall_seconds_excluding_msa=result['inference_wall_seconds_excluding_msa'])
        write_json(a.out / 'summary.json', result)
        write_json(a.out / 'progress.json', dict(phase='complete', completed=len(targets)))
        print(f"[批量完成] {len(targets)}目标；base/codec/Joint各加载一次；{a.out}", flush=True)
    except BaseException as exc:
        a.timing.finish('failed', error_type=type(exc).__name__)
        write_json(a.out / 'failure.json', dict(status='failed_not_complete', error=str(exc), type=type(exc).__name__))
        raise


if __name__ == '__main__':
    main()
