#!/usr/bin/env python3
"""Generate all-atom ensembles for independent monomer, protein-ligand, and protein-protein targets."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import time

from oridynamic.inference_timing import TimingLog
from oridynamic.local_msa import add_arguments, executable, run_search, validate_a3m, write_json
from oridynamic.prepare import AMINO_ACIDS, fasta_records, file_info, seed_list
from oridynamic.complex_inputs import load_native_targets, proteins

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--fasta', type=Path, help='Multiple FASTA records, each treated as an independent monomer')
    source.add_argument('--manifest', type=Path, help='Target manifest with per-target MSA and initial structures; paths are relative to the JSON file')
    source.add_argument('--input-json', type=Path, help='PTX task array with proteinChain/ligand entries for monomer, PL, or PP inputs')
    p.add_argument('--msa-dir', type=Path, help='Read <target>.a3m in FASTA mode; missing alignments may use local search')
    p.add_argument('--structure-dir', type=Path, help='Optional <target>.cif/.mmcif/.pdb in FASTA mode; predict missing structures')
    p.add_argument('--msa-a3m', type=Path, help='Existing MSA for a single FASTA record; incompatible with --msa-dir')
    p.add_argument('--initial-structure', type=Path, help='Optional PDB/mmCIF anchor for a single FASTA record')
    p.add_argument('--structure-chain', help='Author chain ID of the initial structure for a single FASTA record')
    p.add_argument('--single-sequence', action='store_true', help='Allow query-only MSA for targets without alignments; for smoke testing')
    add_arguments(p)
    p.add_argument('--out', type=Path, required=True, help='New output directory; existing runs are not overwritten or resumed')
    p.add_argument('--paths', type=Path, default=os.environ.get('ORIDYNAMIC_PATHS') or ROOT / 'configs/portable_paths.example.json')
    p.add_argument('--assets-root', type=Path, default=os.environ.get('ORIDYNAMIC_ASSETS') or ROOT / 'assets', help='Read-only CCD resource root')
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
    p.add_argument('--limit', type=int, help='Process only the first N input targets')
    p.add_argument('--plan', action='store_true', help='Read-only preflight; no search, Torch import, or directory creation')
    return p


def load_targets(a):
    if getattr(a, 'input_json', None):
        if any(getattr(a, k, None) for k in ('msa_dir', 'structure_dir', 'msa_a3m', 'initial_structure', 'structure_chain')):
            raise ValueError('Set MSA/anchor paths within each JSON entity/target; do not combine with FASTA shortcut options')
        return load_native_targets(a.input_json, a)
    if a.fasta:
        rows = [dict(name=n.split()[0], sequence=s.upper()) for n, s in fasta_records(a.fasta)]
        base = a.fasta.resolve().parent
        single_msa, single_structure = getattr(a, 'msa_a3m', None), getattr(a, 'initial_structure', None)
        chain = getattr(a, 'structure_chain', None)
        if (single_msa or single_structure or chain) and len(rows) != 1:
            raise ValueError('Single-file MSA/structure options require one FASTA record; use directories or a manifest for multiple targets')
        if (single_msa and a.msa_dir) or (single_structure and a.structure_dir):
            raise ValueError('Single-file and corresponding directory options cannot be combined')
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
                    raise ValueError(f"{row['name']}Multiple initial structures found; select one explicitly in a manifest")
                if candidates:
                    row['initial_structure'] = str(candidates[0].resolve())
    else:
        if a.msa_dir or a.structure_dir or any(getattr(a, k, None) for k in ('msa_a3m', 'initial_structure', 'structure_chain')):
            raise ValueError('Set per-target paths in manifest mode; directory options cannot be combined')
        data = json.loads(a.manifest.read_text())
        if data.get('schema') not in ('oridynamic_offline_batch_v1', 'oridynamic_batch_request_v1'):
            raise ValueError('Unknown manifest schema')
        rows = data['targets']
        base = a.manifest.resolve().parent
    if not isinstance(rows, list) or not rows:
        raise ValueError('At least one sequence is required')
    names, result = set(), []
    for item in rows:
        row = dict(item)
        validate_name(row['name'])
        if row['name'] in names:
            raise ValueError(f"Duplicate target name: {row['name']}")
        names.add(row['name'])
        sequence = row['sequence']
        if not sequence or set(sequence) - (AMINO_ACIDS - {'X'}):
            raise ValueError(f"{row['name']}Use uppercase sequences containing the 20 standard amino acids")
        # 丢弃无关输入字段；公共入口不读取轨迹/GT/frame0字段。
        row = {k: row[k] for k in ('name', 'sequence', 'msa_a3m', 'initial_structure', 'structure_chain') if k in row}
        for key in ('msa_a3m', 'initial_structure'):
            if row.get(key):
                path = Path(row[key])
                path = (path if path.is_absolute() else base / path).resolve(strict=True)
                row[key] = str(path)
                row[key + '_identity'] = file_info(path)
        if row.get('initial_structure') and Path(row['initial_structure']).suffix.lower() not in ('.cif', '.mmcif', '.pdb'):
            raise ValueError('Initial structure must be PDB or mmCIF')
        if row.get('structure_chain') is not None and not row.get('initial_structure'):
            raise ValueError('structure_chain requires initial_structure')
        if row.get('msa_a3m'):
            _, row['msa_records'] = validate_a3m(Path(row['msa_a3m']), sequence)
            row['msa_mode'] = 'provided'
        elif a.msa_local_db:
            row['msa_mode'] = 'local'
        elif a.single_sequence:
            row['msa_mode'] = 'single_sequence'
        else:
            raise ValueError(f"{row['name']}Missing MSA: supply A3M or --msa-local-db; online search is not automatic")
        result.append(row)
    if a.limit is not None:
        if a.limit < 1:
            raise ValueError('--limit must be positive')
        result = result[:a.limit]
    return result


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name):
        raise ValueError('Target names must contain 1-120 letters, digits, underscores, dots, or hyphens')


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
        raise ValueError('Sample counts and chunk sizes must be positive')
    if not 1 <= a.anchor_candidates <= 32 or not 0 <= a.anchor_seed <= 2**32-a.anchor_candidates:
        raise ValueError('Invalid anchor budget or seed range')
    if a.single_sequence and a.msa_local_db:
        raise ValueError('Query-only MSA and local search are mutually exclusive')
    targets = load_targets(a)
    a.out = a.out.absolute()
    if a.out.exists() or a.out.is_symlink():
        raise ValueError(f'Output already exists; refusing to overwrite: {a.out}')
    if not a.assets_root.is_dir():
        raise ValueError(f"CCD resource directory not found: {a.assets_root}. Set ORIDYNAMIC_ASSETS or --assets-root; see INSTALL.md.")
    a.assets_root = a.assets_root.resolve()
    for name in ('common/components.cif', 'common/components.cif.rdkit_mol.pkl'):
        if not (a.assets_root / name).is_file():
            raise ValueError(f'Missing read-only CCD resource: {name}')
    paths = json.loads(a.paths.read_text())
    weights = {}
    for key in ('native', 'codec', 'joint'):
        path = Path(paths['checkpoints'][key])
        weights[key] = (path if path.is_absolute() else a.paths.resolve().parent / path).resolve(strict=True)
        if not weights[key].is_file():
            raise ValueError(f'Checkpoint is not a file: {weights[key]}')
    from oridynamic.predict import select_runtime
    a.runtime_root = select_runtime(ROOT)
    if paths.get('runtime_source_root'):
        select_runtime(Path(paths['runtime_source_root']))
    if a.msa_local_db:
        a.msa_local_db = a.msa_local_db.resolve(strict=True)
        if a.out.resolve().is_relative_to(a.msa_local_db):
            raise ValueError('Output cannot be inside the MSA database directory')
    plan = dict(schema='oridynamic_resident_batch_v1', targets=targets,
        seeds=seeds, samples_per_seed=a.samples_per_seed, sample_chunk_size=a.sample_chunk_size,
        anchor_candidates=a.anchor_candidates, phase_order=['all_base_anchors_and_contexts', 'all_joint_ensembles'],
        online_search=False, weights={k:str(v) for k,v in weights.items()})
    if a.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print('[Preflight complete] No search, model loading, or files written; local tool availability is checked before search')
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
                            raise ValueError(f"{target['name']}/{protein['name']} MSA changed after input validation")
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
            from oridynamic.batch_runtime import ResidentRuntime
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
        print(f"[Batch complete] {len(targets)} targets; base/codec/Joint each loaded once; {a.out}", flush=True)
    except BaseException as exc:
        a.timing.finish('failed', error_type=type(exc).__name__)
        write_json(a.out / 'failure.json', dict(status='failed_not_complete', error=str(exc), type=type(exc).__name__))
        raise


if __name__ == '__main__':
    main()
