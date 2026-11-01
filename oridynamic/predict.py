#!/usr/bin/env python3
"""Single-protein inference from sequence and optional structure to an all-atom ensemble."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from oridynamic.prepare import build_request, file_info
from oridynamic.local_msa import add_arguments, attach_local_msa, preflight, validate_modes
from oridynamic.inference_timing import TimingLog


def select_runtime(root: Path) -> Path:
    root = root.resolve(strict=True)
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("This release uses its own runtime; runtime-root cannot point to another repository")
    for name in ("oridynamic/runtime_utils.py", "protenix/model/modules/joint_ensemble.py"):
        if not (root / name).is_file():
            raise ValueError(f"Runtime code is missing: {root / name}")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"runner", "protenix", "configs", "scripts"}:
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError(f"Already imported a module from another repository: {name}：{path}; start a new Python process")
    sys.path.insert(0, str(root))
    return root


def main(argv=None):
    started = time.perf_counter()
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True, help="One protein sequence")
    p.add_argument("--name", help="Target name; defaults to the FASTA header")
    p.add_argument("--out", type=Path, required=True, help="New output directory; existing output is not overwritten")
    p.add_argument("--paths", type=Path, default=os.environ.get("ORIDYNAMIC_PATHS") or here / "configs/portable_paths.example.json", help="Read-only checkpoint path JSON; paths are relative to that JSON file")
    p.add_argument("--runtime-root", type=Path, help="Legacy option; only this release directory is accepted")
    p.add_argument("--assets-root", type=Path, default=os.environ.get("ORIDYNAMIC_ASSETS") or here / "assets", help="Read-only native PTX CCD resource root")
    p.add_argument("--seeds", default="0", help="Comma-separated generation seeds")
    p.add_argument("--samples-per-seed", type=int, default=4, help="Final conformation count per seed")
    p.add_argument("--sample-chunk-size", type=int, default=1, help="Generation chunk size; does not change the total sample count")
    p.add_argument("--initial-structure", type=Path, help="Optional complete single-chain PDB/mmCIF; not used as a trunk template")
    p.add_argument("--structure-chain", help="Initial structure author chain ID; if omitted, exactly one protein chain is required")
    p.add_argument("--msa-a3m", type=Path, help="Existing unpaired A3M")
    p.add_argument("--msa-server-mode", choices=("protenix", "colabfold"), default="protenix")
    p.add_argument("--single-sequence", action="store_true", help="Offline query-only MSA for smoke testing; not the normal quality protocol")
    add_arguments(p)
    p.add_argument("--anchor-candidates", type=int, default=4, help="PTX anchor candidates, each with a separate seed")
    p.add_argument("--anchor-seed", type=int, default=101)
    p.add_argument("--attention-chunk-size", type=int, default=32, help="Attention chunk size for trunk, codec, and refiner")
    p.add_argument("--attention-backend", choices=("torch", "cuequivariance"), default="torch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="Trunk/codec/DiT precision; refinement retains native precision boundaries")
    p.add_argument("--check", action="store_true", help="CPU native feature and strict model loading check; no MSA search or model forward pass")
    p.add_argument("--trace-inputs", action="store_true", help="Record feature, context, and pre-sampling RNG fingerprints without changing sampling")
    a = p.parse_args(argv)
    try:
        validate_modes(a)
        if not 1 <= a.anchor_candidates <= 32 or a.attention_chunk_size < 1:
            raise ValueError("Anchor candidates must be 1-32 and attention chunk size must be positive")
        if not 0 <= a.anchor_seed < 2**32 - a.anchor_candidates + 1:
            raise ValueError("Invalid anchor seed range")
        native_input, request = build_request(a)
        a.fasta = a.fasta.resolve()
        if a.initial_structure:
            a.initial_structure = a.initial_structure.resolve()
        if a.msa_a3m:
            a.msa_a3m = a.msa_a3m.resolve()
        if a.msa_local_db:
            local_check = preflight(a)
            a.msa_local_db = Path(local_check["database_root"])
            a.msa_search_executable = local_check["executables"]["colabfold_search"]["path"]
            a.msa_mmseqs = local_check["executables"]["mmseqs"]["path"]
            request["msa"] = {"mode": "local", "server_used": False, "status": "pending"}
        if "X" in native_input[0]["sequences"][0]["proteinChain"]["sequence"]:
            raise ValueError("All-atom inference requires the 20 standard amino acids; X is not accepted")
        request.update(schema="oridynamic_sequence_request_v1", status="cpu_check" if a.check else "submitted",
                       pending=["See completion.json or failure.json for execution status"],
                       sampling_protocol="checkpoint_bound_joint_SDE20_and_native_anchor_SDE200")
        request.pop("ready_for_inference", None)
        if a.single_sequence:
            request["msa"] = {"mode": "single_sequence", "server_used": False}
        if not a.initial_structure:
            request["anchor"].update(candidate_seeds=list(range(a.anchor_seed, a.anchor_seed+a.anchor_candidates)),
                                     samples_per_seed=1)
        paths = json.loads(a.paths.read_text())
        a.runtime_root = select_runtime(a.runtime_root or here)
        if paths.get("runtime_source_root"):
            select_runtime(Path(paths["runtime_source_root"]))
        if not a.assets_root.is_dir():
            raise ValueError(f"CCD resource directory not found: {a.assets_root}. Set ORIDYNAMIC_ASSETS or --assets-root; see INSTALL.md.")
        a.assets_root = a.assets_root.resolve()
        def weight_path(value):
            path = Path(value)
            return (path if path.is_absolute() else a.paths.resolve().parent / path).resolve(strict=True)
        weights = {k: weight_path(paths["checkpoints"][k]) for k in ("native", "codec", "joint")}
        for path in weights.values():
            if not path.is_file():
                raise ValueError(f"Checkpoint is not a regular file: {path}")
        for filename in ("common/components.cif", "common/components.cif.rdkit_mol.pkl"):
            if not (a.assets_root / filename).is_file():
                raise ValueError(f"Missing read-only CCD resource: {a.assets_root / filename}; automatic download is disabled")
        if a.out.exists() or a.out.is_symlink():
            raise ValueError(f"Output exists; choose a new directory: {a.out}")
        a.out = a.out.absolute()
        a.out.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))

    # 在任何PTX/Torch导入之前隔离可写缓存和后端；不修改容器或父shell。
    os.environ.update(PROTENIX_ROOT_DIR=str(a.assets_root), LAYERNORM_TYPE="torch",
        PROTENIX_DISABLE_FUSED_LN="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        TORCH_EXTENSIONS_DIR=str(a.out / ".cache/torch_extensions"),
        XDG_CACHE_HOME=str(a.out / ".cache"), HF_HOME=str(a.out / ".cache/huggingface"),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if a.check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(a.out)  # 原生相对路径缓存也限定在当前新任务，不写源码或共享资源目录。
    # 搜索发生在Torch/PTX导入和模型加载之前；失败不会静默改成在线或query-only。
    from oridynamic.local_msa import write_json
    write_json(a.out / "request.json", request)
    a.timing = TimingLog(a.out, started)
    try:
        if a.msa_local_db:
            with a.timing.stage('msa_local'):
                attach_local_msa(a, native_input, request)
                write_json(a.out / "request.json", request)
        with a.timing.stage('runtime_import'):
            from oridynamic import inference_runtime as runtime
        result = runtime.run(a, paths, weights, native_input)
        result['timing_file'] = 'timing.json'
        a.timing.finish('cpu_check_complete' if a.check else 'inference_complete',
            total_samples=result.get('total_samples', 0),
            max_cuda_allocated_gib=result.get('max_cuda_allocated_gib'),
            max_cuda_reserved_gib=result.get('max_cuda_reserved_gib'))
        runtime.save_json(a.out / "completion.json", result)
    except Exception as exc:
        a.timing.finish('failed', error_type=type(exc).__name__)
        write_json(a.out / "failure.json", {"error": str(exc), "type": type(exc).__name__,
            "status": "failed_not_complete"})
        raise
    print(f"[{'CPU check' if a.check else 'Inference'} complete] {a.out}", flush=True)


if __name__ == "__main__":
    main()
