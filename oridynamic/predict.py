#!/usr/bin/env python3
"""单蛋白：原生MSA/特征→自动或用户anchor→Joint→CIF；不启动训练。"""
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
        raise ValueError("此发布版只允许自身代码目录，不接受其他仓库runtime-root")
    for name in ("oridynamic/runtime_utils.py", "protenix/model/modules/joint_ensemble.py"):
        if not (root / name).is_file():
            raise ValueError(f"缺少现有运行代码：{root / name}")
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"runner", "protenix", "configs", "scripts"}:
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError(f"已导入其他仓库的{name}：{path}，请使用新Python进程")
    sys.path.insert(0, str(root))
    return root


def main():
    started = time.perf_counter()
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", type=Path, required=True, help="一条蛋白序列")
    p.add_argument("--name", help="目标名，默认FASTA标题")
    p.add_argument("--out", type=Path, required=True, help="新的输出目录；拒绝覆盖")
    p.add_argument("--paths", type=Path, default=here / "configs/portable_paths.example.json", help="只读权重路径JSON；相对路径基于该JSON")
    p.add_argument("--runtime-root", type=Path, help="兼容旧参数，但只允许本发布目录")
    p.add_argument("--assets-root", type=Path, default=here / "assets", help="原生PTX只读CCD资源根")
    p.add_argument("--seeds", default="0", help="最终生成种子，逗号分隔")
    p.add_argument("--samples-per-seed", type=int, default=4, help="每种子的最终构象数")
    p.add_argument("--sample-chunk-size", type=int, default=1, help="生成分块，不改变总样本数")
    p.add_argument("--initial-structure", type=Path, help="可选完整单链PDB/mmCIF，不作为trunk模板")
    p.add_argument("--structure-chain", help="初始结构的author链ID；不填时必须只有一条蛋白链")
    p.add_argument("--msa-a3m", type=Path, help="已有unpaired A3M")
    p.add_argument("--msa-server-mode", choices=("protenix", "colabfold"), default="protenix")
    p.add_argument("--single-sequence", action="store_true", help="不联网，仅query MSA；用于离线smoke，非正常MSA质量协议")
    add_arguments(p)
    p.add_argument("--anchor-candidates", type=int, default=4, help="PTX候选数，每候选一个独立seed")
    p.add_argument("--anchor-seed", type=int, default=101)
    p.add_argument("--attention-chunk-size", type=int, default=32, help="复用原生trunk/codec/refiner注意力分块")
    p.add_argument("--attention-backend", choices=("torch", "cuequivariance"), default="torch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="trunk/codec/DiT；精修按父模型native精度")
    p.add_argument("--check", action="store_true", help="CPU原生特征及真实模型严格加载检查；不搜索MSA、不执行模型前向")
    p.add_argument("--trace-inputs", action="store_true", help="记录特征、C0、在线上下文及采样前RNG指纹；不改变随机数或采样")
    a = p.parse_args()
    try:
        validate_modes(a)
        if not 1 <= a.anchor_candidates <= 32 or a.attention_chunk_size < 1:
            raise ValueError("anchor候选须1–32，attention chunk须为正")
        if not 0 <= a.anchor_seed < 2**32 - a.anchor_candidates + 1:
            raise ValueError("anchor seed范围错误")
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
            raise ValueError("首版完整原子推理不接受X；请提供明确的20种氨基酸序列")
        request.update(schema="oridynamic_sequence_request_v1", status="cpu_check" if a.check else "submitted",
                       pending=["执行见completion.json或failure.json"],
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
        a.assets_root = a.assets_root.resolve(strict=True)
        def weight_path(value):
            path = Path(value)
            return (path if path.is_absolute() else a.paths.resolve().parent / path).resolve(strict=True)
        weights = {k: weight_path(paths["checkpoints"][k]) for k in ("native", "codec", "joint")}
        for path in weights.values():
            if not path.is_file():
                raise ValueError(f"权重不是普通文件：{path}")
        for filename in ("common/components.cif", "common/components.cif.rdkit_mol.pkl"):
            if not (a.assets_root / filename).is_file():
                raise ValueError(f"缺少只读CCD资源：{a.assets_root / filename}；不会自动下载")
        if a.out.exists() or a.out.is_symlink():
            raise ValueError(f"输出已存在，请换新目录：{a.out}")
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
    print(f"[{'CPU检查' if a.check else '推理'}完成] {a.out}", flush=True)


if __name__ == "__main__":
    main()
