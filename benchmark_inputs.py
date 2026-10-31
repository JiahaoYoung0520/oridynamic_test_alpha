#!/usr/bin/env python3
"""四种anchor/MSA输入组合顺序计时；只调用本目录predict，不修改模型或启动训练。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from local_msa import add_arguments, preflight, query_sequence, validate_a3m, write_json
from prepare import file_info, seed_list


HERE = Path(__file__).resolve().parent
CASES = {
    "anchor_msa": (True, True, "用户提供anchor＋MSA"),
    "anchor_only": (True, False, "用户只提供anchor"),
    "msa_only": (False, True, "用户只提供MSA"),
    "neither": (False, False, "用户都不提供"),
}
# 先跑完整链路，产生另三组要回灌的同一anchor/MSA；不是GT或内部数据。
ORDER = ("neither", "anchor_msa", "msa_only", "anchor_only")


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="新的四组输出目录，拒绝覆盖")
    p.add_argument("--search-mode", choices=("local", "online"), required=True,
                   help="缺MSA时实际搜索；online会把序列发送到服务方")
    add_arguments(p)
    p.add_argument("--msa-server-mode", choices=("protenix", "colabfold"), default="protenix")
    p.add_argument("--assets-root", type=Path, default=HERE / "assets")
    p.add_argument("--paths", type=Path, default=HERE / "configs/portable_paths.example.json")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--samples-per-seed", type=int, default=2)
    p.add_argument("--sample-chunk-size", type=int, default=1)
    p.add_argument("--anchor-candidates", type=int, default=4)
    p.add_argument("--anchor-seed", type=int, default=101)
    p.add_argument("--attention-chunk-size", type=int, default=32)
    p.add_argument("--attention-backend", choices=("torch", "cuequivariance"), default="torch")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--plan", action="store_true", help="只打印四组计划，不搜索、不写文件、不加载模型")
    return p


def command(args, case, anchor, msa):
    have_anchor, have_msa, _ = CASES[case]
    cmd = [sys.executable, "-B", "-u", str(HERE / "predict.py"),
           "--out", str(args.out / case)]
    for key in ("fasta", "assets_root", "paths", "seeds", "samples_per_seed", "sample_chunk_size",
                "anchor_candidates", "anchor_seed", "attention_chunk_size", "attention_backend", "dtype", "device"):
        cmd.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    if have_anchor:
        cmd.extend(("--initial-structure", str(anchor)))
    if have_msa:
        cmd.extend(("--msa-a3m", str(msa)))
    elif args.search_mode == "local":
        for key in ("msa_local_db", "msa_search_executable", "msa_mmseqs", "msa_device",
                    "msa_threads", "msa_uniref_db", "msa_env_db"):
            cmd.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    else:
        cmd.extend(("--msa-server-mode", args.msa_server_mode))
    return cmd


def used_msa(folder, sequence):
    """从实际配置/原生JSON读取已用A3M，不猜搜索器文件名。"""
    folder = folder.resolve()
    cfg = json.loads((folder / "resolved_config.json").read_text())
    input_path = Path(cfg["input_json_path"]).resolve(strict=True)
    if not input_path.is_relative_to(folder):
        raise ValueError("实际输入JSON不在本组输出中")
    inp = json.loads(input_path.read_text())
    msa = Path(inp[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]).resolve(strict=True)
    _, count = validate_a3m(msa, sequence)
    return msa, count


def summarize(folder, case, process_seconds):
    completion = json.loads((folder / "completion.json").read_text())
    timing = json.loads((folder / "timing.json").read_text())
    if ((folder / "failure.json").exists() or completion["status"] != "inference_complete"
            or timing["status"] != "inference_complete"):
        raise ValueError("失败/CPU预检/未完成任务不能当推理计时")
    stages = timing["stages"]
    if any(e["status"] != "complete" or not math.isfinite(e["seconds"]) or e["seconds"] < 0 for e in stages):
        raise ValueError("阶段计时未完成或数值无效")
    names = [e["stage"] for e in stages]
    anchor, msa, label = CASES[case]
    searches = names.count("msa_local") + names.count("msa_online")
    if searches != (0 if msa else 1) or ((names.count("ptx_anchor") > 0) != (not anchor)):
        raise ValueError(f"{case}实际执行路径与四组定义不符")
    def total(*selected):
        return sum(e["seconds"] for e in stages if e["stage"] in selected)
    return dict(case=case, label=label, process_seconds=process_seconds,
                wall_seconds=timing["wall_seconds"], msa_search_seconds=total("msa_local", "msa_online"),
                anchor_predict_seconds=total("ptx_anchor"),
                context_seconds=total("base_context", "online_context"),
                ensemble_seconds=timing["ensemble_seconds"], total_samples=timing["total_samples"],
                ensemble_seconds_per_sample=timing["ensemble_seconds_per_sample"],
                gpu_name=timing.get("gpu_name"), max_cuda_allocated_gib=timing.get("max_cuda_allocated_gib"),
                max_cuda_reserved_gib=timing.get("max_cuda_reserved_gib"),
                output=str(folder), timing_sha256=file_info(folder / "timing.json")["sha256"])


def save_summary(out, rows):
    ordered = [rows[k] for k in CASES if k in rows]
    write_json(out / "summary.json", ordered)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    print("\n输入组合                    总进程(s)  MSA(s)   anchor(s) ensemble(s)", flush=True)
    for row in ordered:
        print(f'{row["label"]:<24} {row["process_seconds"]:9.2f} {row["msa_search_seconds"]:8.2f} '
              f'{row["anchor_predict_seconds"]:9.2f} {row["ensemble_seconds"]:11.2f}', flush=True)


def main():
    p = parser()
    args = p.parse_args()
    sequence = query_sequence(args.fasta)
    seed_list(args.seeds)
    if min(args.samples_per_seed, args.sample_chunk_size, args.attention_chunk_size) < 1:
        p.error("样本数与分块必须为正")
    if not 1 <= args.anchor_candidates <= 32 or not 0 <= args.anchor_seed <= 2**32 - args.anchor_candidates:
        p.error("anchor候选数或seed超出范围")
    if args.out.is_symlink():
        p.error("输出不能是符号链接")
    for key in ("fasta", "out", "paths", "assets_root"):
        setattr(args, key, getattr(args, key).resolve())
    if args.out.exists():
        p.error("输出已存在，请换新目录；不覆盖或拼接历史计时")
    if args.search_mode == "local":
        report = preflight(args)
        args.msa_local_db = Path(report["database_root"])
        args.msa_search_executable = report["executables"]["colabfold_search"]["path"]
        args.msa_mmseqs = report["executables"]["mmseqs"]["path"]
    elif args.msa_local_db:
        p.error("online模式不能传入本地库，避免误判搜索来源")
    anchor = args.out / "neither/anchor.cif"
    msa = args.out / "neither/ACTUAL_SEARCH_OUTPUT.a3m"  # 首组完成后从真实配置替换。
    if args.plan:
        print(json.dumps({c: command(args, c, anchor, msa) for c in ORDER}, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    # 不继承其他worktree的Python导入路径，搜索器由独立绝对路径指定。
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    write_json(args.out / "benchmark.json", dict(schema="oridynamic_input_timing_v1",
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source=file_info(Path(__file__)), execution_order=ORDER,
        provided_inputs="reused_from_neither_run_not_GT", fresh_process_per_case=True,
        cache_policy="no_local_search_reuse_os_and_server_cache_uncontrolled",
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES")))
    rows = {}
    try:
        for case in ORDER:
            cmd = command(args, case, anchor, msa)
            write_json(args.out / f"command_{case}.json", cmd)
            print(f"\n[四组计时 {len(rows)+1}/4] {CASES[case][2]}；同一张卡顺序执行", flush=True)
            started = time.perf_counter()
            subprocess.run(cmd, cwd=HERE, env=env, check=True)
            seconds = time.perf_counter() - started
            folder = args.out / case
            row = summarize(folder, case, seconds)
            actual_msa, count = used_msa(folder, sequence)
            row.update(msa_sha256=file_info(actual_msa)["sha256"], msa_records=count, msa_query_only=count == 1)
            if case == "neither":
                if not actual_msa.is_relative_to(folder):
                    raise ValueError("自动搜索的MSA必须位于本次输出内")
                msa = actual_msa
                write_json(args.out / "reused_inputs.json", dict(anchor=file_info(anchor), msa=file_info(msa)))
            row["msa_matches_first_run"] = row["msa_sha256"] == file_info(msa)["sha256"]
            rows[case] = row
            save_summary(args.out, rows)
        print(f"[四组完成] {args.out / 'summary.csv'}；是单次路径计时，不是质量或冷缓存基准", flush=True)
    except BaseException as exc:
        write_json(args.out / "failure.json", dict(error=f"{type(exc).__name__}: {exc}", completed=list(rows)))
        raise


if __name__ == "__main__":
    main()
