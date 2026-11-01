#!/usr/bin/env python3
"""Prepare local protein MSAs with ColabFold/MMseqs2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from oridynamic.prepare import AMINO_ACIDS, fasta_records, file_info


def add_arguments(parser):
    parser.add_argument("--msa-local-db", type=Path, help="Enable local search using a prepared ColabFold database directory")
    parser.add_argument("--msa-search-executable", default="colabfold_search", help="Local ColabFold executable, optionally in a separate environment")
    parser.add_argument("--msa-mmseqs", default="mmseqs", help="MMseqs2 executable")
    parser.add_argument("--msa-device", choices=("cpu", "gpu"), default="cpu", help="Search device; CUDA_VISIBLE_DEVICES controls GPU visibility")
    parser.add_argument("--msa-threads", type=int, default=16, help="Maximum search threads, capped by CPU affinity")
    parser.add_argument("--msa-uniref-db", default="uniref30_2302_db", help="UniRef database prefix, passed to --db1")
    parser.add_argument("--msa-env-db", default="colabfold_envdb_202108_db", help="Environmental database prefix, passed to --db3 (not template database --db2)")


def validate_modes(args):
    if sum(bool(x) for x in (args.msa_a3m, args.single_sequence, args.msa_local_db)) > 1:
        raise ValueError("Choose only one of --msa-a3m, --single-sequence, or --msa-local-db")
    if args.check and not (args.msa_a3m or args.single_sequence):
        raise ValueError("predict --check does not search; run oridynamic msa first and supply --msa-a3m")


def query_sequence(fasta):
    rows = fasta_records(Path(fasta))
    if len(rows) != 1:
        raise ValueError("Each local inference search requires one protein sequence")
    sequence = rows[0][1].upper()
    if set(sequence) - (AMINO_ACIDS - {"X"}):
        raise ValueError("Use the 20 standard amino acids; X, multiple chains, and ligand FASTA are not accepted")
    return sequence


def executable(value):
    found = shutil.which(str(value))
    if not found:
        raise ValueError(f"Executable not found: {value}; see INSTALL.md; no automatic installation or network fallback")
    return str(Path(found).resolve())


def search_environment(out, threads, device):
    env = dict(os.environ)
    # 不把其他Python环境的PYTHONPATH/PYTHONHOME带入ColabFold。
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               XDG_CACHE_HOME=str(out / ".cache"), MPLCONFIGDIR=str(out / ".cache/matplotlib"),
               TMPDIR=str(out / "tmp"))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def preflight(args):
    sequence = query_sequence(args.fasta)
    if args.msa_threads < 1:
        raise ValueError("--msa-threads must be positive")
    if not args.msa_local_db:
        raise ValueError("Missing --msa-local-db")
    database = args.msa_local_db.resolve(strict=True)
    if not database.is_dir():
        raise ValueError(f"Database root is not a directory: {database}")
    if Path(args.out).resolve().is_relative_to(database):
        raise ValueError("MSA/inference output cannot be inside the database directory")
    db_info = {}
    for role, name in (("uniref", args.msa_uniref_db), ("environment", args.msa_env_db)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Database prefixes must be filenames, not paths")
        marker = database / (name + ".dbtype")
        db_info[role] = dict(prefix=name, dbtype=file_info(marker))
    threads = min(args.msa_threads, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    search = executable(args.msa_search_executable)
    mmseqs = executable(args.msa_mmseqs)
    return dict(schema="oridynamic_local_msa_v1", status="preflight_not_searched",
                query=file_info(args.fasta), sequence=sequence, database_root=str(database),
                databases=db_info, database_hash_scope="dbtype_only_not_full_database",
                executables={"colabfold_search": file_info(Path(search)), "mmseqs": file_info(Path(mmseqs))},
                threads=threads, device=args.msa_device, server_used=False, gpu_server=False,
                template_search=False, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def command(args, out, report):
    return [report["executables"]["colabfold_search"]["path"],
            str(out / "query.fasta"), report["database_root"], str(out / "raw"),
            "--mmseqs", report["executables"]["mmseqs"]["path"],
            "--db1", args.msa_uniref_db, "--db3", args.msa_env_db,
            "--threads", str(report["threads"]), "--gpu", "1" if args.msa_device == "gpu" else "0",
            "--gpu-server", "0", "--db-load-mode", "0", "--use-env", "1",
            "--use-templates", "0", "--filter", "1", "--unpack", "1"]


def validate_a3m(path, sequence):
    """校验所有行的比对列；仅移除单体ColabFold注释和插入空位'.'。"""
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("#"):
        if lines[0].strip() != f"#{len(sequence)}\t1":
            raise ValueError("ColabFold output length does not match this monomer/single-copy input")
        lines = lines[1:]
    records, header, parts = [], None, []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header, parts = line, []
        elif header is None:
            raise ValueError("A3M header is missing")
        else:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    if not records or records[0][1] != sequence:
        raise ValueError("Search A3M query differs from the input FASTA")
    normalized = []
    for header, row in records:
        if len(header) < 2 or not re.fullmatch(r"[A-Za-z.\-]+", row):
            raise ValueError("A3M contains empty records or invalid characters")
        row = row.replace(".", "")
        aligned = "".join(c for c in row if not c.islower())
        if len(aligned) != len(sequence):
            raise ValueError(f"A3M alignment column counts differ: {header}")
        if set(aligned) - set("ACDEFGHIKLMNPQRSTVWYBXZUO-"):
            raise ValueError(f"A3M contains characters outside the native protein alphabet: {header}")
        normalized.extend((header, row))
    return "\n".join(normalized) + "\n", len(records)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_search(args, out):
    report = preflight(args)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError(f"MSA output already exists; refusing to overwrite or reuse unchecked intermediate files: {out}")
    out = out.resolve()
    # 只允许新输出位于数据库外；杜绝把中间文件写进共享库。
    if out.is_relative_to(Path(report["database_root"])):
        raise ValueError("MSA output cannot be inside the database directory")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", ".cache"):
        (out / name).mkdir()
    (out / "query.fasta").write_text(">query\n" + report["sequence"] + "\n")
    env = search_environment(out, report["threads"], args.msa_device)
    report["command"] = command(args, out, report)
    report["environment"] = {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "MMSEQS_IGNORE_INDEX", "OMP_NUM_THREADS", "JAX_PLATFORMS")}
    write_json(out / "search.json", report)
    start = time.monotonic()
    try:
        # 帮助检查能识别旧版不支持GPU等参数的情况；不搜索、不分配模型。
        help_result = subprocess.run([report["command"][0], "--help"], env=env, cwd=out,
                                     capture_output=True, text=True, check=True, timeout=60)
        help_text = help_result.stdout + help_result.stderr
        (out / "colabfold_help.txt").write_text(help_text)
        for option in ("--gpu", "--gpu-server", "--db3", "--use-templates", "--unpack"):
            if option not in help_text:
                raise ValueError(f"colabfold_search does not support {option}; install a compatible version following INSTALL.md")
        version = subprocess.run([report["executables"]["mmseqs"]["path"], "version"], env=env,
                                 cwd=out, capture_output=True, text=True, check=True, timeout=30)
        report["mmseqs_version"] = version.stdout.strip()
        report["status"] = "running"
        write_json(out / "search.json", report)
        print(f"[Local MSA] device={args.msa_device} threads={report['threads']}; log={out / 'search.log'}", flush=True)
        # 日志逐行前台显示；不启动常驻服务或后台任务。
        with (out / "search.log").open("x") as log:
            with subprocess.Popen(report["command"], cwd=out, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait() != 0:
                        raise RuntimeError(f"Local MSA search failed (code={process.returncode}); see {out / 'search.log'}")
                except BaseException:
                    # 只清理本次子进程组（包括MMseqs），不留GPU子进程占用显存。
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        candidates = sorted((out / "raw").glob("*.a3m"))
        if len(candidates) != 1:
            raise ValueError(f"Single-sequence search must produce one final A3M; found {len(candidates)}; refusing to select one implicitly")
        text, count = validate_a3m(candidates[0], report["sequence"])
        a3m = out / "unpaired.a3m"
        a3m.write_text(text)
        report.update(status="complete", elapsed_seconds=time.monotonic()-start, n_records=count,
                      raw_a3m=file_info(candidates[0]), a3m=file_info(a3m), query_only=(count == 1))
        write_json(out / "search.json", report)
        print(f"[Local MSA complete] records={count} query_only={count == 1}；{a3m}", flush=True)
        return a3m, report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-start)
        write_json(out / "search.json", report)
        raise


def attach_local_msa(args, native_input, request):
    a3m, report = run_search(args, args.out / "msa")
    args.msa_a3m = a3m  # 原生特征入口使用已有A3M分支，不再调用服务器。
    native_input[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"] = str(a3m)
    request["msa"] = dict(mode="local", server_used=False, input=report["a3m"],
                          n_records=report["n_records"], query_only=report["query_only"],
                          search_report=file_info(args.out / "msa/search.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New MSA output directory; failed run directories cannot be reused")
    parser.add_argument("--check", action="store_true", help="Check paths, database markers, and executables only; no search or file writes")
    add_arguments(parser)
    args = parser.parse_args()
    if args.check:
        report = preflight(args)
        report["command"] = command(args, args.out.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        run_search(args, args.out)


if __name__ == "__main__":
    main()
