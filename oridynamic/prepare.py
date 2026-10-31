#!/usr/bin/env python3
"""仅准备单蛋白推理请求：标准库、无模型/联网/实验；不是推理入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


def file_info(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"输入不是非空普通文件：{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def fasta_records(path: Path) -> list[tuple[str, str]]:
    records = []
    header, sequence = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header, sequence = line[1:].strip(), []
            if not header:
                raise ValueError("FASTA/A3M标题不能为空")
        elif header is None:
            raise ValueError("输入必须有以 > 开头的标题")
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    if not records or any(not seq for _, seq in records):
        raise ValueError("FASTA/A3M记录不能为空")
    return records


def seed_list(text: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError("seed须为逗号分隔整数") from exc
    if len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seed须唯一且在[0, 2**32)内")
    return seeds


def build_request(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = fasta_records(args.fasta)
    if len(rows) != 1:
        raise ValueError("准备版每次仅处理一条蛋白序列；不能静默把多链当单链")
    title, sequence = rows[0]
    sequence = sequence.upper()
    if set(sequence) - AMINO_ACIDS:
        raise ValueError("仅接受20种标准氨基酸和X；修饰残基/配体需后续原生JSON接口")
    name = args.name or title.split()[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("目标名需1–120个安全字符，请用--name指定字母/数字/下划线/短横线")
    seeds = seed_list(args.seeds)
    if args.samples_per_seed < 1 or args.sample_chunk_size < 1:
        raise ValueError("样本数和chunk必须为正整数")
    protein = {"sequence": sequence, "count": 1}
    msa = {"mode": "auto", "server_mode": args.msa_server_mode, "executed": False}
    if args.msa_a3m:
        query = fasta_records(args.msa_a3m)[0][1]
        query = "".join(ch for ch in query if not ch.islower() and ch not in "-.")
        if query != sequence:
            raise ValueError("A3M首条query去gap/插入后与输入序列不同")
        msa = {"mode": "provided", "input": file_info(args.msa_a3m),
               "validation": "query_only_native_featurizer_not_run"}
        protein["unpairedMsaPath"] = msa["input"]["path"]
    anchor = {"source": "ptx_predict", "policy": "native_ranking_score_top1",
              "candidate_seeds": [101], "samples_per_seed": 4,
              "candidate_budget_status": "proposed_not_benchmarked",
              "selection_executed": False}
    if args.initial_structure:
        if args.initial_structure.suffix.lower() not in {".cif", ".mmcif", ".pdb"}:
            raise ValueError("初始结构应为PDB/mmCIF文件")
        anchor = {"source": "user_structure", "input": file_info(args.initial_structure),
                  "validation": "file_only_identity_and_geometry_not_checked"}
    request = {
        "schema": "oridynamic_inference_preparation_v1",
        "status": "prepared_not_executed", "ready_for_inference": False,
        "target": name, "input": file_info(args.fasta), "sequence_length": len(sequence),
        "msa": msa, "anchor": anchor,
        "ensemble": {"seeds": seeds, "samples_per_seed": args.samples_per_seed,
                     "total_samples": len(seeds) * args.samples_per_seed,
                     "sample_chunk_size": args.sample_chunk_size},
        "model_route": "checkpoint_condition_contract_pending_validation",
        "template_search": False, "dit_reference": False,
        "sampling_protocol": "pending_saved_checkpoint_and_verified_inference_binding",
        "pending": ["MSA/原生特征适配", "无标签Joint batch与固定/在线上下文",
                    "anchor身份完整性/实际预测与选择", "GPU生成/CIF/显存验收"],
    }
    return [{"name": name, "sequences": [{"proteinChain": protein}]}], request


def write_request(out: Path, native_input: list[dict], request: dict) -> None:
    # 独占创建：拒绝旧目录和链接，不覆盖任何既有实验。
    if out.exists() or out.is_symlink():
        raise ValueError(f"输出已存在，请选新目录：{out}")
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("input.json", native_input), ("request.json", request)):
        with (out / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="单条蛋白FASTA")
    parser.add_argument("--name", help="输出目标标识；默认FASTA首标题的首词")
    parser.add_argument("--seeds", default="0,1,2,3", help="最终ensemble生成种子")
    parser.add_argument("--samples-per-seed", default=25, type=int, help="每个种子的最终样本数")
    parser.add_argument("--sample-chunk-size", default=1, type=int, help="计划生成批量；显存尚未验证")
    parser.add_argument("--initial-structure", type=Path, help="可选decoder anchor；本轮只检查文件")
    parser.add_argument("--msa-a3m", type=Path, help="已有单链unpaired A3M；本轮只检查query")
    parser.add_argument("--msa-server-mode", choices=["protenix", "colabfold"], default="protenix",
                        help="拟使用的原生MSA服务；当前命令不会联网")
    parser.add_argument("--out", type=Path, help="可选的新请求目录；省略则仅打印")
    args = parser.parse_args()
    try:
        native_input, request = build_request(args)
        if args.out:
            write_request(args.out, native_input, request)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(request, ensure_ascii=False, indent=2))
    print("[准备完成] 未搜索MSA、未加载模型、未生成结构。")


if __name__ == "__main__":
    main()
