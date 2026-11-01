#!/usr/bin/env python3
"""Validate single-protein inputs without model inference."""
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
        raise ValueError(f"Input must be a nonempty regular file: {path}")
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
                raise ValueError("FASTA/A3M headers must not be empty")
        elif header is None:
            raise ValueError("Input requires a header starting with >")
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    if not records or any(not seq for _, seq in records):
        raise ValueError("FASTA/A3M records must not be empty")
    return records


def seed_list(text: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError("Seeds must be comma-separated integers") from exc
    if len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be unique and in [0, 2**32)")
    return seeds


def build_request(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = fasta_records(args.fasta)
    if len(rows) != 1:
        raise ValueError("Preparation requires one protein sequence; multiple chains cannot be treated as a single chain")
    title, sequence = rows[0]
    sequence = sequence.upper()
    if set(sequence) - AMINO_ACIDS:
        raise ValueError("Only the 20 standard amino acids and X are accepted in FASTA preparation")
    name = args.name or title.split()[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("Target names require 1-120 safe characters; use --name with letters, digits, underscores, or hyphens")
    seeds = seed_list(args.seeds)
    if args.samples_per_seed < 1 or args.sample_chunk_size < 1:
        raise ValueError("Sample counts and chunk sizes must be positive integers")
    protein = {"sequence": sequence, "count": 1}
    msa = {"mode": "auto", "server_mode": args.msa_server_mode, "executed": False}
    if args.msa_a3m:
        query = fasta_records(args.msa_a3m)[0][1]
        query = "".join(ch for ch in query if not ch.islower() and ch not in "-.")
        if query != sequence:
            raise ValueError("First A3M query differs from the input sequence after removing gaps and insertions")
        msa = {"mode": "provided", "input": file_info(args.msa_a3m),
               "validation": "query_only_native_featurizer_not_run"}
        protein["unpairedMsaPath"] = msa["input"]["path"]
    anchor = {"source": "ptx_predict", "policy": "native_ranking_score_top1",
              "candidate_seeds": [101], "samples_per_seed": 4,
              "candidate_budget_status": "proposed_not_benchmarked",
              "selection_executed": False}
    if args.initial_structure:
        if args.initial_structure.suffix.lower() not in {".cif", ".mmcif", ".pdb"}:
            raise ValueError("Initial structure must be a PDB/mmCIF file")
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
        "pending": ["MSA/native feature preparation", "Label-free Joint batch and fixed/online context",
                    "Anchor identity completeness, prediction, and selection", "GPU generation, CIF, and memory validation"],
    }
    return [{"name": name, "sequences": [{"proteinChain": protein}]}], request


def write_request(out: Path, native_input: list[dict], request: dict) -> None:
    # 独占创建：拒绝旧目录和链接，不覆盖任何既有实验。
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output exists; choose a new directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("input.json", native_input), ("request.json", request)):
        with (out / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="Single protein FASTA")
    parser.add_argument("--name", help="Output target identifier; defaults to the first word of the FASTA header")
    parser.add_argument("--seeds", default="0,1,2,3", help="Final ensemble generation seeds")
    parser.add_argument("--samples-per-seed", default=25, type=int, help="Final sample count per seed")
    parser.add_argument("--sample-chunk-size", default=1, type=int, help="Planned generation batch size; GPU capacity has not been validated")
    parser.add_argument("--initial-structure", type=Path, help="Optional decoder anchor; this command only checks the file")
    parser.add_argument("--msa-a3m", type=Path, help="Existing single-chain unpaired A3M; this command only checks the query")
    parser.add_argument("--msa-server-mode", choices=["protenix", "colabfold"], default="protenix",
                        help="Planned native MSA service; this command does not connect to the network")
    parser.add_argument("--out", type=Path, help="Optional new request directory; otherwise print only")
    args = parser.parse_args()
    try:
        native_input, request = build_request(args)
        if args.out:
            write_request(args.out, native_input, request)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(request, ensure_ascii=False, indent=2))
    print("[Preparation complete] No MSA search, model loading, or structure generation.")


if __name__ == "__main__":
    main()
