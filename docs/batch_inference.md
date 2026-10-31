# Batch inference

FASTA and request manifests describe independent protein targets. Use `--input-json` for Protenix-format monomer, protein–ligand, and protein–protein tasks; every entity within a task is generated together. See the [complex-input guide](complex_inputs.md).

The batch runner reuses the existing model, strict checkpoint loading, atom-identity mapping, and sampling operators.

## Inputs and execution

Run from this repository with the dependencies installed:

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-dir msas \
  --seeds 0,1 --samples-per-seed 8 --sample-chunk-size 1 \
  --out runs/batch_v1
```

The first word of each FASTA header is its unique target name. Alignments are read from `msas/<target>.a3m`. Add `--structure-dir anchors` to provide matching `.cif`, `.mmcif`, or `.pdb` files. Targets without a supplied structure receive a predicted anchor.

An initial structure does not replace the MSA or chemical reference features (`ref_pos`). It serves as the decoder anchor, rather than a trunk template. Its sequence and required heavy atoms must match the input; missing residues or atoms produce an error.

Use `--manifest request.json` to specify custom paths and chain IDs. Relative paths are resolved from the JSON directory:

```json
{
  "schema": "oridynamic_batch_request_v1",
  "targets": [
    {"name": "protein_a", "sequence": "ACDEFG", "msa_a3m": "msas/a.a3m"},
    {"name": "protein_b", "sequence": "ACDEFGH", "initial_structure": "anchors/b.cif", "structure_chain": "A"}
  ]
}
```

The second target has no MSA, so supply a local database as described below. This interface reads the declared molecular inputs, not MD trajectories or research caches. The earlier `oridynamic_offline_batch_v1` sequence/MSA manifest is also accepted.

`--limit 2` restricts processing to the first two targets. `--plan` checks input and resource paths without searching, importing Torch, creating files, or running the model.

## Local and online MSAs

With the database and ColabFold/MMseqs2 tools installed, search and inference can be combined:

```bash
python -B -u -m oridynamic \
  --fasta sequences.fasta --msa-local-db /data/colabfold_db \
  --msa-search-executable /opt/colabfold/bin/colabfold_search \
  --msa-mmseqs /opt/mmseqs/bin/mmseqs --msa-threads 32 \
  --seeds 0,1 --samples-per-seed 8 --out runs/local_batch_v1
```

You can also supply `--msa-dir`: existing alignments are reused, and only missing ones are searched. The wrapper invokes the search tool separately for each target; it does not guarantee a single database load. Search completes before model loading. Temporary files stay within the new run directory, and public databases are read-only. A failed search stops the run.

See [INSTALL.md](../INSTALL.md) for local installation, CPU/GPU search, and the separate search container. Online search is a separate preparation step described in the [online MSA guide](msa_online.md). `--single-sequence` is an explicit query-only smoke-test option, not a substitute for MSA-based quality evaluation.

## Model lifecycle and memory

1. Validate inputs and prepare all MSAs.
2. Load base Protenix once, compute each target's frozen context `C0`, and predict or map its anchor.
3. After all anchors and contexts are saved, load the autoencoder, DiT, and Joint state and inject LoRA into the base model.
4. Process targets in turn, computing the online LoRA context and generating samples with the selected anchor.

The base preparation phase must finish before `build_joint()` modifies the native model. Each target still requires its own forward passes. By default, four anchor candidates are generated and the native ranking score selects one; all final samples use that anchor. `--anchor-candidates 1` changes this budget explicitly.

Models remain in the process, with CPU/GPU offloading between phases and intermediate tensors released between targets. The defaults `--sample-chunk-size 1` and `--attention-chunk-size 32` limit working memory, but do not guarantee support for arbitrary lengths on smaller GPUs. Frozen context caches grow quadratically with token count, so reserve output-disk space accordingly. Caches are specific to each run.

## Outputs and timing

| File | Contents |
| :--- | :--- |
| `targets/<name>/anchor.cif` | Selected initial structure; full-precision coordinates are retained in `prepared.pt` |
| `prepared.pt`, `prepared.json` | Inputs, frozen context, atom identities, and anchor, with restricted loading and hash/contract checks |
| `ensemble.cif`, `proposals.cif` | Final conformations and decoded proposals for the target |
| `seed_*.pt` | Per-seed tensors |
| `run_metadata.json`, `completion.json` | Target configuration, provenance, and completion status |
| `progress.json` | Current phase and target position |
| `summary.json` | Shared initialization, per-target timings, and model-load counts |
| `timing.json` | Overall wall time and separately recorded stages |
| `anchor_phase/timing.json`, `ensemble_phase/timing.json` | Per-target phase timing |

Each ensemble/proposal CIF combines samples from all requested seeds. The final count per target is the number of seeds multiplied by `--samples-per-seed`. Chunk size controls how many samples are processed together, not the total count; changing chunk size does not guarantee bitwise-identical stochastic samples.

Overall wall time includes local MSA search. `inference_wall_seconds_excluding_msa` starts after search and includes initialization, model forward passes, and output writing. Per-target timing excludes time spent waiting for other targets. `ensemble_seed` is nested within `ensemble_target`; do not add both when aggregating elapsed time.

The runner stops on failure and preserves existing output. It requires a new output directory and does not automatically resume or overwrite runs. Existing anchors can be explicitly reused through a new input manifest.

Cache writing converts NumPy scalar metadata to Python built-in types; reading retains `weights_only=True`. For caches produced before the `numpy._core.multiarray.scalar` fix, rerun with a new `--out` directory. Old caches are not migrated automatically. Previously valid plain-string/tensor caches retain the same format.

## Validation scope

CPU tests cover the lifecycle, mixed MSA routing, atom identities, restricted cache serialization, and repeated sampling with small real codec/DiT/diffusion modules. The five-target GPU smoke test has also completed, generating ten final conformations with one base, codec, and Joint load per batch.

Matched single-target/batch comparisons, full local-database searches, lower-memory GPUs, and formal speed or quality benchmarks require separate validation. The interface organization was informed by [BioEmu2](https://github.com/microsoft/bioemu2); its implementation, training procedure, sampling protocol, and license were not copied.
