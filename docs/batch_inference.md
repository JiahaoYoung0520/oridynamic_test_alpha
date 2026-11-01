# Batch inference

FASTA and request manifests describe independent protein targets. Use `--input-json` for Protenix-format monomer, protein–ligand, and protein–protein tasks; every entity within a task is generated together. See the [complex-input guide](complex_inputs.md).

The batch runner reuses the existing model, strict checkpoint loading, atom-identity mapping, and sampling operators.

## Inputs and execution

Complete the [installation and resource setup](../INSTALL.md) first. Run from the code checkout with the environment activated and `ORIDYNAMIC_PATHS` / `ORIDYNAMIC_ASSETS` set:

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

Local search defaults to CPU. Add `--msa-device gpu` to use a compatible GPU-enabled ColabFold/MMseqs2 installation; GPU visibility follows `CUDA_VISIBLE_DEVICES`.

You can also supply `--msa-dir`: existing alignments are reused, and only missing ones are searched. The wrapper invokes the search tool separately for each protein entity; it does not guarantee a single database load. Search completes before model loading. Temporary files stay within the new run directory, and public databases are read-only. A failed search stops the run.

See [INSTALL.md](../INSTALL.md) for local installation, CPU/GPU search, and the separate search container. Online search is a separate preparation step described in the [online MSA guide](msa_online.md). `--single-sequence` is an explicit query-only smoke-test option, not a substitute for MSA-based quality evaluation.

## Model lifecycle and memory

1. Validate inputs and prepare all MSAs.
2. Load base Protenix once, prepare each target, and predict or map its anchor. Base context is cached for compatibility with older checkpoints; the released joint model decodes with the subsequent online LoRA context.
3. After all anchors and contexts are saved, load the autoencoder, DiT, and Joint state and inject LoRA into the base model.
4. Process targets in turn, computing the online LoRA context and generating samples with the selected anchor.

All target anchors are prepared before the joint-model LoRA parameters are applied. Each target still requires its own forward passes. By default, four anchor candidates are generated and the native ranking score selects one; all final samples use that anchor. `--anchor-candidates 1` changes this budget explicitly.

Models remain in the process, with CPU/GPU offloading between phases and intermediate tensors released between targets. The defaults `--sample-chunk-size 1` and `--attention-chunk-size 32` limit working memory, but do not guarantee support for arbitrary lengths on smaller GPUs. The pair-context tensors in each run-specific `prepared.pt` grow quadratically with token count, so reserve output-disk space accordingly. They are not a shared cache across runs.

Additional controls include `--anchor-seed` (default 101), `--device` (default `cuda:0`), `--dtype` (`bf16` or `fp32`, default `bf16` for trunk/codec/DiT), and `--attention-backend` (`torch` by default; `cuequivariance` requires separately installed compatible extensions). These do not replace the checkpoint-bound flow/refiner sampling configuration.

## Outputs and timing

All paths below are relative to the directory passed to `--out`.

| Path | Contents |
| :--- | :--- |
| `request.json`, `progress.json` | Batch request and phase/target progress |
| `summary.json`, `timing.json` | Batch completion summary, model-load counts, and timing |
| `failure.json` | Failure details when an error occurs after run initialization; absent after success |
| `targets/<name>/anchor.cif` | Selected initial structure |
| `targets/<name>/proposals.cif`, `targets/<name>/ensemble.cif` | Decoded proposals and final conformations |
| `targets/<name>/prepared.pt`, `targets/<name>/prepared.json` | Prepared inputs, full-precision anchor, context tensors, identities, and integrity metadata |
| `targets/<name>/seed_<seed>.pt` | Per-seed proposal/final tensors |
| `targets/<name>/input.json`, `targets/<name>/resolved_config.json` | Effective native molecular input and model configuration |
| `targets/<name>/run_metadata.json`, `targets/<name>/completion.json` | Target provenance, sampling settings, and completion status |
| `targets/<name>/anchor_phase/timing.json`, `targets/<name>/ensemble_phase/timing.json` | Per-target phase timings |

For JSON inputs, protein queries and alignments are stored under `targets/<name>/chains/<first_entity_chain_id>/`: `query.fasta`, copied `provided.a3m`/`paired.a3m`, or query-only `query.a3m`, depending on the selected MSA mode. Local search creates `msa/unpaired.a3m` and `msa/search.json` there. For FASTA/manifest inputs, the same files are directly under `targets/<name>/`. Homomeric copies of one protein entity share its alignment. `input_fingerprints.json` is written within the target directory only with `--trace-inputs`.

Each ensemble/proposal CIF combines samples from all requested seeds. The final count per target is the number of seeds multiplied by `--samples-per-seed`. Chunk size controls how many samples are processed together, not the total count; changing chunk size changes the assignment of intermediate SDE noise draws and can produce different conformations. Even with the same seed and chunk size, identical GPU coordinates are not guaranteed. See [reproducibility notes](model_card.md#reproducibility).

Overall wall time includes local MSA search. `inference_wall_seconds_excluding_msa` starts after search and includes initialization, model forward passes, and output writing. Per-target timing excludes time spent waiting for other targets. `ensemble_seed` is nested within `ensemble_target`; do not add both when aggregating elapsed time.

The runner stops on failure and preserves existing output. It requires a new output directory and does not automatically resume or overwrite runs. Existing anchors can be explicitly reused through a new input manifest.

Cache writing converts NumPy scalar metadata to Python built-in types; reading retains `weights_only=True`. For caches produced before the `numpy._core.multiarray.scalar` fix, rerun with a new `--out` directory. Old caches are not migrated automatically. Previously valid plain-string/tensor caches retain the same format.

## Validation scope

CPU tests cover the lifecycle, mixed MSA routing, atom identities, restricted cache serialization, and repeated sampling with small real codec/DiT/diffusion modules. The author-supplied H20 report records ten final conformations from five bundled MSA targets using Joint200, with one base, codec, and Joint load per batch. This is reported execution evidence; see [model notes](model_card.md#validation-and-scope) for its scope and the distinction from earlier Joint11500 tests.

Matched single-target/batch comparisons, full local-database searches, lower-memory GPUs, and formal speed or quality benchmarks require separate validation. The interface organization was informed by [BioEmu2](https://github.com/microsoft/bioemu2); its implementation, training procedure, sampling protocol, and license were not copied.
