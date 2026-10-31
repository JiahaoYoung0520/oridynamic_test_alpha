# Installation and Docker

OriDynamic supports editable installation with `python -m pip install -e .` and installation from a built wheel. The public commands are `oridynamic sample`, `oridynamic msa`, and `oridynamic doctor`. See the [README](README.md) for a quick start.

Use the Python 3.11 / CUDA 12.6 environment below in a dedicated environment. This package is not currently distributed through PyPI. Weights are hosted in the currently private [Hugging Face repository](https://huggingface.co/DanielYoung0520/oridynamic_v0). Set `ORIDYNAMIC_PATHS` and `ORIDYNAMIC_ASSETS`, or pass `--paths` and `--assets-root` explicitly.

Inference requires this repository, its matching inference checkpoints, CCD resources, and the appropriate MSA inputs or search resources. FASTA entries describe independent proteins; use the [complex JSON format](docs/complex_inputs.md) for protein–ligand and protein–protein systems. Keep other Protenix installations and research worktrees out of this environment's `PYTHONPATH`.

## Batch inference

From this repository, use `python -m oridynamic sample`. After installation, `oridynamic sample` is available from other working directories. See the [batch guide](docs/batch_inference.md) for existing MSAs, local search, and optional initial structures, or the [online MSA guide](docs/msa_online.md) to prepare alignments separately.

The supplied Docker recipe uses the same CLI. A batch with existing MSAs can be run as follows after building the image:

```bash
mkdir batch_output
docker run --rm --network none --gpus device=0 \
  -v /absolute/path/weights:/weights:ro -v /absolute/path/assets:/assets:ro \
  -v /absolute/path/user_inputs:/input:ro -v "$PWD/batch_output:/output" \
  oridynamic-inference:local sample \
  --paths /opt/oridynamic/configs/container_paths.example.json --assets-root /assets \
  --fasta /input/sequences.fasta --msa-dir /input/msas --out /output/run
```

The inference image does not include MSA databases or search tools. Generate MSAs with a separate search container, or install ColabFold/MMseqs2 in the inference environment and mount the databases for automatic local search. The new Docker build recipe has not yet been independently validated.

## 1. Inference environment

The tested runtime uses Linux x86_64, Python 3.11, and PyTorch 2.7.1 with CUDA 12.6. Create a dedicated environment:

```bash
conda create -y -p "$PWD/.envs/ptx" python=3.11
conda run -p "$PWD/.envs/ptx" python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
conda run -p "$PWD/.envs/ptx" python -m pip install -r requirements-inference.txt
conda run -p "$PWD/.envs/ptx" python -m pip install --no-deps -e .
```

Alternatively, build the supplied inference image from this repository:

```bash
docker build -f docker/Dockerfile.inference -t oridynamic-inference:local .
```

The default Torch backend does not require compiling fused LayerNorm. The optional cuequivariance backend requires compatible extensions and explicit selection. Successful inference in an existing environment does not validate a newly built Docker image.

## 2. Model weights and CCD

Download the matching inference bundle from [Hugging Face](https://huggingface.co/DanielYoung0520/oridynamic_v0). While the model repository is private, use an account with access:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download DanielYoung0520/oridynamic_v0 \
  protenix_base_default_v1.0.0.pt \
  all_atom_struct_vae_step0100000.pt \
  joint_step0000200.pt \
  weights_manifest.json \
  --local-dir weights
```

Run this command from the code repository root. The expected layout is:

```text
weights/
  protenix_base_default_v1.0.0.pt
  all_atom_struct_vae_step0100000.pt
  joint_step0000200.pt
  weights_manifest.json
assets/common/
  components.cif
  components.cif.rdkit_mol.pkl
```

The three checkpoints must be the matching inference exports. Older training checkpoints with the same filenames are not accepted. The loader checks their schemas, base/codec hashes, architecture, and Joint parameter sets, and uses restricted deserialization. The exports retain the network parameters, inference configuration, and latent normalization statistics without training state or research data.

Prepare the matching CCD files, including the RDKit companion file, according to the [Protenix project](https://github.com/bytedance/Protenix). Use a trusted source for the pickle file. These resources are not downloaded automatically.

For custom locations, pass `--paths my_paths.json --assets-root /path/to/assets`. The JSON format is shown in [portable_paths.example.json](configs/portable_paths.example.json); relative checkpoint paths are resolved from the JSON file's directory. The Hugging Face bundle contains weights only; a prebuilt container image is not provided. Resource redistribution terms are tracked separately from the source-code release.

## 3. Checks and GPU smoke tests

```bash
python -B -m unittest discover -s tests -v
python -B -m oridynamic.predict --fasta examples/monomer.fasta --single-sequence \
  --check --out checks/new_install
CUDA_VISIBLE_DEVICES=0 python -B -u -m oridynamic.predict \
  --fasta examples/monomer.fasta --single-sequence --anchor-candidates 1 \
  --seeds 0,1 --samples-per-seed 2 --sample-chunk-size 1 --out runs/quick_smoke_v1
```

After building the inference image, use an available GPU and a new output directory for a container smoke test:

```bash
mkdir new_ensemble_output
docker run --rm --network none --gpus device=0 \
  -v /absolute/path/weights:/weights:ro -v /absolute/path/assets:/assets:ro \
  -v "$PWD/new_ensemble_output:/output" oridynamic-inference:local \
    --paths /opt/oridynamic/configs/container_paths.example.json --assets-root /assets \
    --fasta /opt/oridynamic/examples/monomer.fasta --single-sequence --anchor-candidates 1 \
    --seeds 0,1 --samples-per-seed 2 --sample-chunk-size 1 --out /output/run
```

The earlier Joint11500 release completed a five-target GPU smoke test covering three monomers, one PL complex, and one PP assembly in the existing runtime, with two conformations per target. That run predates Joint200; validation for the current checkpoint is documented in [model notes](docs/model_card.md). This validates the execution path for those inputs. Use normal MSA inputs for quality evaluation. Chunking and CPU offloading reduce memory use, but no sequence-length guarantees are established for 24 GB or 48 GB GPUs.

## 4. Local MSA search with ColabFold/MMseqs2

Install the search tools and databases separately. OriDynamic calls the public ColabFold/MMseqs2 tools; it does not provide a persistent search service. Existing alignments can be supplied directly with `--msa-a3m`.

A separate search environment helps avoid dependency conflicts. The host can launch a search container:

```bash
docker pull ghcr.io/sokrypton/colabfold:1.6.3-cuda12
MSA_IMAGE=ghcr.io/sokrypton/colabfold:1.6.3-cuda12 MSA_DEVICE=gpu MSA_GPUS=0 \
  bash tools/search_msa_docker.sh input.fasta /absolute/path/colabfold_db new_search_run \
    --msa-threads 16
```

The wrapper expects the image to be present locally. It mounts the source, input, and database read-only, disables networking, writes to a new output directory, and records the image ID. Use `MSA_DEVICE=cpu` for CPU search. Pass the resulting alignment as `--msa-a3m new_search_run/msa/unpaired.a3m`. Search provenance is recorded in the adjacent `msa/search.json`.

A search-image recipe is also provided in `docker/Dockerfile.msa`. To install the tools in a separate Python environment:

```bash
python3.11 -m venv .envs/msa
.envs/msa/bin/pip install 'colabfold==1.6.3'
mkdir -p vendor/mmseqs18
curl -fL https://mmseqs.com/archive/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/mmseqs-linux-gpu.tar.gz \
  -o vendor/mmseqs18/mmseqs.tar.gz
tar -xzf vendor/mmseqs18/mmseqs.tar.gz -C vendor/mmseqs18
vendor/mmseqs18/mmseqs/bin/mmseqs version
.envs/msa/bin/colabfold_search --help
```

Consult the [MMseqs2 installation instructions](https://github.com/soedinglab/MMseqs2#installation) for platform compatibility. Database preparation requires substantial storage and memory; follow the [ColabFold database guide](https://github.com/sokrypton/ColabFold#generating-msas-for-large-scale-structurecomplex-predictions). Existing databases can be reused read-only. Do not rebuild a shared production database in place or run `makepaddedseqdb` on it.

The default database prefixes are `uniref30_2302_db` and `colabfold_envdb_202108_db`. Override them with `--msa-uniref-db` and `--msa-env-db`. The template database (`db2`) is not used by this entry point.

Check the local search configuration without searching or writing files:

```bash
oridynamic msa --fasta examples/monomer.fasta --out runs/msa_check \
  --msa-local-db /absolute/path/colabfold_db \
  --msa-search-executable "$PWD/.envs/msa/bin/colabfold_search" \
  --msa-mmseqs "$PWD/vendor/mmseqs18/mmseqs/bin/mmseqs" --check
```

See the [README](README.md#msa-options) for search followed by inference in one command. Search defaults to CPU with 16 threads; GPU search must be selected explicitly. Search completes before model loading. The wrapper does not start a persistent GPU server, search templates, or modify databases.

Database checks inspect the `dbtype` markers, rather than hashing the complete databases. The query sequence and alignment columns are validated before inference. A failed search stops the run; there is no automatic fallback to an online service or query-only input. A full local-database search still requires end-to-end validation.

The [batch guide](docs/batch_inference.md#outputs-and-timing) defines search and inference timing separately. Online search also includes networking, queueing, and service-side caching, so its timing should be reported separately from local search.

## 5. Sources and scope

The package reuses Protenix networks and features and calls the public ColabFold/MMseqs2 CLI. It preserves [LICENSE-PTX](LICENSE-PTX) and [source attribution](docs/SOURCES.md). Dependencies, model weights, and public databases retain their respective terms.

The code package includes inference modules and six small example MSAs. Model weights, CCD resources, full MSA databases, training entry points, and MD datasets are supplied separately or excluded. User inputs, search results, and generated structures remain in each run's output directory.
