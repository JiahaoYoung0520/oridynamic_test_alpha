# Installation and Docker

OriDynamic supports editable installation with `python -m pip install -e .` and installation from a built wheel. The public commands are `oridynamic sample`, `oridynamic msa`, and `oridynamic doctor`. See the [README](README.md) for a quick start.

Use the Python 3.11 / CUDA 12.6 environment below in a dedicated environment. This package is not currently distributed through PyPI. The two OriDynamic weights are hosted on [Hugging Face](https://huggingface.co/DanielYoung0520/oridynamic_v0); the Protenix base and CCD resources are obtained from the official Protenix distribution. Set `ORIDYNAMIC_PATHS` and `ORIDYNAMIC_ASSETS`, or pass `--paths` and `--assets-root` explicitly.

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

The tested runtime uses Linux x86_64, Python 3.11, and PyTorch 2.7.1 with CUDA 12.6. For a fresh checkout and environment:

```bash
git clone https://github.com/JiahaoYoung0520/oridynamic_test_alpha.git
cd oridynamic_test_alpha

conda create -n oridynamic python=3.11 -y
conda activate oridynamic

python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
oridynamic --help
```

Keep this environment activated and run subsequent commands from the code repository root. If the code is already cloned, start with environment setup inside that checkout. The package reads its Python dependencies from `requirements-inference.txt`; the CUDA-specific Torch wheel is installed first.

Alternatively, build the supplied inference image from this repository:

```bash
docker build -f docker/Dockerfile.inference -t oridynamic-inference:local .
```

The default Torch backend does not require compiling fused LayerNorm. The optional cuequivariance backend requires compatible extensions and explicit selection. Successful inference in an existing environment does not validate a newly built Docker image.

## 2. Model weights and CCD

Download the OriDynamic autoencoder and joint weights from [Hugging Face](https://huggingface.co/DanielYoung0520/oridynamic_v0), then obtain **Protenix base v1.0.0** from the official upstream URL. Our Hugging Face repository distributes only the two OriDynamic checkpoints; no login or invitation is required.

```bash
mkdir -p weights assets/common
python -m pip install -U huggingface_hub
hf download DanielYoung0520/oridynamic_v0 \
  ae.ckpt oridynamic.ckpt weights_manifest.json \
  --local-dir weights

curl -fL --retry 3 \
  https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt \
  -o weights/protenix_base_default_v1.0.0.pt
printf '%s  %s\n' \
  '2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04' \
  weights/protenix_base_default_v1.0.0.pt | sha256sum -c -
```

After also downloading the CCD files below, the local resource layout is:

```text
weights/
  protenix_base_default_v1.0.0.pt
  ae.ckpt
  oridynamic.ckpt
  weights_manifest.json
assets/common/
  components.cif
  components.cif.rdkit_mol.pkl
```

Use the exact official Protenix v1.0.0 file and the matching OriDynamic exports. The official file is accepted only with the pinned SHA-256 above; a same-named file from another model version is not interchangeable. The loader checks export schemas, base/codec compatibility, architecture, and Joint parameter sets using restricted deserialization. OriDynamic exports retain model parameters, inference configuration, and latent normalization statistics without optimizer state or research data.

Earlier installations using the verified OriDynamic-wrapped Protenix v1 base remain compatible. For a fresh install, use the official file above. Renaming an arbitrary checkpoint does not make it compatible.

Download both CCD files from the official URLs used by this release. The inference runtime does not fetch them automatically:

```bash
curl -fL --retry 3 \
  https://protenix.tos-cn-beijing.volces.com/common/components.cif \
  -o assets/common/components.cif
curl -fL --retry 3 \
  https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl \
  -o assets/common/components.cif.rdkit_mol.pkl
```

The RDKit companion is a pickle resource; use the official source above. Once the files are present, set the paths in the same shell:

```bash
export ORIDYNAMIC_PATHS="$PWD/configs/portable_paths.example.json"
export ORIDYNAMIC_ASSETS="$PWD/assets"
oridynamic doctor
```

`ORIDYNAMIC_ASSETS` names the directory containing `common/`, not `common/` itself. `oridynamic doctor` checks package availability and file paths; it does not load model weights or validate a GPU forward pass.

For custom locations, pass `--paths my_paths.json --assets-root /path/to/assets`. The JSON format is shown in [portable_paths.example.json](configs/portable_paths.example.json); relative checkpoint paths are resolved from the JSON file's directory. The Hugging Face bundle contains OriDynamic weights and documentation; a prebuilt container image is not provided. Resource redistribution terms are tracked separately from the source-code release.

## 3. Checks and GPU smoke tests

These commands assume the activated environment and resource variables from sections 1–2. The test suite and example files are in the Git checkout. Each check or inference run needs a new output directory.

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

An author-supplied H20 deployment report dated 2026-11-01 records successful execution with the current model on the five bundled MSA examples (three monomers, one PL complex, one PP assembly), producing ten final conformations. It used an existing dependency image, not a fresh build of the public Dockerfile. See [model notes](docs/model_card.md) for the evidence and its limits. Use normal MSA inputs for quality evaluation. Chunking and CPU offloading reduce memory use, but no sequence-length guarantees are established for 24 GB or 48 GB GPUs.

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

See the [README](README.md#msa-options) for search followed by inference in one command. Search defaults to CPU with 16 threads; GPU search must be selected explicitly with `--msa-device gpu` (the default is `--msa-device cpu`). Search completes before model loading. The wrapper does not start a persistent GPU server, search templates, or modify databases.

Database checks inspect the `dbtype` markers, rather than hashing the complete databases. The query sequence and alignment columns are validated before inference. A failed search stops the run; there is no automatic fallback to an online service or query-only input. A full local-database search still requires end-to-end validation.

The [batch guide](docs/batch_inference.md#outputs-and-timing) defines search and inference timing separately. Online search also includes networking, queueing, and service-side caching, so its timing should be reported separately from local search.

## 5. Sources and scope

OriDynamic inference code and its autoencoder and joint-model inference weights are released under [Apache-2.0](LICENSE), with copyright held by LGLab Frontier AI Group for its contributions. The package reuses Protenix networks and features and calls the public ColabFold/MMseqs2 CLI. It preserves [LICENSE-PTX](LICENSE-PTX), [NOTICE](NOTICE), and [source attribution](docs/SOURCES.md). Third-party dependencies, external weights, and public databases retain their respective terms.

The Git checkout includes inference modules, documentation, six example MSAs, tests, and the Docker/MSA helper scripts. A built wheel contains the Python modules and packaged configuration JSON files; it does not bundle the examples, documentation, tests, or shell helpers. Keep the checkout for the documented examples and checks. Model weights and CCD resources are downloaded separately. Full MSA databases, training code, and MD datasets are not included. User inputs, search results, and generated structures stay in the output directory.
