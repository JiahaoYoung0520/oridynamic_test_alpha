<div align="center">

<img src="docs/images/oridynamic-logo.png" alt="OriDynamic" width="680">

**All-atom conformational ensembles for biomolecular systems**

Alpha inference release

[Weights](https://huggingface.co/DanielYoung0520/oridynamic_v0) · [Installation](#installation) · [Quick start](#quick-start) · [MSA options](#msa-options) · [Batch and complexes](#batch-and-complexes) · [Documentation](#documentation)

</div>

OriDynamic generates conformational ensembles for protein monomers, protein–ligand complexes, and protein–protein assemblies. Provide molecular sequences and composition, together with an existing MSA or a local MSA database. An initial structure is optional: when omitted, OriDynamic predicts one before ensemble generation.

| System | Input |
| :--- | :--- |
| Protein monomer | FASTA sequence |
| Protein–ligand complex | JSON with protein sequences and ligand definitions |
| Protein–protein assembly | JSON with the interacting protein chains |

Multiple systems can be processed in one run. Models are loaded once per batch, and each system is processed independently.

## Installation

The tested inference environment uses **Linux, Python 3.11, PyTorch 2.7.1 with CUDA 12.6, and an NVIDIA GPU**.

```bash
git clone https://github.com/JiahaoYoung0520/oridynamic_test_alpha.git
cd oridynamic_test_alpha

conda create -n oridynamic python=3.11 -y
conda activate oridynamic

python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
oridynamic --help
```

The package installs its Python dependencies. For alternative environment setup and the Docker build recipe, see the [installation guide](INSTALL.md).

## Model resources

Inference requires three matching checkpoints (backbone, autoencoder, and joint model) and the CCD chemical-component resources. Prepare them as described in [Model weights and CCD](INSTALL.md#2-model-weights-and-ccd), then configure their locations:

```bash
export ORIDYNAMIC_PATHS="$PWD/configs/portable_paths.example.json"
export ORIDYNAMIC_ASSETS="/absolute/path/to/assets"
oridynamic doctor
```

The example configuration expects checkpoints in the repository's `weights/` directory. For another location, copy and edit the [path configuration](configs/portable_paths.example.json); relative checkpoint paths are resolved from that configuration file. `ORIDYNAMIC_ASSETS` points to the directory containing `common/components.cif` and `common/components.cif.rdkit_mol.pkl`.

> **Weights:** The matching bundle is hosted on [Hugging Face](https://huggingface.co/DanielYoung0520/oridynamic_v0). It is currently private; access and `hf auth login` are required. The default Joint export is `joint_step0000200.pt`, a predicted-anchor adaptation of Joint11500. See [model notes](docs/model_card.md) for its scope.

## Quick start

Run the five bundled examples: three monomers, one protein–ligand complex, and one protein–protein assembly. Their six protein MSAs are included, so no MSA search tools or databases are needed for this example.

```bash
oridynamic sample \
  --input-json examples/five_cases_with_msa.json \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases
```

This generates **two conformations per system**. Final structures are written to `runs/five_cases/targets/<target>/ensemble.cif`. Configure the environment, weights, and CCD resources above before running. See the [example guide](examples/README.md) for the included alignments.

For your own protein sequence, provide its matching A3M alignment:

```bash
oridynamic sample \
  --fasta protein.fasta \
  --msa-a3m protein.a3m \
  --seeds 0,1 \
  --samples-per-seed 8 \
  --sample-chunk-size 1 \
  --out runs/protein
```

This generates **16 conformations**: two seeds with eight samples each. Final structures are written to `runs/protein/targets/<target>/ensemble.cif`, where `<target>` is the first word of the FASTA header. Use a new output directory for each run.

<details>
<summary><strong>Start from an existing structure</strong></summary>

Supply a PDB or mmCIF structure to use as the initial structure:

```bash
oridynamic sample \
  --fasta protein.fasta \
  --msa-a3m protein.a3m \
  --initial-structure initial.cif \
  --structure-chain A \
  --seeds 0,1 --samples-per-seed 8 \
  --out runs/with_structure
```

The selected chain must match the input sequence and provide the required heavy atoms. See the [input guide](docs/complex_inputs.md) for structure-matching requirements.

</details>

## MSA options

| Available resources | How to use them |
| :--- | :--- |
| The five bundled examples | Use `examples/five_cases_with_msa.json`; alignments are included. |
| An existing A3M alignment | Pass `--msa-a3m protein.a3m`. |
| A local ColabFold/MMseqs2 database | Pass `--msa-local-db /data/colabfold_db` to search before inference. |
| An online MSA service | Follow the [online MSA guide](docs/msa_online.md), then provide the resulting A3M file. |

For local search, first install ColabFold/MMseqs2 and prepare the databases using the [MSA setup guide](INSTALL.md#4-local-msa-search-with-colabfoldmmseqs2):

```bash
oridynamic sample \
  --fasta protein.fasta \
  --msa-local-db /data/colabfold_db \
  --msa-threads 16 \
  --seeds 0,1 --samples-per-seed 8 \
  --out runs/from_sequence
```

For a minimal installation smoke test, `--single-sequence` uses only the query sequence. Use an MSA for normal ensemble-generation runs.

## Batch and complexes

**Independent proteins.** Each FASTA entry is treated as a separate target. Place the corresponding alignments in `msas/<target>.a3m`:

```bash
oridynamic sample \
  --fasta sequences.fasta --msa-dir msas \
  --seeds 0,1 --samples-per-seed 8 \
  --out runs/batch
```

Optional initial structures can be supplied with `--structure-dir anchors`, using matching target names. See the [batch guide](docs/batch_inference.md) for mixed inputs and per-target settings.

**Protein–ligand and protein–protein systems.** Describe the chains and ligands together in a JSON input. The bundled examples contain three monomers, one protein–ligand complex, and one protein–protein assembly:

```bash
oridynamic sample \
  --input-json examples/five_cases_with_msa.json \
  --seeds 0 --samples-per-seed 4 --sample-chunk-size 1 \
  --out runs/complex_examples
```

The [complex-input guide](docs/complex_inputs.md) covers ligand definitions, chain-specific MSAs, and optional structures. For new sequences, supply matching MSAs or use local search. The [example guide](examples/README.md) also covers the original sequence-only input.

## Documentation

| Guide | Contents |
| :--- | :--- |
| [Installation](INSTALL.md) | Environment setup, checkpoints, CCD resources, and Docker |
| [Batch inference](docs/batch_inference.md) | Input naming, sampling options, output files, and timing |
| [Complex inputs](docs/complex_inputs.md) | Protein–ligand and protein–protein input formats |
| [Online MSA](docs/msa_online.md) | Preparing alignments with an external service |
| [Examples](examples/README.md) | Five example systems and installation smoke tests |
| [Model notes](docs/model_card.md) | Model configuration and scope |
