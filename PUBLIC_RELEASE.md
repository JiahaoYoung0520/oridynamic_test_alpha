# Release scope: sequence to ensemble

This repository contains the OriDynamic alpha inference code. The source snapshot is hosted at [oridynamic_test_alpha](https://github.com/JiahaoYoung0520/oridynamic_test_alpha). Model weights and public molecular resources are distributed separately; public checkpoint downloads have not yet been configured.

For a clean source bundle, use `tools/release_bundle.py export` to copy the explicit allowlist into a new directory. The package includes the Python CLI, environment/resource checks, optional initial-structure and MSA inputs, CPU CI, and Docker recipes. Installation excludes weights, assets, user inputs, and run outputs. See the [release guide](docs/releasing.md).

The recommended `python -m oridynamic` entry point includes anchor prediction and selection by native confidence. It uses local MSA search or supplied alignments; online preparation is documented separately. `predict.py` remains available for compatibility with earlier tasks.

Users supply protein sequences, optional ligand definitions and partner chains, and the required runtime resources. They can provide an initial structure and control seeds and sample counts. Research datasets, MD trajectories, latent caches, and precomputed trunk features are not required.

## Code, weights, and resources

- **Code:** inference entry points, networks and features, MSA integration, CIF export, installation/Docker instructions, examples, and CPU tests.
- **Weights:** three inference exports and a SHA manifest in a separate weights bundle. Exported tensor values match their source checkpoints. The exports retain architecture, normalization, and sampling information, and exclude optimizers, training configuration, RNG state, and internal paths. Older training checkpoints are not accepted directly.
- **Public resources:** CCD and MSA databases are installed separately and accessed read-only during inference.
- **Attribution and licensing:** upstream notices are retained. Licensing and redistribution terms for local additions, weights, and databases remain separate release considerations.

## Excluded from the source bundle

Training, validation, and test datasets; full target lists and splits; trajectories and reference coordinates; precomputed research caches; internal evaluation outputs; training entry points, losses, optimizers, and data-preparation workflows; and local user inputs, logs, and generated structures.

The five selected systems in `examples/five_cases.json` are explicitly included as sequence/composition examples: three monomers, one PL complex, and one PP assembly. Their selection rationale is documented in [examples/README.md](examples/README.md). They do not constitute an unbiased benchmark and contain no reference coordinates, MD trajectories, precomputed MSAs, or internal paths.

`code_manifest.json` records the allowlisted source files and their hashes. `.dockerignore` limits the build context, and `.gitignore` excludes local resources and outputs from ordinary staging. Neither replaces the source allowlist or removes files already tracked by Git.

A directory used for inference may contain local outputs even if it began as a clean export. Review the manifest when publishing updates, and handle weight distribution separately. The default anchor is predicted and selected by native ranking; users may provide a complete structure. Reference trajectories are not used to select or repair the anchor.
