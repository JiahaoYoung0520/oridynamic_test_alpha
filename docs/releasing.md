# Source release guide

## Repository organization

The package and documentation organization was informed by [BioEmu2](https://github.com/microsoft/bioemu2) and [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina). This concerns installation, CLI organization, resources, and containers; their training and sampling implementations are not included.

```text
pyproject.toml              # Package installation, CLI, Python and dependency versions
oridynamic/                # Public sample / msa / doctor entry points
batch_predict.py           # Batch orchestration with models retained across targets
batch_runtime.py
protenix/ runner/ configs/ # Required inference dependencies; no training entry points
examples/                  # Single-target, batch, and complex inputs
docs/                      # Input, MSA, model, and release guides
docker/                    # Separate inference and MSA-search recipes
tests/ .github/workflows/  # CPU checks; no automatic GPU runs or checkpoint downloads
tools/release_bundle.py    # Source allowlist, hash checks, and clean export
```

The flat modules retain existing imports and checkpoint keys. The package does not depend on the research repository's `PYTHONPATH`. Use a dedicated environment to avoid conflicting Protenix, `runner`, or `configs` packages.

`complex_inputs.py` adapts the native Protenix JSON format. The allowlisted examples contain three monomers, one PL complex, and one PP assembly using sequences and a public CCD identifier. They include no MD coordinates or internal dataset paths. See the [complex-input guide](complex_inputs.md).

## Exporting a source bundle

After reviewing the intended source changes, refresh the manifest and export into a new directory:

```bash
python -B tools/release_bundle.py refresh
python -B tools/release_bundle.py check
python -B tools/release_bundle.py export --out /absolute/path/new_repo
```

Only allowlisted files are copied. Weights, assets, inputs, run outputs, caches, internal plans, research data, training code, and Git history are excluded. The exporter rejects symbolic links and existing output directories.

`refresh` updates the maintainer's manifest; it is not an independent review of numerical behavior or licensing. Review the source changes before refreshing it. Stage only the intended release files when updating the Git repository.

## Validation and publication status

- [x] The alpha source snapshot has been pushed to [GitHub](https://github.com/JiahaoYoung0520/oridynamic_test_alpha).
- [x] Local CPU regression and offline wheel-installation checks have been recorded.
- [x] The five-target GPU smoke test completed with matching inference weights: three monomers, one PL complex, and one PP assembly, with two final conformations each.
- [ ] Complete the licensing decisions for local additions and the redistribution conditions for weights, CCD resources, and MSA databases. Upstream notices remain in place.
- [ ] Configure weight hosting, public download instructions, and the matching hash manifest.
- [ ] Add the final maintainer and paper-citation information.
- [ ] Independently validate the new inference Docker build, normal MSA inputs, supplied-structure paths, and full local-database search.
- [ ] Establish sequence-length and memory limits on smaller GPUs before publishing hardware guarantees.
- [ ] Confirm the remote GitHub CI result and any separately requested independent review.

The completed smoke test establishes execution for its five inputs, not benchmark quality or the remaining deployment checks. The package retains `Private :: Do Not Upload` to prevent accidental PyPI publication until that distribution is ready.

The Docker entry point is `oridynamic`. To use a legacy script, override the entry point explicitly and follow that script's parameter interface.
