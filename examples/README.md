# Five example systems

[five_cases_with_msa.json](five_cases_with_msa.json) contains three protein monomers, one protein–ligand complex, and one protein–protein assembly, with matching precomputed A3M alignments in [msas/](msas/). Paths are relative to the JSON file, so the examples work after cloning the repository into a new location.

## Run the examples

These files are in the GitHub code checkout, not the Hugging Face weight repository or the installed wheel. Run the commands below from the root of the code checkout.

Follow [INSTALL.md](../INSTALL.md) to install the inference environment and configure `ORIDYNAMIC_PATHS` and `ORIDYNAMIC_ASSETS` for the model weights and CCD resources. MSA search tools and databases are not needed for the bundled examples.

```bash
oridynamic sample --input-json examples/five_cases_with_msa.json \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases
```

This generates two final conformations per system, or ten in total. Each final ensemble is saved as `runs/five_cases/targets/<name>/ensemble.cif`. Use a new output directory for each run. Add `--plan` to validate inputs and resource paths without model inference or output creation.

## Included alignments

| System | Type | Chain | Residues | A3M records, including query |
| :--- | :--- | :--- | ---: | ---: |
| [6jv8_A](https://www.rcsb.org/structure/6JV8) | Monomer | A | 76 | 119 |
| [7n0j_E](https://www.rcsb.org/structure/7N0J) | Monomer | E | 101 | 227 |
| [6tly_A](https://www.rcsb.org/structure/6TLY) | Monomer | A | 99 | 2,653 |
| [5T3N](https://www.rcsb.org/structure/5T3N) | Protein–ligand, `CCD_75G` | A | 153 | 4,941 |
| A00C3, DynaPPI | Protein–protein | A | 107 | 4,398 |
| A00C3, DynaPPI | Protein–protein | B | 210 | 4,584 |

The six files occupy about 3.6 MB in total. They reuse existing per-protein alignments from the ATLAS, MISATO, and DynaPPI data preparation. For 5T3N, query columns 7–159 of the existing 161-residue alignment were retained to match the supplied 153-residue sequence, together with insertions between those columns. No sequence search was rerun for this bundle.

Every query matches its example sequence exactly, and every aligned row has the required number of columns. All six files are unpaired MSAs; the PP chains have separate alignments, with no supplied paired MSA. Ligands do not require MSAs. Record counts describe the supplied files, not the number of rows retained by model featurization.

These demonstrations were selected from earlier experiments and are not an unbiased benchmark. Their processed sequences may retain tags or differ from complete PDB SEQRES sequences. Preserve the supplied sequences. The examples contain no MD trajectories, reference structures, model caches, or internal research paths.

## Other MSA modes

The original [five_cases.json](five_cases.json) contains the same systems without MSA paths. To search your own local database:

```bash
oridynamic sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_local_search
```

For an explicitly query-only installation test:

```bash
oridynamic sample --input-json examples/five_cases.json --single-sequence \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_query_only
```

The earlier query-only five-target GPU smoke test used Joint11500. Separately, the author-supplied H20 report records successful Joint200 execution with the bundled alignments. These checks establish execution for those inputs, not ensemble quality; see [model notes](../docs/model_card.md#validation-and-scope). See the [complex-input guide](../docs/complex_inputs.md) for new systems, per-chain MSAs, and optional initial structures.
