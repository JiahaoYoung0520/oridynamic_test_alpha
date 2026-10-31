# Five example systems

[five_cases.json](five_cases.json) contains three protein monomers, one protein–ligand complex, and one protein–protein assembly. The input consists of sequences, chain composition, and a public CCD identifier. It contains no trajectories, reference coordinates, precomputed MSAs, caches, or internal research paths.

| Case | System | Selection rationale |
| :--- | :--- | :--- |
| [6jv8_A](https://www.rcsb.org/structure/6JV8) | Monomer | Small system with favorable historical RMSF-profile correlation |
| [7n0j_E](https://www.rcsb.org/structure/7N0J) | Monomer | Small system with favorable historical RMSF-profile correlation |
| [6tly_A](https://www.rcsb.org/structure/6TLY) | Monomer | Small system with a historical fluctuation-amplitude ratio closer to one |
| [5T3N](https://www.rcsb.org/structure/5T3N) | Protein–ligand | A 153-residue protein with `CCD_75G`; selected for lower historical ligand-pose error |
| A00C3, a public DynaPPI system | Protein–protein | Two chains totaling 317 residues; a smaller historical PP case with favorable RMSF-profile correlation |

These are selected demonstrations, not a random test set or an unbiased benchmark. The sequences follow the previously processed system segments, including retained tags and termini, and may differ from complete PDB SEQRES sequences. Preserve the supplied sequences when running these examples.

The 5T3N example uses public `CCD_75G` chemical features. Historical scores obtained with MD-derived chemical features are not reproduced merely by running this input. Historical selection also used an online-codec/32-candidate configuration, while this release uses fixed codec context and four anchor candidates. The five-system release smoke test has now completed; it validates execution, not those historical quality scores. The PP input requires no DynaPPI trajectory downloads.

## Inference with local MSAs

Follow [INSTALL.md](../INSTALL.md) to configure `ORIDYNAMIC_PATHS`, `ORIDYNAMIC_ASSETS`, the search tools, and public databases:

```bash
oridynamic sample --input-json examples/five_cases.json \
  --msa-local-db /data/colabfold_db --msa-threads 16 \
  --seeds 0 --samples-per-seed 4 --sample-chunk-size 1 \
  --out runs/five_cases
```

For existing MSAs, set `unpairedMsaPath` in each `proteinChain` object and omit `--msa-local-db`. This example generates four final conformations per system, or 20 in total. By default, four initial-structure candidates are generated and one is selected as the anchor. Those candidates are separate from the final ensemble sample count.

## Offline smoke test

With the required weights and CCD resources already installed, query-only input can be selected explicitly for a small test without MSA search:

```bash
oridynamic sample --input-json examples/five_cases.json --single-sequence \
  --seeds 0 --samples-per-seed 2 --sample-chunk-size 1 \
  --out runs/five_cases_smoke
```

This five-target configuration has completed on an NVIDIA H200 with the matching inference weights, producing two final conformations per target. Output files were present and the final, proposal, and latent tensors were finite. This is an execution smoke test, not an assessment under normal MSA conditions.

Add `--plan` for a read-only input and resource-path check. It performs no model forward pass and does not replace real CCD featurization. Use an available GPU and a new output directory. See the [complex-input guide](../docs/complex_inputs.md) for per-chain MSAs and optional initial structures.
