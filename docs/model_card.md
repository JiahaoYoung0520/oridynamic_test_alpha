# Model and sampling

## Model components

OriDynamic generates all-atom conformational ensembles with a structural autoencoder, a latent flow module, and a coordinate refine module. The flow module samples token-level latents, the decoder produces structural proposals, and the refine module generates final conformations conditioned on those proposals. The latent width is 32.

| Checkpoint | Role |
| :--- | :--- |
| `protenix_base_default_v1.0.0.pt` | Cofolding backbone and molecular features |
| `ae.ckpt` | Structural autoencoder |
| `oridynamic.ckpt` | Flow module, refine module, trunk LoRA, and latent normalization |

Monomers, protein–ligand complexes, and protein–protein assemblies use the same released checkpoint pair. Obtain the weights and CCD resources using [INSTALL.md](../INSTALL.md#2-model-weights-and-ccd).

## Sampling configuration

When an initial structure is supplied, it becomes the decoder anchor. Otherwise, the backbone predicts four candidates using 200 SDE steps and ten trunk cycles, and selects the highest native ranking score. The number of candidates is controlled by `--anchor-candidates`.

The released model uses the same LoRA-adapted trunk context for the frozen autoencoder, flow module, and refine module. The selected anchor defines the canonical coordinate frame. Explicit reference conditioning in the latent flow module is disabled.

| Component | Default setting |
| :--- | :--- |
| Flow module | Heun integration, 32 steps |
| Refine module | SDE sampling, 20 steps, initial noise scale 16 |
| Trunk, autoencoder, and flow module precision | BF16 |
| Coordinate diffusion precision | FP32 |
| TF32 | Disabled |

Final conformations per target equal the number of seeds multiplied by `--samples-per-seed`. See the [batch guide](batch_inference.md) for sampling controls and output files.

## Model scope

The alpha checkpoint received additional adaptation on monomer and protein–ligand systems from the original test split; results on those systems are not independent held-out evaluation. Protein–protein systems were not included in this additional adaptation.

Generated conformations form an unordered ensemble. They do not provide physical transition times or calibrated basin populations. CIF B-factors are placeholders, not pLDDT or RMSF.

## Reproducibility

Record checkpoint hashes, inputs and MSAs, seeds, chunk sizes, attention backend, precision, and software/hardware versions. Matching seeds do not guarantee identical GPU coordinates. Changing `--sample-chunk-size` changes the assignment of intermediate SDE noise draws and can change individual conformations; keep it fixed when comparing samples.

## Checkpoint identity

<details>
<summary>File sizes and SHA-256 checksums</summary>

| File | Bytes | SHA-256 |
| :--- | ---: | :--- |
| `ae.ckpt` | 813638719 | `950ca1971e9b86d3c5560df694328d389877722569ddc40e0a48d6e11ac5e0ea` |
| `oridynamic.ckpt` | 948744621 | `3719e2eb8465d21116b0442702132a2f95ff5ae9792edba265fd21271228c9ad` |
| `protenix_base_default_v1.0.0.pt` (official external dependency) | 1475950125 | `2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04` |

</details>
