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
