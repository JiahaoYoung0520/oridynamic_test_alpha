# Model notes and scope

## Default model

The alpha release uses three matching inference exports: Protenix base v1.0.0, the full autoencoder with latent width 32 at step 100,000, and Joint11500. The Joint checkpoint contains the latent generator, so a separate Stage-B checkpoint is not required.

An experimental 500-step adaptation for predicted anchors has not replaced these defaults. Training checkpoints cannot be supplied directly. Inference exports use restricted loading and strict checks of the base checkpoint hashes and architecture; they exclude optimizers, schedulers, training RNG state, and dataset paths.

## Sampling configuration

1. Native sequence, MSA, and CCD features condition generation. Without a supplied structure, Protenix predicts four anchor candidates using 200 SDE steps and ten trunk cycles, then selects the top native ranking score.
2. The codec uses frozen base context `C0`. The latent generator and refiner use the online LoRA context.
3. The anchor defines the canonical coordinate frame. The autoencoder decoder returns displacements, which the caller adds to the anchor.
4. Stage-B uses 32 Heun steps. The coordinate refiner uses 20 SDE steps with an initial noise scale of 16.
5. The trunk, codec, and DiT default to BF16. Coordinate diffusion retains its native FP32 boundaries, and TF32 is disabled.

Packaging and documentation changes preserve these model and sampling settings. Chemical reference features (`ref_pos`) are distinct from the supplied full-system anchor. If no initial structure is supplied, one is predicted. Explicit DiT reference conditioning remains disabled in this release.

Reference trajectories are not used to select, replace, or repair predictions. Four initial-structure candidates do not imply four final conformational basins.

## Supported inputs and limitations

- FASTA describes monomers; JSON describes monomer, PL, or PP systems. Multiple FASTA records are independent targets.
- PL and PP inputs use native featurization and whole-complex sampling, retaining ligand atoms and relative partner geometry. See the [input format](complex_inputs.md).
- The five-system GPU smoke test completed for three monomers, one PL complex, and one PP assembly. This establishes execution and output generation for these inputs, not benchmark quality.
- The legacy Joint11500 PP application retains out-of-distribution limitations. This release does not claim a newly trained four-source model.
- Chunking and stage-wise offloading are implemented. Model sharding, cooperative multi-GPU inference, CPU-only generation, and arbitrary-length operation on 24 GB GPUs have not been established.
- Generated conformations form an unordered ensemble, not an MD time series. Physical kinetics or thermodynamic basin populations are not guaranteed.
- CIF B-factors are placeholders and should not be interpreted as pLDDT or RMSF.
- Full local MSA search, smaller-GPU limits, and task-specific quality require separate evaluation. Inputs, checkpoints, and databases retain their respective usage terms.
