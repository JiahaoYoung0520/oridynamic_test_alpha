# Model notes and scope

## Default model

The alpha release uses three matching inference exports: Protenix base v1.0.0, the full autoencoder with latent width 32 at step 100,000, and **Joint200** (`joint_step0000200.pt`). Joint200 is the checkpoint saved after 200 adaptation updates from Joint11500, using predicted anchors and online LoRA context for the codec. The step number refers to this adaptation run. The Joint export contains the latent flow module, coordinate refiner, trunk LoRA weights, and latent normalization statistics; a separate flow checkpoint is not required.

Use the matching [Hugging Face bundle](https://huggingface.co/DanielYoung0520/oridynamic_v0) with the code revision recorded in its `weights_manifest.json`. Training checkpoints cannot be supplied directly. Inference exports use restricted loading and strict checks of checkpoint hashes, architecture, and parameter sets. Optimizers, schedulers, training RNG state, and local dataset paths are excluded.

## Sampling configuration

1. Native sequence, MSA, and CCD features condition generation. Without a supplied structure, the backbone predicts four anchor candidates using 200 SDE steps and ten trunk cycles, then selects the top native ranking score.
2. Joint200 uses the same online LoRA trunk context for the frozen codec, latent flow module, and coordinate refiner. The loader selects this route from checkpoint metadata. Older Joint11500 inference exports retain their original fixed-base codec context when explicitly selected in a custom path configuration.
3. The anchor defines the canonical coordinate frame. The autoencoder decoder returns displacements, which the caller adds to the anchor.
4. The flow module uses 32 Heun steps. The coordinate refiner uses 20 SDE steps with an initial noise scale of 16.
5. The trunk, codec, and DiT default to BF16. Coordinate diffusion retains its native FP32 boundaries, and TF32 is disabled.

Chemical reference features (`ref_pos`) are distinct from the full-system anchor. Explicit DiT reference conditioning remains disabled. No reference trajectory is needed at inference. Four initial-structure candidates do not imply four final conformational basins.

## Validation and scope

- Joint200 was adapted using labeled monomer and protein–ligand systems from the original test split. Performance on these systems is **not independent held-out evaluation**. This checkpoint should not be used to reproduce earlier Joint11500 benchmark claims without accounting for the adaptation.
- Protein–protein systems were excluded from this adaptation and remain outside its training scope. The PP input and sampling interface is available; task-specific quality requires separate evaluation.
- The exported Joint200 tensors are checked against the supplied training checkpoint. Full-size CPU model loading and small-module sampling regressions validate the weight and context routes. These checks do not establish structural quality or full-size GPU performance.
- The earlier Joint11500 release completed a five-system query-only GPU smoke test: three monomers, one PL complex, and one PP assembly. That result does not constitute a GPU validation of Joint200.
- FASTA describes monomers; JSON describes monomer, PL, or PP systems. Multiple FASTA records are independent targets. Complex sampling retains ligand atoms and relative partner geometry; see the [input format](complex_inputs.md).
- Chunking and stage-wise offloading are implemented. Model sharding, cooperative multi-GPU inference, CPU-only generation, and arbitrary-length operation on 24 GB GPUs have not been established.
- Generated conformations form an unordered ensemble, not an MD time series. Physical kinetics or thermodynamic basin populations are not guaranteed. CIF B-factors are placeholders, not pLDDT or RMSF.
