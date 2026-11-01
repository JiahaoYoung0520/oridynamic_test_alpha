# Model notes and scope

## Default model

The alpha release uses the official **Protenix base v1.0.0**, `ae.ckpt` (the structural autoencoder, latent width 32), and `oridynamic.ckpt` (latent flow, coordinate refiner, trunk LoRA weights, and latent normalization statistics). A separate flow checkpoint is not required.

The OriDynamic files are hosted on [Hugging Face](https://huggingface.co/DanielYoung0520/oridynamic_v0). Obtain the Protenix base directly from the official URL in [INSTALL.md](../INSTALL.md#2-model-weights-and-ccd). Use the code revision recorded in `weights_manifest.json` or a compatible later revision. The loader verifies the exact upstream base hash, the matching codec identity, and strict model parameter sets using restricted deserialization. The two OriDynamic files are unchanged tensors under neutral filenames; filenames do not determine checkpoint compatibility.

For provenance, the autoencoder was saved at step 100,000. The joint model is the checkpoint previously called **Joint200**, saved after 200 predicted-anchor adaptation updates from Joint11500. It uses online LoRA context for the frozen codec. The release exports exclude optimizers, schedulers, training RNG state, and local dataset paths.

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
- An author-supplied deployment report dated **2026-11-01** records Joint200 inference on an NVIDIA H20 with the five bundled MSA examples (three monomers, one PL complex, one PP assembly), producing ten conformations. It also records twelve conformations each for a monomer, PL, and PP system. These are reported execution checks; raw H20 output artifacts have not been independently reviewed in this release update. They do not establish structural quality or an independent benchmark.
- That H20 deployment used an existing image matching the pinned Python dependencies. It does not validate a fresh build of `docker/Dockerfile.inference`. An earlier query-only five-system smoke test used Joint11500 and is separate evidence.
- FASTA describes monomers; JSON describes monomer, PL, or PP systems. Multiple FASTA records are independent targets. Complex sampling retains ligand atoms and relative partner geometry; see the [input format](complex_inputs.md).
- Chunking and stage-wise offloading are implemented. Model sharding, cooperative multi-GPU inference, CPU-only generation, and arbitrary-length operation on 24 GB GPUs have not been established.
- Generated conformations form an unordered ensemble, not an MD time series. Physical kinetics or thermodynamic basin populations are not guaranteed. CIF B-factors are placeholders, not pLDDT or RMSF.

## Reproducibility

Record the checkpoint hashes, inputs and MSAs, seeds, chunk sizes, attention backend, precision, and software/hardware environment. Matching seeds do not guarantee identical coordinates on GPU. Changing `--sample-chunk-size` can change which intermediate SDE noise draws are assigned to each sample and produce different conformations; it is not merely a memory setting. Keep it fixed when comparing individual samples. Reported same-seed differences on H20 have not been traced to a specific operation or independently reproduced here.
