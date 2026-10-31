# Source provenance and extraction scope

On 2026-10-31, the Python package, CLI, tutorials, and separation of source code, resources, and containers were organized with reference to [BioEmu2](https://github.com/microsoft/bioemu2) and [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina). Their source code, models, installation versions, and licenses were not copied.

The batch entry points `batch_predict.py`, `batch_runtime.py`, and `oridynamic/__main__.py` reuse the local `inference_runtime.py`, `local_msa.py`, and `runtime_utils.py`. All base anchors and frozen contexts are prepared before Joint loading and ensemble generation. The entry points do not import the research repository. Original source fingerprints are retained; current release hashes are recorded in `code_manifest.json`.

On 2026-10-30, the necessary inference implementations were extracted from the verified project snapshot. Original file hashes are recorded in `runtime_sources.json`. Source paths in that record provide provenance and are not runtime import paths. Attribution does not grant additional redistribution rights.

| Release location | Retained | Excluded |
| :--- | :--- | :--- |
| `protenix/model/protenix.py` | Native network and inference path, preserving checkpoint keys | Training loops, training forward path, training noise sampler |
| `protenix/model/modules/all_atom_struct_vae.py` | Reference encoding cache, displacement decoding, checkpoint keys | Training forward path, KL objective, training initialization helpers |
| `protenix/model/modules/latent_dit.py` | Existing DiT network and flow velocity prediction | Flow-matching training loop, objective, and optimizer |
| `protenix/model/modules/joint_ensemble.py` | Existing generation/refinement, Heun/SDE sampling, dual contexts | Replay/native objectives, ensemble losses, training forward path |
| `runtime_utils.py` | Strict loading, atom identities, chunking, and CIF writing extracted from earlier runners | Earlier training entry points, MD loading, training-state saving |
| `protenix/data/inference/` and feature dependencies | Native JSON, CCD, and MSA features | Training datasets, mixtures, and trajectory adapters |
| `runner/msa_search.py` and dependencies | Legacy native online-search support; local search uses `local_msa.py` | Training runners and research batch experiments |
| `configs/inference_defaults.json` | Inference settings and confidence bins | Dataset paths, optimizers, schedulers, and loss weights |
| `benchmark_batch.py`, `compare_runs.py` | Generic timing with existing MSAs and comparisons using a common anchor | Research data and new search/model implementations |

The extraction preserves required parameters, random-feature buffers, and state-dict keys for strict base/codec/Joint loading. The Joint checkpoint includes the DiT. During loading, `requires_grad` is used to validate the historical checkpoint parameter groups; all parameters are frozen afterward. Training-mode entry points are rejected. These restrictions define the distributed inference interface, rather than preventing downstream users from developing their own training code.

The default Torch LayerNorm follows the tested runtime without a fused LayerNorm compilation dependency. Native low-level operators may retain general-purpose autograd, training flags, or checkpointing utilities; these are network dependencies, not the AE/Stage-B/Joint training recipe.

The public inputs support monomers, PL complexes, and PP assemblies with whole-complex sampling. Execution checks and task-specific quality evaluations are distinct; see the [model notes](docs/model_card.md).

Local MSA search calls the public ColabFold/MMseqs2 CLI. It does not copy private search orchestration code. Model weights, CCD resources, and MSA databases are supplied separately. The upstream [LICENSE-PTX](LICENSE-PTX) and source copyright headers are retained. Local additions, weights, and databases require their own applicable redistribution terms.

Predictions do not read ground-truth structures or MD trajectories for selection. Default predicted anchors, user-provided structures, and historical multi-anchor experiments should be documented separately when comparing results.
