# Sources and scope

OriDynamic reuses Protenix network and molecular-feature implementations. The upstream [LICENSE-PTX](../LICENSE-PTX) and copyright notices in source files are retained. Dependencies, weights, and databases remain subject to their respective terms.

The repository contains the inference portions of the structural autoencoder, latent flow module, coordinate refiner, and supporting cofolding network. Training loops, optimizers, replay objectives, MD loaders, and training datasets are excluded. Matching model weights and CCD resources are supplied separately; see the [installation guide](../INSTALL.md).

| Location | Purpose |
| :--- | :--- |
| `oridynamic/` | Public CLI, input preparation, batch orchestration, MSA search adapter, and inference runtime |
| `protenix/`, `runner/`, `configs/` | Required network, feature, and sampling implementations and inference configuration |
| `examples/` | Five sequence/composition inputs and six matching precomputed protein MSAs |
| `docs/`, `docker/` | User guides and environment build recipes |
| `tests/` | CPU regression and installation tests |

The runtime imports only the distributed inference code. Checkpoint parameter names and sampling algorithms are retained. The Joint checkpoint includes the latent flow module, and all model parameters are frozen during inference.

Local MSA search calls the public ColabFold/MMseqs2 CLI. The six example alignments are reused from the existing ATLAS, MISATO, and DynaPPI preparation; their sequence matching and the 5T3N column crop are described in the [example guide](../examples/README.md). Full search databases are not bundled.

Package and documentation organization was informed by [BioEmu2](https://github.com/microsoft/bioemu2) and [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina). Their source code, models, training and sampling implementations, and licenses were not copied.

Predictions do not read ground-truth structures or MD trajectories for selection. Execution checks and task-specific quality evaluations are distinct; see the [model notes](model_card.md).
