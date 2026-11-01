# Monomer, protein–ligand, and protein–protein inputs

Use `oridynamic sample --input-json complex.json ...` with a Protenix task array. Each entry in `sequences` describes an entity using `proteinChain` or `ligand`. All chains and ligands within one task are processed together through structure prediction, ensemble generation, and CIF export.

Multiple FASTA entries describe independent monomers. Use JSON to describe interacting chains in one complex.

## Input format

```json
[
  {
    "name": "my_complex",
    "sequences": [
      {"proteinChain": {"id": ["A"], "count": 1, "sequence": "ACDEFGHIK", "unpairedMsaPath": "msas/A.a3m"}},
      {"proteinChain": {"id": ["B"], "count": 1, "sequence": "LMNPQRSTV", "unpairedMsaPath": "msas/B.a3m"}},
      {"ligand": {"id": ["L"], "count": 1, "ligand": "CCD_ATP"}}
    ]
  }
]
```

These short sequences illustrate the format. The [five examples with bundled MSAs](../examples/five_cases_with_msa.json) contain real input sequences and alignment paths. Specify a unique list of chain IDs for each entity; multiple copies of the same entity can use `id: ["A", "B"], count: 2`.

Relative MSA, ligand-file, and initial-structure paths are resolved from the input JSON directory. Protein sequences must use the 20 standard amino acids. The public adapter accepts `proteinChain` and `ligand` entities and rejects unsupported task/entity fields. Nucleic acids, standalone ion entities, modified protein residues, and inter-entity covalent constraints are outside this release interface.

## MSAs and chemical reference features

- Each protein entity can provide `unpairedMsaPath` and an optional `pairedMsaPath`. The query sequence is validated for each alignment.
- When an unpaired MSA is absent, `--msa-local-db` searches separately for each entity. Copies of the same entity share an MSA.
- Local search produces unpaired alignments. Supplied paired A3Ms are passed to the native Protenix pairing logic.
- `--single-sequence` permits query-only input for protein entities without a supplied MSA, for smoke tests. The batch entry point does not contact an online MSA service automatically.
- Ligands such as `CCD_75G` are resolved against the local CCD resources. SMILES and `FILE_relative/path.mol` inputs use native Protenix parsing and conformer generation.
- Chemical reference features (`ref_pos`) and the full-complex decoder anchor have different roles. The native featurizer supplies chemical graphs, atom names, elements, charges, validity masks, bonds, and atom-to-token mappings; ligand atoms remain part of ensemble generation.

## Optional initial structure

Add these fields to the task object:

```json
"initial_structure": "anchors/full_complex.cif",
"structure_chains": {"A": "X", "B": "Y", "L": "Z"}
```

Mapping keys are input JSON chain IDs; values are author chain IDs in the initial structure. Without this mapping, matching chain IDs are required.

The structure must contain all input chains and the required heavy atoms. Residue order, component names, atom names, and elements are matched explicitly. CCD ligands must use the corresponding CCD atom names. SMILES/MOL atom names follow the Protenix conventions; the loader does not infer an isomorphism or add missing atoms.

An `anchor.cif` produced by this entry point can be reused directly. Homomeric chains are not automatically permuted, and ligands are not aligned or centered independently. After mapping, one canonical frame is established for the complete complex.

Without a supplied structure, Protenix predicts the entire complex from its protein and ligand inputs. The native `ranking_score` selects the anchor. The resulting `anchor.cif`, `proposals.cif`, and `ensemble.cif` include every chain and ligand, with identity and coordinate checks on readback.

All supported input types use the same `ae.ckpt` and `oridynamic.ckpt` files. See [model and sampling settings](model_card.md).
