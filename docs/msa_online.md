# Preparing MSAs with an online service

The batch entry point consumes existing MSAs or searches a local database. Online MSA preparation is a separate step that sends sequences to a third-party service. Use local search for confidential sequences.

Follow the [ColabFold instructions](https://github.com/sokrypton/ColabFold#generating-msas-for-small-scale-local-structurecomplex-predictions-using-the-msa-server) to install a compatible environment, then run on a machine with internet access:

```bash
colabfold_batch sequences.fasta online_msas --msa-only
```

This command generates MSAs only. It does not invoke OriDynamic or predict an initial structure. Follow the service's usage terms and rate limits. The official Colab interface can also be used to obtain A3M files.

Copy the alignments to the inference machine and check that each query sequence matches its FASTA input exactly. If alignment filenames differ from the first words of the FASTA headers, specify their `msa_a3m` paths explicitly in a [request manifest](batch_inference.md#inputs-and-execution).

Run inference locally:

```bash
python -B -u -m oridynamic --manifest request.json \
  --seeds 0,1 --samples-per-seed 8 --out runs/from_online_msas_v1
```

Report online preparation time separately from local inference time. The single-target `python -m oridynamic.predict` entry point retains its earlier online-search compatibility, while the recommended `oridynamic sample` entry point accepts existing MSAs, local search, or explicit query-only smoke tests.
