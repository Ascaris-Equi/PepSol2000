# PepSol2000 / SolPepBench

**PepSol2000** is the repository for **SolPepBench**, a solvent-conditioned peptide solubility dataset and benchmark.

The central design choice of SolPepBench is to treat solubility as a property of a **peptide-solvent observation**, rather than as a sequence-only property. The same peptide sequence may therefore have different solubility labels under different solvent or formulation conditions.

## Overview

SolPepBench is intended for binary classification of peptide solubility under solvent-specific conditions.

Given:

```text
peptide sequence + solvent / formulation condition
```

the task is to predict:

```text
y = 1  soluble
y = 0  insoluble
```

## Current processed release

The current processed release contains peptide-solvent-level solubility records generated from the raw wide-format input table.

| Quantity | Value |
|---|---:|
| Raw wide peptide rows | 1,957 |
| Labeled peptide-solvent records | 4,405 |
| Unique peptides in long table | 1,337 |
| Solvents with labels | 7 |
| Soluble records | 2,844 |
| Insoluble records | 1,561 |
| Positive rate | 0.6456 |
| Sequences tested in multiple solvents | 1,337 |
| Discordant solvent-label sequences | 839 |
| Duplicate sequence-solvent groups | 219 |
| Conflicting sequence-solvent groups | 21 |

The main processed long-format table is:

```text
data/processed/long_peptide_solvent.csv
```

This table represents the dataset at the peptide-solvent observation level.

## Repository structure

```text
data/
  raw/
    pep.csv

  processed/
    clean_wide.csv
    long_peptide_solvent.csv
    features_for_model.csv
    peptide_features.csv
    solvent_summary.csv
    sequence_label_variability.csv
    duplicate_conflicts.csv
    sequence_qc.csv
    mw_discrepancies.csv
    unknown_label_values.csv
    solvent_metadata_template.csv
    summary.json
    paper_numbers.txt

  metadata/
    dataset_metadata.json
    data_dictionary.csv
    checksums.sha256

docs/
  paper_numbers.txt

scripts/
  build_dataset.py

baselines/
  README.md

splits/
  README.md
```

## Key files

| File | Description |
|---|---|
| `data/raw/pep.csv` | Raw wide-format peptide table. |
| `data/processed/long_peptide_solvent.csv` | Main long-format peptide-solvent solubility table. |
| `data/processed/features_for_model.csv` | Feature-oriented table prepared for downstream modeling. |
| `data/processed/peptide_features.csv` | Peptide-level derived feature table. |
| `data/processed/solvent_summary.csv` | Solvent-level label summary. |
| `data/processed/sequence_label_variability.csv` | Sequence-level variability across solvent conditions. |
| `data/processed/duplicate_conflicts.csv` | Duplicate sequence-solvent groups with conflicting labels. |
| `data/processed/summary.json` | Machine-readable summary statistics. |
| `docs/paper_numbers.txt` | Human-readable summary numbers for manuscript writing. |
| `data/metadata/dataset_metadata.json` | Dataset-level metadata. |
| `data/metadata/data_dictionary.csv` | Initial data dictionary. |
| `data/metadata/checksums.sha256` | SHA-256 checksums for raw, processed, and metadata files. |
| `scripts/build_dataset.py` | Dataset construction script used to generate the processed outputs. |

## Prediction unit

The prediction unit is a **peptide-solvent observation**.

This means that the same peptide sequence can appear more than once if it was measured or labeled under different solvent conditions. This structure is intentional and is central to the benchmark.

A simplified example is:

| Peptide sequence | Solvent condition | Label |
|---|---|---:|
| PEPTIDEA | solvent A | 1 |
| PEPTIDEA | solvent B | 0 |

Such cases are not treated as simple duplicates. They are evidence that peptide solubility may be condition-dependent.

## Why solvent-conditioned solubility?

Many peptide solubility datasets are organized as if each peptide sequence has a single intrinsic solubility label. In practice, solubility depends on experimental or formulation conditions, including solvent environment.

SolPepBench is organized to support models and analyses that account for this condition dependence.

## Current status

This repository currently provides:

- raw input data;
- processed peptide-solvent-level tables;
- dataset-level summary statistics;
- metadata files;
- checksums;
- the dataset construction script;
- placeholder directories for future baselines and official splits.

The following components may be added in later releases:

- official train/validation/test splits;
- baseline model training code;
- evaluation scripts;
- Croissant metadata;
- automated reporting utilities.

## Intended use

SolPepBench is intended for:

- benchmarking solvent-conditioned peptide solubility prediction methods;
- studying peptide solubility variation across solvent conditions;
- developing peptide-solvent representation learning methods;
- supporting manuscript-level analysis of condition-dependent peptide solubility.

## Out-of-scope use

This dataset is not intended for:

- direct clinical decision-making;
- manufacturing release decisions;
- replacement of experimental solubility testing;
- claims about peptide biological activity, toxicity, safety, or efficacy.

## Reproducibility notes

Processed files in `data/processed/` were generated from the raw input table using:

```text
scripts/build_dataset.py
```

The summary statistics used in the manuscript should be taken from:

```text
docs/paper_numbers.txt
data/processed/summary.json
data/metadata/dataset_metadata.json
```

## License

This repository uses separate licenses for code and data.

- Code: MIT License. See `LICENSE-CODE.md`.
- Data, metadata, and documentation: Creative Commons Attribution 4.0 International License. See `LICENSE-DATA.md`.

## Citation

A formal citation will be added after manuscript submission or publication.

For now, please cite this repository as the initial SolPepBench / PepSol2000 dataset release.
