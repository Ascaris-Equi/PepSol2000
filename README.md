# SolPepBench

Anonymous review artifact for:

**SolPepBench: A Solvent-Conditioned Dataset and Benchmark Definition for Peptide Solubility Prediction**

SolPepBench is a solvent-conditioned peptide solubility dataset organized at the peptide--solvent observation level. The main prediction unit is a pair consisting of a peptide sequence and a solvent/formulation condition, with a binary label indicating whether the peptide is soluble under that condition.

## Current anonymous review artifact

This artifact includes:

- raw wide-format input data;
- processed long-format peptide--solvent data;
- quality-control summaries and metadata;
- official split files;
- lightweight reference baseline results;
- checksums;
- Croissant machine-readable metadata;
- scripts for dataset processing, split construction, and baseline evaluation.

## Dataset summary

The current processed release contains:

| Quantity | Value |
|---|---:|
| Raw wide peptide rows | 1,957 |
| Labeled peptide--solvent records | 4,405 |
| Unique peptide sequences | 1,337 |
| Solvent conditions | 7 |
| Soluble records | 2,844 |
| Insoluble records | 1,561 |
| Positive rate | 0.6456 |
| Sequences with discordant labels across solvents | 839 |
| Duplicate sequence--solvent groups | 219 |
| Conflicting sequence--solvent groups | 21 |

## Main files

| Path | Description |
|---|---|
| `data/raw/pep.csv` | Raw wide-format peptide table |
| `data/processed/long_peptide_solvent.csv` | Main long-format peptide--solvent table |
| `data/processed/summary.json` | Machine-readable dataset summary |
| `data/metadata/dataset_metadata.json` | Dataset-level metadata |
| `data/metadata/data_dictionary.csv` | Data dictionary |
| `data/metadata/checksums.sha256` | SHA-256 checksums |
| `data/metadata/croissant.json` | Croissant machine-readable metadata |
| `splits/` | Official split files |
| `results/baselines/` | Reference baseline results |
| `scripts/build_dataset.py` | Dataset construction script |
| `scripts/make_official_splits.py` | Official split construction script |
| `scripts/run_reference_baselines.py` | Reference baseline script |
| `scripts/make_croissant.py` | Croissant metadata generation script |

## Installation

A minimal Python environment is sufficient for the official split and baseline scripts.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-minimal.txt

If using conda:

    conda create -n solpepbench python=3.11 -y
    conda activate solpepbench
    pip install -r requirements-minimal.txt

## Reproduce official splits

From the repository root:

    python scripts/make_official_splits.py \
      --input data/processed/long_peptide_solvent.csv \
      --outdir splits \
      --seed 2026

Expected output files include:

- `splits/pair_stratified_fold0.csv`
- `splits/pair_stratified_5fold.csv`
- `splits/sequence_disjoint_fold0.csv`
- `splits/sequence_disjoint_5fold.csv`
- `splits/solvent_heldout_loso.csv`
- `splits/solvent_heldout_fold0.csv`
- `splits/splits_manifest.json`

## Reproduce reference baselines

From the repository root:

    python scripts/run_reference_baselines.py \
      --data data/processed/long_peptide_solvent.csv \
      --splits splits \
      --outdir results/baselines \
      --seed 2026

Expected output files include:

- `results/baselines/baseline_results.csv`
- `results/baselines/baseline_summary.csv`
- `results/baselines/baseline_table.tex`
- `results/baselines/baseline_metadata.json`
- `results/baselines/baseline_errors.csv`

The reference models are intentionally lightweight and reproducible. They are intended as sanity-check and reference baselines, not as optimized state-of-the-art models.

## Regenerate Croissant metadata

    python scripts/make_croissant.py
    python -m json.tool data/metadata/croissant.json >/tmp/croissant.pretty.json

## Rebuild processed data from raw input

The processed release files are included in the artifact. The preprocessing script used for dataset construction is:

    python scripts/build_dataset.py

If modifying preprocessing logic, regenerate processed outputs, official splits, baseline results, checksums, and Croissant metadata before reporting new results.

## Official evaluation protocols

The release includes three official evaluation families:

1. **Pair-stratified**: peptide--solvent observations are split while preserving label balance.
2. **Sequence-disjoint**: peptide sequences are not shared between train and test.
3. **Solvent-held-out**: solvent identities are held out to evaluate transfer across solvent conditions.

The `splits/` directory is part of the released benchmark artifact. Downstream users should report the exact split files, random seed, repository version, and checksum used in experiments.

## Intended use

SolPepBench is intended for:

- dataset analysis of solvent-conditioned peptide solubility labels;
- development and evaluation of solvent-aware peptide solubility predictors;
- studying when sequence-only label collapse discards condition-dependent variation;
- reproducible benchmarking under pair-stratified, sequence-disjoint, and solvent-held-out protocols.

## Out-of-scope use

SolPepBench is not intended for:

- clinical decision-making;
- manufacturing release decisions;
- regulatory submission;
- claims about biological efficacy or safety;
- replacing experimental solubility testing.

Predictions from models trained on this dataset should be treated as hypotheses for experimental follow-up, not definitive formulation decisions.

## Licenses

Code is released under the MIT license. Data, metadata, documentation, split files, and result tables are released under CC BY 4.0 for the anonymous review artifact.

## Anonymous review note

This repository is prepared as an anonymous review artifact. The public camera-ready release should replace anonymous metadata with final author, repository, archival DOI, and citation information.
