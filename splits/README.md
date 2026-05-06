# Official SolPepBench split files

This directory contains official split files for the anonymous SolPepBench review artifact.

The split files were generated with:

    python scripts/make_official_splits.py \
      --input data/processed/long_peptide_solvent.csv \
      --outdir splits \
      --seed 2026

## Files

- `pair_stratified_fold0.csv`
- `pair_stratified_5fold.csv`
- `sequence_disjoint_fold0.csv`
- `sequence_disjoint_5fold.csv`
- `solvent_heldout_loso.csv`
- `solvent_heldout_fold0.csv`
- `splits_manifest.json`

## Protocols

- Pair-stratified splits evaluate in-distribution peptide--solvent prediction.
- Sequence-disjoint splits evaluate generalization to unseen peptide sequences.
- Solvent-held-out splits evaluate transfer across solvent conditions.

Downstream users should report the exact split files, seed, repository version, and checksums used in experiments.
