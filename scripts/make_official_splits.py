#!/usr/bin/env python3
"""
Create official train/validation/test splits for SolPepBench / PepSol2000.

This script creates:
  1. pair_stratified_fold0.csv
  2. pair_stratified_5fold.csv
  3. sequence_disjoint_fold0.csv
  4. sequence_disjoint_5fold.csv
  5. solvent_heldout_loso.csv
  6. solvent_heldout_fold0.csv
  7. splits_manifest.json
  8. README.md

The script is intentionally dependency-light and uses only pandas, numpy, and scikit-learn.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


POSITIVE_VALUES = {
    "1", "true", "t", "yes", "y", "pos", "positive",
    "soluble", "sol", "s"
}

NEGATIVE_VALUES = {
    "0", "false", "f", "no", "n", "neg", "negative",
    "insoluble", "insol", "i", "not soluble", "non-soluble", "nonsoluble"
}


def detect_column(df, candidates, required=True, role="column"):
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    # Fuzzy fallback.
    normalized = {c.lower().replace("-", "_").replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not detect {role}. Tried candidates: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_label_series(s):
    """
    Normalize a binary label column to integer 0/1.
    Supports numeric 0/1 and common soluble/insoluble strings.
    """
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().unique()
        allowed = set([0, 1, 0.0, 1.0])
        if set(vals).issubset(allowed):
            return s.astype(int)

    out = []
    bad = []
    for idx, val in s.items():
        if pd.isna(val):
            out.append(np.nan)
            bad.append((idx, val))
            continue
        v = str(val).strip().lower()
        if v in POSITIVE_VALUES:
            out.append(1)
        elif v in NEGATIVE_VALUES:
            out.append(0)
        else:
            # Try numeric-looking values.
            try:
                fv = float(v)
                if fv == 1.0:
                    out.append(1)
                elif fv == 0.0:
                    out.append(0)
                else:
                    out.append(np.nan)
                    bad.append((idx, val))
            except Exception:
                out.append(np.nan)
                bad.append((idx, val))

    y = pd.Series(out, index=s.index, name=s.name)
    if y.isna().any():
        examples = bad[:20]
        raise ValueError(
            f"Label column contains unrecognized or missing values. "
            f"First examples: {examples}"
        )
    return y.astype(int)


def safe_stratify_labels(labels, min_count=2):
    """
    Return labels for stratification only if every class has at least min_count examples.
    Otherwise return None.
    """
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return None
    if counts.min() < min_count:
        return None
    return labels


def assign_splits(record_ids, train_ids, val_ids, test_ids, fold=0):
    split = {}
    for rid in train_ids:
        split[rid] = "train"
    for rid in val_ids:
        split[rid] = "val"
    for rid in test_ids:
        split[rid] = "test"

    rows = []
    for rid in record_ids:
        rows.append({
            "record_id": rid,
            "fold": fold,
            "split": split[rid],
        })
    return pd.DataFrame(rows)


def add_context(split_df, df, record_col, seq_col, solvent_col, label_col):
    context = df[[record_col, seq_col, solvent_col, label_col]].copy()
    context.columns = ["record_id", "sequence", "solvent", "label"]
    context["record_id"] = context["record_id"].astype(str)
    out = split_df.merge(context, on="record_id", how="left")
    return out


def make_pair_stratified_holdout(df, record_col, label_col, seed):
    ids = df[record_col].astype(str).to_numpy()
    y = df[label_col].to_numpy()

    strat = safe_stratify_labels(y, min_count=2)
    train_val_ids, test_ids, train_val_y, _ = train_test_split(
        ids,
        y,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )

    strat2 = safe_stratify_labels(train_val_y, min_count=2)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        shuffle=True,
        stratify=strat2,
    )

    return assign_splits(ids, set(train_ids), set(val_ids), set(test_ids), fold=0)


def make_pair_stratified_5fold(df, record_col, label_col, seed, n_splits=5):
    ids = df[record_col].astype(str).to_numpy()
    y = df[label_col].to_numpy()

    counts = pd.Series(y).value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(ids, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(ids))

    all_rows = []
    for fold_idx in range(n_splits):
        test_idx = folds[fold_idx][1]
        val_idx = folds[(fold_idx + 1) % n_splits][1]
        test_ids = set(ids[test_idx])
        val_ids = set(ids[val_idx])
        train_ids = set(ids) - test_ids - val_ids

        fold_df = assign_splits(ids, train_ids, val_ids, test_ids, fold=fold_idx)
        all_rows.append(fold_df)

    return pd.concat(all_rows, ignore_index=True)


def sequence_group_labels(df, seq_col, label_col):
    g = df.groupby(seq_col)[label_col].agg(["mean", "count", "nunique"]).reset_index()

    def group_label(row):
        if row["nunique"] > 1:
            return "mixed"
        return "positive" if row["mean"] >= 0.5 else "negative"

    g["group_label"] = g.apply(group_label, axis=1)
    return g


def make_sequence_disjoint_holdout(df, record_col, seq_col, label_col, seed):
    seq_df = sequence_group_labels(df, seq_col, label_col)
    seqs = seq_df[seq_col].astype(str).to_numpy()
    group_labels = seq_df["group_label"].astype(str).to_numpy()

    strat = safe_stratify_labels(group_labels, min_count=2)
    train_val_seqs, test_seqs, train_val_labels, _ = train_test_split(
        seqs,
        group_labels,
        test_size=0.15,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )

    strat2 = safe_stratify_labels(train_val_labels, min_count=2)
    train_seqs, val_seqs = train_test_split(
        train_val_seqs,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        shuffle=True,
        stratify=strat2,
    )

    train_seqs = set(train_seqs)
    val_seqs = set(val_seqs)
    test_seqs = set(test_seqs)

    ids = df[record_col].astype(str).to_numpy()
    train_ids = set(df.loc[df[seq_col].astype(str).isin(train_seqs), record_col].astype(str))
    val_ids = set(df.loc[df[seq_col].astype(str).isin(val_seqs), record_col].astype(str))
    test_ids = set(df.loc[df[seq_col].astype(str).isin(test_seqs), record_col].astype(str))

    return assign_splits(ids, train_ids, val_ids, test_ids, fold=0)


def make_sequence_disjoint_5fold(df, record_col, seq_col, label_col, seed, n_splits=5):
    seq_df = sequence_group_labels(df, seq_col, label_col)
    seqs = seq_df[seq_col].astype(str).to_numpy()
    group_labels = seq_df["group_label"].astype(str).to_numpy()

    counts = pd.Series(group_labels).value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(seqs, group_labels))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(seqs))

    all_rows = []
    ids = df[record_col].astype(str).to_numpy()

    for fold_idx in range(n_splits):
        test_seq_idx = folds[fold_idx][1]
        val_seq_idx = folds[(fold_idx + 1) % n_splits][1]

        test_seqs = set(seqs[test_seq_idx])
        val_seqs = set(seqs[val_seq_idx])
        train_seqs = set(seqs) - test_seqs - val_seqs

        train_ids = set(df.loc[df[seq_col].astype(str).isin(train_seqs), record_col].astype(str))
        val_ids = set(df.loc[df[seq_col].astype(str).isin(val_seqs), record_col].astype(str))
        test_ids = set(df.loc[df[seq_col].astype(str).isin(test_seqs), record_col].astype(str))

        fold_df = assign_splits(ids, train_ids, val_ids, test_ids, fold=fold_idx)
        all_rows.append(fold_df)

    return pd.concat(all_rows, ignore_index=True)


def make_solvent_heldout_loso(df, record_col, solvent_col):
    ids = df[record_col].astype(str).to_numpy()
    solvents = sorted(df[solvent_col].astype(str).unique())

    if len(solvents) < 3:
        raise ValueError(
            "Need at least 3 solvent conditions to create train/val/test solvent-held-out splits."
        )

    all_rows = []
    for fold_idx, test_solvent in enumerate(solvents):
        val_solvent = solvents[(fold_idx + 1) % len(solvents)]

        test_ids = set(df.loc[df[solvent_col].astype(str) == test_solvent, record_col].astype(str))
        val_ids = set(df.loc[df[solvent_col].astype(str) == val_solvent, record_col].astype(str))
        train_ids = set(ids) - test_ids - val_ids

        fold_df = assign_splits(ids, train_ids, val_ids, test_ids, fold=fold_idx)
        fold_df["heldout_test_solvent"] = test_solvent
        fold_df["heldout_val_solvent"] = val_solvent
        all_rows.append(fold_df)

    return pd.concat(all_rows, ignore_index=True)


def summarize_split(split_df):
    rows = []
    for fold, fold_df in split_df.groupby("fold"):
        for split_name, part in fold_df.groupby("split"):
            row = {
                "fold": int(fold),
                "split": split_name,
                "n_records": int(len(part)),
            }
            if "label" in part.columns:
                row["positive_rate"] = float(part["label"].mean())
                row["n_positive"] = int(part["label"].sum())
                row["n_negative"] = int((1 - part["label"]).sum())
            if "sequence" in part.columns:
                row["n_unique_sequences"] = int(part["sequence"].nunique())
            if "solvent" in part.columns:
                row["n_unique_solvents"] = int(part["solvent"].nunique())
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/long_peptide_solvent.csv")
    parser.add_argument("--outdir", default="splits")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    seq_col = detect_column(
        df,
        ["sequence", "sequence_clean", "sequence_raw", "peptide_sequence", "peptide", "seq", "Sequence", "Peptide"],
        role="sequence column",
    )
    solvent_col = detect_column(
        df,
        ["solvent", "solvent_condition", "condition", "Solvent", "solvent_name"],
        role="solvent column",
    )
    label_col_original = detect_column(
        df,
        ["label", "y", "target", "soluble", "is_soluble", "binary_label", "solubility_label"],
        role="label column",
    )

    record_col = detect_column(
        df,
        ["record_id", "id", "example_id", "observation_id"],
        required=False,
        role="record id column",
    )

    if record_col is None:
        record_col = "record_id"
        df[record_col] = [str(i) for i in range(len(df))]
    else:
        df[record_col] = df[record_col].astype(str)

    df["_label_binary"] = normalize_label_series(df[label_col_original])
    label_col = "_label_binary"

    # Remove rows with missing sequence or solvent.
    before = len(df)
    df = df.dropna(subset=[seq_col, solvent_col, label_col]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing sequence, solvent, or label.")

    df[seq_col] = df[seq_col].astype(str)
    df[solvent_col] = df[solvent_col].astype(str)

    outputs = {}

    pair0 = make_pair_stratified_holdout(df, record_col, label_col, args.seed)
    pair0 = add_context(pair0, df, record_col, seq_col, solvent_col, label_col)
    path = outdir / "pair_stratified_fold0.csv"
    pair0.to_csv(path, index=False)
    outputs[path.name] = summarize_split(pair0)

    pair5 = make_pair_stratified_5fold(df, record_col, label_col, args.seed, n_splits=5)
    pair5 = add_context(pair5, df, record_col, seq_col, solvent_col, label_col)
    path = outdir / "pair_stratified_5fold.csv"
    pair5.to_csv(path, index=False)
    outputs[path.name] = summarize_split(pair5)

    seq0 = make_sequence_disjoint_holdout(df, record_col, seq_col, label_col, args.seed)
    seq0 = add_context(seq0, df, record_col, seq_col, solvent_col, label_col)
    path = outdir / "sequence_disjoint_fold0.csv"
    seq0.to_csv(path, index=False)
    outputs[path.name] = summarize_split(seq0)

    seq5 = make_sequence_disjoint_5fold(df, record_col, seq_col, label_col, args.seed, n_splits=5)
    seq5 = add_context(seq5, df, record_col, seq_col, solvent_col, label_col)
    path = outdir / "sequence_disjoint_5fold.csv"
    seq5.to_csv(path, index=False)
    outputs[path.name] = summarize_split(seq5)

    solv = make_solvent_heldout_loso(df, record_col, solvent_col)
    solv = add_context(solv, df, record_col, seq_col, solvent_col, label_col)
    # The merge may move heldout columns earlier/later; preserve if present.
    path = outdir / "solvent_heldout_loso.csv"
    solv.to_csv(path, index=False)
    outputs[path.name] = summarize_split(solv)

    solv0 = solv.loc[solv["fold"] == 0].copy()
    path = outdir / "solvent_heldout_fold0.csv"
    solv0.to_csv(path, index=False)
    outputs[path.name] = summarize_split(solv0)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "seed": args.seed,
        "detected_columns": {
            "record_id": record_col,
            "sequence": seq_col,
            "solvent": solvent_col,
            "label": label_col_original,
        },
        "n_records": int(len(df)),
        "n_unique_sequences": int(df[seq_col].nunique()),
        "n_unique_solvents": int(df[solvent_col].nunique()),
        "positive_rate": float(df[label_col].mean()),
        "files": outputs,
    }

    with open(outdir / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    readme = f"""# Official SolPepBench Splits

Generated by `scripts/make_official_splits.py`.

## Input

- `{input_path}`

## Seed

- `{args.seed}`

## Detected columns

- record id: `{record_col}`
- sequence: `{seq_col}`
- solvent: `{solvent_col}`
- label: `{label_col_original}`

## Files

- `pair_stratified_fold0.csv`: single 70/15/15 train/validation/test split stratified by peptide--solvent label.
- `pair_stratified_5fold.csv`: five pair-stratified folds. For each fold, one fold is test, the next fold is validation, and the remaining folds are training.
- `sequence_disjoint_fold0.csv`: single 70/15/15 split in which peptide sequences are disjoint across train/validation/test.
- `sequence_disjoint_5fold.csv`: five sequence-disjoint folds.
- `solvent_heldout_loso.csv`: leave-one-solvent-out style folds. Each fold holds out one solvent for test and the next solvent for validation.
- `solvent_heldout_fold0.csv`: fold 0 extracted from `solvent_heldout_loso.csv`.
- `splits_manifest.json`: machine-readable split statistics.

## Split semantics

- Pair-stratified splits measure in-distribution peptide--solvent prediction.
- Sequence-disjoint splits evaluate generalization to unseen peptide sequences.
- Solvent-held-out splits evaluate transfer to unseen solvent conditions.

Downstream papers should report which split file and fold were used.
"""
    with open(outdir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"Wrote official splits to: {outdir}")
    print(json.dumps({
        "n_records": manifest["n_records"],
        "n_unique_sequences": manifest["n_unique_sequences"],
        "n_unique_solvents": manifest["n_unique_solvents"],
        "positive_rate": manifest["positive_rate"],
        "files": list(outputs.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
