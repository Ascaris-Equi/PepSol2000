#!/usr/bin/env python3
"""
Run lightweight reference baselines for SolPepBench / PepSol2000.

The goal of these baselines is not to claim state of the art.
They provide reproducible reference numbers for official splits.

Models:
  - majority_class
  - stratified_dummy
  - sequence_logistic
  - solvent_logistic
  - sequence_solvent_logistic
  - sequence_solvent_random_forest
  - sequence_solvent_gradient_boosting

Outputs:
  - baseline_results.csv
  - baseline_summary.csv
  - baseline_table.tex
  - baseline_errors.csv
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


POSITIVE_VALUES = {
    "1", "true", "t", "yes", "y", "pos", "positive",
    "soluble", "sol", "s"
}

NEGATIVE_VALUES = {
    "0", "false", "f", "no", "n", "neg", "negative",
    "insoluble", "insol", "i", "not soluble", "non-soluble", "nonsoluble"
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

AA_WEIGHTS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


def one_hot_encoder_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_column(df, candidates, required=True, role="column"):
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

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


def clean_sequence(seq):
    if pd.isna(seq):
        return ""
    return "".join([c for c in str(seq).upper().strip() if c.isalpha()])


def sequence_features(seq):
    seq = clean_sequence(seq)
    n = len(seq)
    counts = {aa: seq.count(aa) for aa in AA_LIST}

    if n == 0:
        base = {
            "seq_length": 0,
            "mw_approx": 0.0,
            "hydropathy_mean": 0.0,
            "frac_unknown": 1.0,
        }
        for aa in AA_LIST:
            base[f"count_{aa}"] = 0
            base[f"frac_{aa}"] = 0.0
        for name in [
            "frac_hydrophobic", "frac_polar", "frac_positive", "frac_negative",
            "frac_charged", "frac_aromatic", "frac_glycine", "frac_proline",
            "net_charge_proxy", "abs_net_charge_proxy",
        ]:
            base[name] = 0.0
        return base

    known = sum(counts.values())
    unknown = max(0, n - known)

    mw = sum(AA_WEIGHTS.get(c, 0.0) for c in seq)
    hydro_vals = [KYTE_DOOLITTLE[c] for c in seq if c in KYTE_DOOLITTLE]
    hydro = float(np.mean(hydro_vals)) if hydro_vals else 0.0

    hydrophobic = set("AILMFWYV")
    polar = set("STNQCY")
    positive = set("KRH")
    negative = set("DE")
    charged = positive | negative
    aromatic = set("FWY")

    pos_count = sum(counts[a] for a in positive)
    neg_count = sum(counts[a] for a in negative)
    net_charge = pos_count - neg_count

    feat = {
        "seq_length": n,
        "mw_approx": mw,
        "hydropathy_mean": hydro,
        "frac_unknown": unknown / n,
        "frac_hydrophobic": sum(counts[a] for a in hydrophobic) / n,
        "frac_polar": sum(counts[a] for a in polar) / n,
        "frac_positive": pos_count / n,
        "frac_negative": neg_count / n,
        "frac_charged": sum(counts[a] for a in charged) / n,
        "frac_aromatic": sum(counts[a] for a in aromatic) / n,
        "frac_glycine": counts["G"] / n,
        "frac_proline": counts["P"] / n,
        "net_charge_proxy": net_charge,
        "abs_net_charge_proxy": abs(net_charge),
    }

    for aa in AA_LIST:
        feat[f"count_{aa}"] = counts[aa]
        feat[f"frac_{aa}"] = counts[aa] / n

    return feat


def build_feature_frame(df, seq_col, solvent_col):
    seq_feats = [sequence_features(s) for s in df[seq_col]]
    X = pd.DataFrame(seq_feats)
    X["solvent_norm"] = df[solvent_col].astype(str).str.strip().str.lower().values
    return X


def split_file_protocol(path):
    name = path.name
    if name.startswith("pair_stratified"):
        return "pair-stratified"
    if name.startswith("sequence_disjoint"):
        return "sequence-disjoint"
    if name.startswith("solvent_heldout"):
        return "solvent-held-out"
    return path.stem


def get_model_specs(seed, numeric_cols):
    """
    Return baseline model specifications.
    feature_mode:
      - dummy: uses all columns but ignores them
      - sequence: numeric sequence descriptors only
      - solvent: solvent one-hot only
      - sequence_solvent: numeric sequence descriptors + solvent one-hot
    """

    seq_preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    seq_preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    solvent_preprocess = ColumnTransformer(
        transformers=[
            ("solvent", one_hot_encoder_dense(), ["solvent_norm"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    seq_solv_preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("solvent", one_hot_encoder_dense(), ["solvent_norm"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    seq_solv_preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
            ("solvent", one_hot_encoder_dense(), ["solvent_norm"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return {
        "majority_class": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "stratified_dummy": Pipeline([
            ("clf", DummyClassifier(strategy="stratified", random_state=seed))
        ]),
        "sequence_logistic": Pipeline([
            ("prep", seq_preprocess_scaled),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ]),
        "solvent_logistic": Pipeline([
            ("prep", solvent_preprocess),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ]),
        "sequence_solvent_logistic": Pipeline([
            ("prep", seq_solv_preprocess_scaled),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ]),
        "sequence_solvent_random_forest": Pipeline([
            ("prep", seq_solv_preprocess_tree),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "sequence_solvent_gradient_boosting": Pipeline([
            ("prep", seq_solv_preprocess_tree),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]),
    }


def safe_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        # Single-class fallback.
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1:
            return np.ones(len(X)) if classes[0] == 1 else np.zeros(len(X))
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Logistic-ish squashing.
        return 1.0 / (1.0 + np.exp(-scores))
    pred = model.predict(X)
    return pred.astype(float)


def metric_dict(y_true, y_pred, y_prob):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)
    out["kappa"] = cohen_kappa_score(y_true, y_pred)

    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    try:
        out["brier"] = brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0))
    except Exception:
        out["brier"] = np.nan

    return out


def load_split_assignments(split_path):
    split_df = pd.read_csv(split_path)
    if "record_id" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError(f"Split file {split_path} must contain record_id and split columns.")
    if "fold" not in split_df.columns:
        split_df["fold"] = 0
    split_df["record_id"] = split_df["record_id"].astype(str)
    return split_df


def format_mean_std(mean, std, decimals=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def write_latex_table(summary_df, out_path):
    """
    Write a compact LaTeX tabular body.

    The manuscript can include:

    \\begin{table}[t]
    \\centering
    \\small
    \\caption{Reference baseline performance.}
    \\label{tab:baseline_results}
    \\input{results/baselines/baseline_table.tex}
    \\end{table}
    """
    if summary_df.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("% No baseline results available.\n")
        return

    # Prefer one row per protocol/model.
    display = summary_df.copy()

    preferred_models = [
        "majority_class",
        "stratified_dummy",
        "sequence_logistic",
        "solvent_logistic",
        "sequence_solvent_logistic",
        "sequence_solvent_random_forest",
        "sequence_solvent_gradient_boosting",
    ]
    display["model_order"] = display["model"].apply(
        lambda m: preferred_models.index(m) if m in preferred_models else 999
    )
    protocol_order = {
        "pair-stratified": 0,
        "sequence-disjoint": 1,
        "solvent-held-out": 2,
    }
    display["protocol_order"] = display["protocol"].map(protocol_order).fillna(999)
    display = display.sort_values(["protocol_order", "model_order", "model"])

    lines = []
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Protocol} & \\textbf{Model} & \\textbf{AUROC} & \\textbf{F1} & \\textbf{MCC} & \\textbf{Acc.} \\\\")
    lines.append("\\midrule")

    current_protocol = None
    for _, row in display.iterrows():
        protocol = row["protocol"]
        if current_protocol is not None and protocol != current_protocol:
            lines.append("\\midrule")
        current_protocol = protocol

        model_name = row["model"].replace("_", "\\_")
        auroc = format_mean_std(row.get("roc_auc_mean", np.nan), row.get("roc_auc_std", np.nan))
        f1 = format_mean_std(row.get("f1_mean", np.nan), row.get("f1_std", np.nan))
        mcc = format_mean_std(row.get("mcc_mean", np.nan), row.get("mcc_std", np.nan))
        acc = format_mean_std(row.get("accuracy_mean", np.nan), row.get("accuracy_std", np.nan))

        lines.append(f"{protocol} & {model_name} & {auroc} & {f1} & {mcc} & {acc} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/long_peptide_solvent.csv")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--outdir", default="results/baselines")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    data_path = Path(args.data)
    splits_dir = Path(args.splits)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

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
    df = df.dropna(subset=[seq_col, solvent_col, "_label_binary"]).copy()
    df[seq_col] = df[seq_col].astype(str)
    df[solvent_col] = df[solvent_col].astype(str)
    df[record_col] = df[record_col].astype(str)

    X = build_feature_frame(df, seq_col, solvent_col)
    y = df["_label_binary"].astype(int).to_numpy()
    record_ids = df[record_col].astype(str).to_numpy()

    id_to_pos = {rid: i for i, rid in enumerate(record_ids)}
    numeric_cols = [c for c in X.columns if c != "solvent_norm"]

    model_specs = get_model_specs(args.seed, numeric_cols)

    split_files = [
        splits_dir / "pair_stratified_fold0.csv",
        splits_dir / "sequence_disjoint_fold0.csv",
        splits_dir / "solvent_heldout_loso.csv",
    ]
    split_files = [p for p in split_files if p.exists()]

    if not split_files:
        raise FileNotFoundError(
            f"No split files found in {splits_dir}. "
            f"Run scripts/make_official_splits.py first."
        )

    results = []
    errors = []

    for split_path in split_files:
        protocol = split_file_protocol(split_path)
        split_df = load_split_assignments(split_path)

        for fold in sorted(split_df["fold"].unique()):
            fold_df = split_df.loc[split_df["fold"] == fold].copy()

            train_ids = fold_df.loc[fold_df["split"] == "train", "record_id"].astype(str).tolist()
            val_ids = fold_df.loc[fold_df["split"] == "val", "record_id"].astype(str).tolist()
            test_ids = fold_df.loc[fold_df["split"] == "test", "record_id"].astype(str).tolist()

            # Baselines train on train only; validation is released for future tuning.
            train_idx = [id_to_pos[rid] for rid in train_ids if rid in id_to_pos]
            test_idx = [id_to_pos[rid] for rid in test_ids if rid in id_to_pos]

            if len(train_idx) == 0 or len(test_idx) == 0:
                errors.append({
                    "protocol": protocol,
                    "split_file": split_path.name,
                    "fold": int(fold),
                    "model": "__split__",
                    "error": "Empty train or test split after matching record_id.",
                })
                continue

            X_train = X.iloc[train_idx].copy()
            y_train = y[train_idx]
            X_test = X.iloc[test_idx].copy()
            y_test = y[test_idx]

            for model_name, model in model_specs.items():
                try:
                    fitted = model.fit(X_train, y_train)
                    y_pred = fitted.predict(X_test)
                    y_prob = safe_proba(fitted, X_test)

                    m = metric_dict(y_test, y_pred, y_prob)
                    row = {
                        "protocol": protocol,
                        "split_file": split_path.name,
                        "fold": int(fold),
                        "model": model_name,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "train_positive_rate": float(np.mean(y_train)),
                        "test_positive_rate": float(np.mean(y_test)),
                    }
                    row.update(m)
                    results.append(row)

                except Exception as e:
                    errors.append({
                        "protocol": protocol,
                        "split_file": split_path.name,
                        "fold": int(fold),
                        "model": model_name,
                        "error": repr(e),
                    })

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    results_path = outdir / "baseline_results.csv"
    summary_path = outdir / "baseline_summary.csv"
    errors_path = outdir / "baseline_errors.csv"
    latex_path = outdir / "baseline_table.tex"

    results_df.to_csv(results_path, index=False)
    errors_df.to_csv(errors_path, index=False)

    if not results_df.empty:
        metric_cols = [
            "accuracy", "balanced_accuracy", "precision", "recall", "f1",
            "mcc", "kappa", "roc_auc", "average_precision", "brier",
        ]

        grouped = results_df.groupby(["protocol", "model"], as_index=False)
        summary_parts = []

        for (protocol, model), part in grouped:
            row = {
                "protocol": protocol,
                "model": model,
                "n_folds": int(part["fold"].nunique()),
                "mean_n_train": float(part["n_train"].mean()),
                "mean_n_test": float(part["n_test"].mean()),
                "mean_train_positive_rate": float(part["train_positive_rate"].mean()),
                "mean_test_positive_rate": float(part["test_positive_rate"].mean()),
            }
            for metric in metric_cols:
                row[f"{metric}_mean"] = float(part[metric].mean(skipna=True))
                row[f"{metric}_std"] = float(part[metric].std(skipna=True))
            summary_parts.append(row)

        summary_df = pd.DataFrame(summary_parts)
    else:
        summary_df = pd.DataFrame()

    summary_df.to_csv(summary_path, index=False)
    write_latex_table(summary_df, latex_path)

    metadata = {
        "data": str(data_path),
        "splits_dir": str(splits_dir),
        "outdir": str(outdir),
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
        "positive_rate": float(np.mean(y)),
        "models": list(model_specs.keys()),
        "split_files": [p.name for p in split_files],
        "n_result_rows": int(len(results_df)),
        "n_error_rows": int(len(errors_df)),
    }
    with open(outdir / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote baseline results to: {outdir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
