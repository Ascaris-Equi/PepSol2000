#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements-minimal.txt",
    "LICENSE",
    "LICENSE-DATA",
    "data/raw/pep.csv",
    "data/processed/long_peptide_solvent.csv",
    "data/processed/summary.json",
    "data/metadata/dataset_metadata.json",
    "data/metadata/data_dictionary.csv",
    "data/metadata/checksums.sha256",
    "data/metadata/croissant.json",
    "scripts/build_dataset.py",
    "scripts/make_official_splits.py",
    "scripts/run_reference_baselines.py",
    "scripts/make_croissant.py",
    "splits/splits_manifest.json",
    "splits/pair_stratified_fold0.csv",
    "splits/pair_stratified_5fold.csv",
    "splits/sequence_disjoint_fold0.csv",
    "splits/sequence_disjoint_5fold.csv",
    "splits/solvent_heldout_loso.csv",
    "splits/solvent_heldout_fold0.csv",
    "results/baselines/baseline_metadata.json",
    "results/baselines/baseline_results.csv",
    "results/baselines/baseline_summary.csv",
    "results/baselines/baseline_table.tex",
    "results/baselines/baseline_errors.csv",
]

EXPECTED = {
    "n_records": 4405,
    "n_unique_sequences": 1337,
    "n_unique_solvents": 7,
    "positive_rate": 0.6456299659477867,
}

SPLIT_FILES = [
    "splits/pair_stratified_fold0.csv",
    "splits/pair_stratified_5fold.csv",
    "splits/sequence_disjoint_fold0.csv",
    "splits/sequence_disjoint_5fold.csv",
    "splits/solvent_heldout_loso.csv",
    "splits/solvent_heldout_fold0.csv",
]

def main() -> int:
    errors = []
    warnings = []

    for r in REQUIRED_FILES:
        p = ROOT / r
        if not p.exists():
            errors.append(f"Missing required file: {r}")
        elif p.is_file() and p.stat().st_size == 0:
            warnings.append(f"File exists but is empty: {r}")

    data_path = ROOT / "data/processed/long_peptide_solvent.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)

        for c in ["observation_id", "sequence_clean", "solvent_name", "y"]:
            if c not in df.columns:
                errors.append(f"Missing column in long table: {c}")

        if len(df) != EXPECTED["n_records"]:
            errors.append(f"Unexpected n_records: {len(df)}")

        if "observation_id" in df.columns and not df["observation_id"].is_unique:
            errors.append("observation_id is not unique")

        if "sequence_clean" in df.columns:
            n = df["sequence_clean"].nunique()
            if n != EXPECTED["n_unique_sequences"]:
                errors.append(f"Unexpected n_unique_sequences: {n}")

        if "solvent_name" in df.columns:
            n = df["solvent_name"].nunique()
            if n != EXPECTED["n_unique_solvents"]:
                errors.append(f"Unexpected n_unique_solvents: {n}")

        if "y" in df.columns:
            pos = float(df["y"].mean())
            if abs(pos - EXPECTED["positive_rate"]) > 1e-12:
                errors.append(f"Unexpected positive_rate: {pos}")
            if not set(df["y"].dropna().unique()).issubset({0, 1}):
                errors.append("y contains non-binary values")

        obs_ids = set(df["observation_id"].astype(str)) if "observation_id" in df.columns else set()

        for sf in SPLIT_FILES:
            p = ROOT / sf
            if not p.exists():
                continue
            sdf = pd.read_csv(p)
            if len(sdf) == 0:
                errors.append(f"Split file is empty: {sf}")
                continue

            id_col = None
            for candidate in ["observation_id", "record_id", "id"]:
                if candidate in sdf.columns:
                    id_col = candidate
                    break

            if id_col is None:
                errors.append(f"Split file lacks observation id column: {sf}; columns={list(sdf.columns)}")
                continue

            unknown = set(sdf[id_col].astype(str)) - obs_ids
            if unknown:
                errors.append(f"Split file has {len(unknown)} unknown ids: {sf}")

    for jf in [
        "data/processed/summary.json",
        "data/metadata/dataset_metadata.json",
        "data/metadata/croissant.json",
        "splits/splits_manifest.json",
        "results/baselines/baseline_metadata.json",
    ]:
        p = ROOT / jf
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                errors.append(f"Invalid JSON: {jf}: {e}")

    br = ROOT / "results/baselines/baseline_results.csv"
    if br.exists():
        bdf = pd.read_csv(br)
        if len(bdf) != 63:
            warnings.append(f"baseline_results.csv has {len(bdf)} rows, expected 63")

    berr = ROOT / "results/baselines/baseline_errors.csv"
    if berr.exists():
        txt = berr.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            errors.append("baseline_errors.csv is not empty")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  -", w)
        print()

    if errors:
        print("ERRORS:")
        for e in errors:
            print("  -", e)
        return 1

    print("Release sanity check passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
