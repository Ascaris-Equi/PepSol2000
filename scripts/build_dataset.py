# data.py
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import hashlib
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# 你只需要改这里，如果你的文件名不是 pep.csv
# ============================================================

INPUT_FILE = "pep.csv"
OUT_DIR = "data_out"


# ============================================================
# 基础工具
# ============================================================

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_LIST)

AVG_RESIDUE_MASS = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

WATER_MASS = 18.01528

KYTE_DOOLITTLE = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

PKA = {
    "n_term": 9.69,
    "c_term": 2.34,
    "D": 3.86,
    "E": 4.25,
    "C": 8.33,
    "Y": 10.07,
    "H": 6.00,
    "K": 10.53,
    "R": 12.48,
}

GROUPS = {
    "hydrophobic": set("AVILMFWYCP"),
    "strong_hydrophobic": set("AVILMFW"),
    "polar": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "charged": set("KRHDE"),
    "aromatic": set("FWY"),
    "aliphatic": set("AVIL"),
    "small": set("AGSTCP"),
    "tiny": set("AGS"),
    "proline": set("P"),
    "glycine": set("G"),
    "sulfur": set("CM"),
}

SOLVENT_HINTS = [
    "water",
    "ultrapure",
    "dpbs",
    "pbs",
    "dmso",
    "hcooh",
    "formic",
    "saline",
    "acetic",
    "ammonia",
    "buffer",
    "超纯水",
    "生理盐水",
    "甲酸",
    "乙酸",
    "醋酸",
    "氨水",
]


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def cjk_count(s):
    return sum(1 for ch in str(s) if "\u4e00" <= ch <= "\u9fff")


def weird_count(s):
    weird = "ÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß£¨©¼½º¬µý²âÏµÊýÁ¿Æ¬¶Î�"
    return sum(1 for ch in str(s) if ch in weird)


def fix_mojibake(x):
    """
    修复类似：
    ÄÑ¶ÈÔ¤²â -> 难度预测
    ÄÑ¶ÈÏµÊý -> 难度系数
    ¼ì²â½á¹û -> 检测结果
    £¨º¬ÄÑ¶ÈÆ¬¶Î£© -> （含难度片段）
    """
    if not isinstance(x, str):
        return x

    s = x

    try:
        cand = s.encode("latin1").decode("gb18030")
    except Exception:
        return s

    old_score = cjk_count(s) - weird_count(s)
    new_score = cjk_count(cand) - weird_count(cand)

    if new_score > old_score:
        return cand
    return s


def clean_text(x):
    if not isinstance(x, str):
        return x
    s = fix_mojibake(x)
    s = s.replace("\ufeff", "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_col_name(c):
    """
    把乱七八糟的列名统一成简单英文。
    """
    c0 = str(clean_text(c)).strip()
    low = c0.lower()

    if low.startswith("sequence") or "肽序列" in c0 or c0 in ["序列", "多肽序列"]:
        return "sequence"

    if "难度预测" in c0 or "difficulty prediction" in low:
        return "difficulty_prediction"

    if "难度系数" in c0 or "difficulty score" in low or "difficulty coefficient" in low:
        return "difficulty_score"

    if low in ["mw", "m.w.", "molecular weight", "molecular_weight", "molecular mass"]:
        return "mw"

    if "quantity" in low and "deliver" in low:
        return "quantity_delivered"

    if "hplc" in low:
        return "hplc_result"

    return c0


def dedupe_columns(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def stable_id(prefix, text):
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def slugify(x):
    s = str(clean_text(x)).lower()
    s = s.replace("%", "pct")
    s = s.replace("+", "plus")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    if s[0].isdigit():
        s = "s_" + s
    return s


def parse_number(x):
    if pd.isna(x):
        return np.nan
    s = str(clean_text(x))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan


def parse_quantity_mg(x):
    """
    解析：
    7*2.0mg -> 14.0
    4*3.0mg+2.0mg -> 14.0
    3 x 5 mg -> 15.0
    """
    if pd.isna(x):
        return np.nan

    s = str(clean_text(x)).lower()
    s = s.replace("×", "*").replace("x", "*")
    s = s.replace(" ", "")

    total = 0.0
    found = False

    pattern = re.compile(
        r"(?:(?P<n>\d+(?:\.\d+)?)\*)?"
        r"(?P<amount>\d+(?:\.\d+)?)"
        r"(?P<unit>mg|ug|µg|μg|g)?"
    )

    for m in pattern.finditer(s):
        n = float(m.group("n")) if m.group("n") is not None else 1.0
        amount = float(m.group("amount"))
        unit = m.group("unit") or "mg"

        if unit == "g":
            amount_mg = amount * 1000.0
        elif unit in ["ug", "µg", "μg"]:
            amount_mg = amount / 1000.0
        else:
            amount_mg = amount

        total += n * amount_mg
        found = True

    return total if found else np.nan


def normalize_sequence(x):
    if pd.isna(x):
        return ""
    s = str(clean_text(x)).upper()
    s = re.sub(r"\s+", "", s)
    return s


def normalize_label(x):
    """
    Soluble -> 1
    Insoluble -> 0
    空值 / 未识别 -> NaN
    """
    if pd.isna(x):
        return np.nan

    s = str(clean_text(x)).strip()
    if not s:
        return np.nan

    low = s.lower()
    compact = re.sub(r"[\s_\-]+", "", low)

    if compact in ["na", "nan", "none", "null", "n/a", "nt", "nottested", "/", "-"]:
        return np.nan

    # 注意先判断 insoluble，因为 insoluble 里面包含 soluble
    if "insoluble" in compact or "not soluble" in low or "不溶" in s or "难溶" in s:
        return 0

    if "soluble" in compact or "可溶" in s:
        return 1

    return np.nan


def read_csv_auto(path):
    """
    自动尝试编码和分隔符。
    """
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "latin1"]
    seps = [None, ",", "\t", ";"]

    best = None
    best_info = None
    best_score = -10**18

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception:
                continue

            if df.shape[1] <= 1:
                score = df.shape[1]
            else:
                fixed_cols = [normalize_col_name(c) for c in df.columns]
                has_seq = int("sequence" in fixed_cols)

                label_cols = 0
                for c in df.columns:
                    vals = df[c].map(normalize_label)
                    if vals.notna().sum() > 0:
                        label_cols += 1

                weird = sum(weird_count(c) for c in fixed_cols)

                score = df.shape[1] + has_seq * 100 + label_cols * 10 - weird * 0.1

            if score > best_score:
                best_score = score
                best = df
                best_info = {
                    "encoding": enc,
                    "sep": "auto" if sep is None else sep,
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "score": float(score),
                }

    if best is None:
        raise RuntimeError(f"读取失败：{path}")

    return best, best_info


# ============================================================
# peptide feature
# ============================================================

def is_canonical(seq):
    return bool(seq) and all(a in AA_SET for a in seq)


def noncanonical_symbols(seq):
    return "".join(sorted(set(a for a in seq if a not in AA_SET)))


def seq_entropy(seq):
    if not seq:
        return np.nan
    L = len(seq)
    cnt = Counter(seq)
    e = 0.0
    for n in cnt.values():
        p = n / L
        e -= p * math.log(p, 2)
    return e


def basic_fraction(pH, pKa):
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def acidic_fraction(pH, pKa):
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def net_charge(seq, pH=7.0):
    if not is_canonical(seq):
        return np.nan

    cnt = Counter(seq)

    pos = 0.0
    neg = 0.0

    pos += basic_fraction(pH, PKA["n_term"])
    neg += acidic_fraction(pH, PKA["c_term"])

    pos += cnt["K"] * basic_fraction(pH, PKA["K"])
    pos += cnt["R"] * basic_fraction(pH, PKA["R"])
    pos += cnt["H"] * basic_fraction(pH, PKA["H"])

    neg += cnt["D"] * acidic_fraction(pH, PKA["D"])
    neg += cnt["E"] * acidic_fraction(pH, PKA["E"])
    neg += cnt["C"] * acidic_fraction(pH, PKA["C"])
    neg += cnt["Y"] * acidic_fraction(pH, PKA["Y"])

    return pos - neg


def estimate_pI(seq):
    if not is_canonical(seq):
        return np.nan

    lo, hi = 0.0, 14.0
    for _ in range(70):
        mid = (lo + hi) / 2
        q = net_charge(seq, mid)
        if q > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def max_run(seq, group):
    best = 0
    cur = 0
    for a in seq:
        if a in group:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def peptide_features(seq):
    seq = normalize_sequence(seq)
    L = len(seq)
    cnt = Counter(seq)
    ok = is_canonical(seq)

    row = {
        "sequence_clean": seq,
        "length_raw": L,
        "is_canonical_aa": bool(ok),
        "noncanonical_symbols": noncanonical_symbols(seq),
    }

    for aa in AA_LIST:
        row[f"count_{aa}"] = cnt.get(aa, 0)
        row[f"frac_{aa}"] = cnt.get(aa, 0) / L if ok and L > 0 else np.nan

    if not ok:
        row.update({
            "length": np.nan,
            "mw_calc_avg": np.nan,
            "gravy": np.nan,
            "hydropathy_std": np.nan,
            "sequence_entropy": np.nan,
            "sequence_entropy_norm20": np.nan,
            "net_charge_pH7": np.nan,
            "abs_net_charge_pH7": np.nan,
            "charge_per_residue_pH7": np.nan,
            "pI_calc": np.nan,
            "fcr": np.nan,
            "ncpr": np.nan,
        })
        for g in GROUPS:
            row[f"count_{g}"] = np.nan
            row[f"frac_{g}"] = np.nan
            row[f"max_run_{g}"] = np.nan
        return row

    row["length"] = L
    row["mw_calc_avg"] = sum(AVG_RESIDUE_MASS[a] for a in seq) + WATER_MASS

    hydros = np.array([KYTE_DOOLITTLE[a] for a in seq], dtype=float)
    row["gravy"] = float(hydros.mean())
    row["hydropathy_std"] = float(hydros.std())

    e = seq_entropy(seq)
    row["sequence_entropy"] = e
    row["sequence_entropy_norm20"] = e / math.log(20, 2)

    q7 = net_charge(seq, 7.0)
    row["net_charge_pH7"] = q7
    row["abs_net_charge_pH7"] = abs(q7)
    row["charge_per_residue_pH7"] = q7 / L
    row["pI_calc"] = estimate_pI(seq)

    n_pos = sum(cnt.get(a, 0) for a in GROUPS["positive"])
    n_neg = sum(cnt.get(a, 0) for a in GROUPS["negative"])
    n_charged = n_pos + n_neg

    row["fcr"] = n_charged / L
    row["ncpr"] = (n_pos - n_neg) / L

    for g, aset in GROUPS.items():
        n = sum(cnt.get(a, 0) for a in aset)
        row[f"count_{g}"] = n
        row[f"frac_{g}"] = n / L
        row[f"max_run_{g}"] = max_run(seq, aset)

    return row


# ============================================================
# 统计函数
# ============================================================

def numeric_summary(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q1": float(x.quantile(0.25)),
        "median": float(x.median()),
        "q3": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def detect_solvent_columns(df):
    solvents = []

    meta_cols = {
        "sequence",
        "difficulty_prediction",
        "difficulty_score",
        "mw",
        "quantity_delivered",
        "hplc_result",
    }

    for c in df.columns:
        if c in meta_cols:
            continue

        low = str(c).lower()
        name_hint = any(h in low for h in SOLVENT_HINTS)

        vals = df[c].map(normalize_label)
        n_label = int(vals.notna().sum())
        n_nonempty = int(df[c].replace("", np.nan).notna().sum())

        label_hint = False
        if n_nonempty > 0 and n_label > 0:
            label_hint = n_label / n_nonempty >= 0.5

        if name_hint or label_hint:
            solvents.append(c)

    return solvents


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if pd.isna(obj):
        return None
    return obj


# ============================================================
# 主程序
# ============================================================

def main():
    mkdir(OUT_DIR)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"找不到 {INPUT_FILE}。请把 pep.csv 和 data.py 放在同一个文件夹，或者修改 INPUT_FILE。"
        )

    print("Reading:", INPUT_FILE)

    raw, read_info = read_csv_auto(INPUT_FILE)

    # 清理列名和单元格
    df = raw.copy()
    df.columns = dedupe_columns([normalize_col_name(c) for c in df.columns])

    for c in df.columns:
        df[c] = df[c].map(clean_text)

    df = df.replace({
        "": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "na": np.nan,
        "n/a": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NULL": np.nan,
        "null": np.nan,
    })

    if "sequence" not in df.columns:
        raise RuntimeError(
            "没有识别到 sequence 列。请检查第一列是不是 Sequence 或 Sequence（含难度片段）。"
        )

    # 增加基础字段
    df.insert(0, "source_row_index", range(len(df)))
    df["sequence_raw"] = df["sequence"]
    df["sequence_clean"] = df["sequence"].map(normalize_sequence)
    df["sequence_id"] = df["sequence_clean"].map(lambda x: stable_id("pep", x))

    if "difficulty_prediction" in df.columns:
        df["difficulty_prediction"] = df["difficulty_prediction"].map(
            lambda x: str(x).lower().strip() if pd.notna(x) else np.nan
        )

    if "difficulty_score" in df.columns:
        df["difficulty_score_num"] = df["difficulty_score"].map(parse_number)

    if "mw" in df.columns:
        df["mw_num"] = df["mw"].map(parse_number)

    if "quantity_delivered" in df.columns:
        df["quantity_total_mg"] = df["quantity_delivered"].map(parse_quantity_mg)

    if "hplc_result" in df.columns:
        df["hplc_percent"] = df["hplc_result"].map(parse_number)

    solvent_cols = detect_solvent_columns(df)

    if len(solvent_cols) == 0:
        raise RuntimeError("没有识别到 solvent label 列。请检查 Soluble/Insoluble 是否存在。")

    # ------------------------------------------------------------
    # 1. 保存清理后的宽表
    # ------------------------------------------------------------
    clean_wide_path = os.path.join(OUT_DIR, "clean_wide.csv")
    df.to_csv(clean_wide_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. 宽表转长表：一个非空 solvent cell = 一个 observation
    # ------------------------------------------------------------
    long_rows = []

    for _, r in df.iterrows():
        for solv in solvent_cols:
            raw_label = r.get(solv, np.nan)
            y = normalize_label(raw_label)

            if pd.isna(y):
                continue

            solvent_id = slugify(solv)

            obs_key = f"{r['sequence_id']}|{solvent_id}|{r['source_row_index']}"
            obs_id = stable_id("obs", obs_key)

            row = {
                "observation_id": obs_id,
                "source_row_index": r["source_row_index"],
                "sequence_id": r["sequence_id"],
                "sequence_raw": r["sequence_raw"],
                "sequence_clean": r["sequence_clean"],
                "solvent_name": solv,
                "solvent_id": solvent_id,
                "label_raw": raw_label,
                "y": int(y),
            }

            for col in [
                "difficulty_prediction",
                "difficulty_score",
                "difficulty_score_num",
                "mw",
                "mw_num",
                "quantity_delivered",
                "quantity_total_mg",
                "hplc_result",
                "hplc_percent",
            ]:
                if col in df.columns:
                    row[col] = r.get(col, np.nan)

            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(OUT_DIR, "long_peptide_solvent.csv")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 3. peptide features
    # ------------------------------------------------------------
    unique_pep = df[["sequence_id", "sequence_raw", "sequence_clean"]].drop_duplicates("sequence_id")

    feat_rows = []
    for _, r in unique_pep.iterrows():
        f = peptide_features(r["sequence_clean"])
        f["sequence_id"] = r["sequence_id"]
        f["sequence_raw"] = r["sequence_raw"]
        feat_rows.append(f)

    pep_feat = pd.DataFrame(feat_rows)

    id_cols = ["sequence_id", "sequence_raw", "sequence_clean"]
    other_cols = [c for c in pep_feat.columns if c not in id_cols]
    pep_feat = pep_feat[id_cols + other_cols]

    peptide_feature_path = os.path.join(OUT_DIR, "peptide_features.csv")
    pep_feat.to_csv(peptide_feature_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 4. long table 合并 peptide features
    # ------------------------------------------------------------
    feature_df = long_df.merge(
        pep_feat.drop(columns=["sequence_raw"], errors="ignore"),
        on=["sequence_id", "sequence_clean"],
        how="left",
    )

    feature_path = os.path.join(OUT_DIR, "features_for_model.csv")
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 5. solvent label summary
    # ------------------------------------------------------------
    solvent_summary_rows = []

    for solv in solvent_cols:
        sub = long_df[long_df["solvent_name"] == solv]
        n_total = len(df)
        n_labeled = len(sub)
        n_soluble = int((sub["y"] == 1).sum()) if n_labeled else 0
        n_insoluble = int((sub["y"] == 0).sum()) if n_labeled else 0

        solvent_summary_rows.append({
            "solvent_name": solv,
            "solvent_id": slugify(solv),
            "n_labeled_records": n_labeled,
            "n_soluble": n_soluble,
            "n_insoluble": n_insoluble,
            "positive_rate": n_soluble / n_labeled if n_labeled else np.nan,
            "n_missing_cells": n_total - n_labeled,
            "n_unique_peptides": sub["sequence_id"].nunique() if n_labeled else 0,
        })

    solvent_summary = pd.DataFrame(solvent_summary_rows)
    solvent_summary_path = os.path.join(OUT_DIR, "solvent_summary.csv")
    solvent_summary.to_csv(solvent_summary_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 6. sequence variability
    # ------------------------------------------------------------
    seq_var_rows = []

    if len(long_df) > 0:
        for (sid, seq), sub in long_df.groupby(["sequence_id", "sequence_clean"], dropna=False):
            n_solvents = sub["solvent_id"].nunique()
            n_records = len(sub)
            n_soluble = int((sub["y"] == 1).sum())
            n_insoluble = int((sub["y"] == 0).sum())
            seq_var_rows.append({
                "sequence_id": sid,
                "sequence_clean": seq,
                "n_records": n_records,
                "n_solvents_tested": n_solvents,
                "n_soluble": n_soluble,
                "n_insoluble": n_insoluble,
                "has_discordant_solvent_labels": int(n_soluble > 0 and n_insoluble > 0),
            })

    seq_var = pd.DataFrame(seq_var_rows)
    seq_var_path = os.path.join(OUT_DIR, "sequence_label_variability.csv")
    seq_var.to_csv(seq_var_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 7. duplicate conflict QC
    # ------------------------------------------------------------
    conflict_rows = []

    if len(long_df) > 0:
        for (seq, solv), sub in long_df.groupby(["sequence_clean", "solvent_name"], dropna=False):
            labels = sorted(sub["y"].dropna().astype(int).unique().tolist())
            conflict_rows.append({
                "sequence_clean": seq,
                "solvent_name": solv,
                "n_records": len(sub),
                "n_unique_labels": len(labels),
                "labels_observed": ";".join(map(str, labels)),
                "has_duplicate": int(len(sub) > 1),
                "has_conflict": int(len(labels) > 1),
                "observation_ids": ";".join(sub["observation_id"].astype(str).tolist()),
            })

    duplicate_conflicts = pd.DataFrame(conflict_rows)

    if len(duplicate_conflicts) > 0:
        duplicate_conflicts = duplicate_conflicts.sort_values(
            ["has_conflict", "has_duplicate", "n_records"],
            ascending=False,
        )

    duplicate_conflicts_path = os.path.join(OUT_DIR, "duplicate_conflicts.csv")
    duplicate_conflicts.to_csv(duplicate_conflicts_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 8. sequence QC
    # ------------------------------------------------------------
    seq_qc = pep_feat[[
        "sequence_id",
        "sequence_raw",
        "sequence_clean",
        "length_raw",
        "length",
        "is_canonical_aa",
        "noncanonical_symbols",
    ]].copy()

    seq_qc["missing_sequence"] = seq_qc["sequence_clean"].map(
        lambda x: int(not isinstance(x, str) or len(x) == 0)
    )

    seq_qc_path = os.path.join(OUT_DIR, "sequence_qc.csv")
    seq_qc.to_csv(seq_qc_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 9. Mw discrepancy
    # ------------------------------------------------------------
    if "mw_num" in df.columns:
        mw_check = df[["sequence_id", "sequence_clean", "mw", "mw_num"]].drop_duplicates("sequence_id")
        mw_check = mw_check.merge(
            pep_feat[["sequence_id", "mw_calc_avg"]],
            on="sequence_id",
            how="left",
        )
        mw_check["mw_abs_diff"] = (mw_check["mw_num"] - mw_check["mw_calc_avg"]).abs()
        mw_check = mw_check.sort_values("mw_abs_diff", ascending=False)
    else:
        mw_check = pd.DataFrame()

    mw_check_path = os.path.join(OUT_DIR, "mw_discrepancies.csv")
    mw_check.to_csv(mw_check_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 10. unknown label values
    # ------------------------------------------------------------
    unknown_rows = []
    for solv in solvent_cols:
        for idx, val in df[solv].items():
            if pd.isna(val):
                continue
            y = normalize_label(val)
            if pd.isna(y):
                unknown_rows.append({
                    "row_index": idx,
                    "solvent_column": solv,
                    "raw_value": val,
                })

    unknown_labels = pd.DataFrame(unknown_rows)
    unknown_labels_path = os.path.join(OUT_DIR, "unknown_label_values.csv")
    unknown_labels.to_csv(unknown_labels_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 11. solvent metadata template
    # ------------------------------------------------------------
    solvent_meta_rows = []

    for solv in solvent_cols:
        low = str(solv).lower()
        sid = slugify(solv)

        solvent_class = "unknown"
        is_aqueous = 0
        is_buffer = 0
        is_salt = 0
        is_organic = 0
        is_acid = 0
        is_base = 0

        if "water" in low or "ultrapure" in low or "超纯水" in low:
            solvent_class = "water"
            is_aqueous = 1

        if "dpbs" in low:
            solvent_class = "dpbs"
            is_aqueous = 1
            is_buffer = 1
            is_salt = 1

        elif "pbs" in low:
            solvent_class = "pbs"
            is_aqueous = 1
            is_buffer = 1
            is_salt = 1

        if "saline" in low or "生理盐水" in low:
            solvent_class = "saline"
            is_aqueous = 1
            is_salt = 1

        if "dmso" in low:
            solvent_class = "dmso"
            is_organic = 1

        if "hcooh" in low or "formic" in low or "甲酸" in low:
            solvent_class = "formic_acid"
            is_organic = 1
            is_acid = 1

        if "acetic" in low or "乙酸" in low or "醋酸" in low:
            solvent_class = "acetic_acid"
            is_aqueous = 1
            is_acid = 1

        if "ammonia" in low or "氨水" in low:
            solvent_class = "ammonia_water"
            is_aqueous = 1
            is_base = 1

        solvent_meta_rows.append({
            "solvent_name": solv,
            "solvent_id": sid,
            "solvent_class": solvent_class,
            "is_aqueous": is_aqueous,
            "is_buffer": is_buffer,
            "is_salt_containing": is_salt,
            "is_organic": is_organic,
            "is_acidic_condition": is_acid,
            "is_basic_condition": is_base,
            "pH": "",
            "ionic_strength_M": "",
            "dielectric_constant": "",
            "kamlet_taft_alpha": "",
            "kamlet_taft_beta": "",
            "kamlet_taft_pi_star": "",
            "hansen_delta_d": "",
            "hansen_delta_p": "",
            "hansen_delta_h": "",
            "notes": "manual fill if available",
        })

    solvent_meta = pd.DataFrame(solvent_meta_rows)
    solvent_meta_path = os.path.join(OUT_DIR, "solvent_metadata_template.csv")
    solvent_meta.to_csv(solvent_meta_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 12. 总 summary
    # ------------------------------------------------------------
    n_wide = int(len(df))
    n_long = int(len(long_df))
    n_unique_pep_wide = int(df["sequence_id"].nunique())
    n_unique_pep_long = int(long_df["sequence_id"].nunique()) if n_long else 0
    n_solvents = int(len(solvent_cols))
    n_solvents_labeled = int(long_df["solvent_id"].nunique()) if n_long else 0

    n_soluble = int((long_df["y"] == 1).sum()) if n_long else 0
    n_insoluble = int((long_df["y"] == 0).sum()) if n_long else 0
    positive_rate = float(long_df["y"].mean()) if n_long else None

    canonical_count = int((pep_feat["is_canonical_aa"] == True).sum())
    noncanonical_count = int((pep_feat["is_canonical_aa"] != True).sum())

    if len(seq_var) > 0:
        multi_solvent_seq = int((seq_var["n_solvents_tested"] >= 2).sum())
        discordant_seq = int((seq_var["has_discordant_solvent_labels"] == 1).sum())
    else:
        multi_solvent_seq = 0
        discordant_seq = 0

    if len(duplicate_conflicts) > 0:
        duplicate_groups = int((duplicate_conflicts["has_duplicate"] == 1).sum())
        conflict_groups = int((duplicate_conflicts["has_conflict"] == 1).sum())
    else:
        duplicate_groups = 0
        conflict_groups = 0

    difficulty_counts = {}
    if "difficulty_prediction" in df.columns:
        difficulty_counts = df["difficulty_prediction"].fillna("missing").value_counts().to_dict()

    summary = {
        "read_info": read_info,
        "wide_peptide_rows": n_wide,
        "labeled_peptide_solvent_records": n_long,
        "unique_peptides_wide": n_unique_pep_wide,
        "unique_peptides_long": n_unique_pep_long,
        "detected_solvent_columns": solvent_cols,
        "n_detected_solvent_columns": n_solvents,
        "n_solvents_with_labels": n_solvents_labeled,
        "label_counts": {
            "soluble": n_soluble,
            "insoluble": n_insoluble,
            "positive_rate": positive_rate,
        },
        "canonical_sequence_count": canonical_count,
        "noncanonical_or_missing_sequence_count": noncanonical_count,
        "peptide_length_stats": numeric_summary(pep_feat["length"]),
        "mw_input_stats": numeric_summary(df["mw_num"]) if "mw_num" in df.columns else None,
        "quantity_total_mg_stats": numeric_summary(df["quantity_total_mg"]) if "quantity_total_mg" in df.columns else None,
        "hplc_percent_stats": numeric_summary(df["hplc_percent"]) if "hplc_percent" in df.columns else None,
        "difficulty_prediction_counts": difficulty_counts,
        "sequences_tested_in_multiple_solvents": multi_solvent_seq,
        "sequences_with_discordant_solvent_labels": discordant_seq,
        "duplicate_sequence_solvent_groups": duplicate_groups,
        "conflicting_sequence_solvent_groups": conflict_groups,
        "unknown_label_values_count": int(len(unknown_labels)),
        "output_files": {
            "clean_wide": clean_wide_path,
            "long_peptide_solvent": long_path,
            "peptide_features": peptide_feature_path,
            "features_for_model": feature_path,
            "solvent_summary": solvent_summary_path,
            "sequence_label_variability": seq_var_path,
            "duplicate_conflicts": duplicate_conflicts_path,
            "sequence_qc": seq_qc_path,
            "mw_discrepancies": mw_check_path,
            "unknown_label_values": unknown_labels_path,
            "solvent_metadata_template": solvent_meta_path,
        },
    }

    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # 13. 给我看的 paper_numbers.txt
    # ------------------------------------------------------------
    paper_path = os.path.join(OUT_DIR, "paper_numbers.txt")

    with open(paper_path, "w", encoding="utf-8") as f:
        f.write("SolPepBench / pep.csv data audit\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. File reading\n")
        f.write("-" * 60 + "\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Read info: {read_info}\n\n")

        f.write("2. Core dataset size\n")
        f.write("-" * 60 + "\n")
        f.write(f"Wide peptide rows: {n_wide}\n")
        f.write(f"Labeled peptide-solvent records: {n_long}\n")
        f.write(f"Unique peptides in wide table: {n_unique_pep_wide}\n")
        f.write(f"Unique peptides in long table: {n_unique_pep_long}\n")
        f.write(f"Detected solvent columns: {n_solvents}\n")
        f.write(f"Solvent columns with labels: {n_solvents_labeled}\n\n")

        f.write("3. Label balance\n")
        f.write("-" * 60 + "\n")
        f.write(f"Soluble records: {n_soluble}\n")
        f.write(f"Insoluble records: {n_insoluble}\n")
        f.write(f"Positive rate: {positive_rate}\n\n")

        f.write("4. Detected solvent columns\n")
        f.write("-" * 60 + "\n")
        for s in solvent_cols:
            f.write(f"- {s}\n")
        f.write("\n")

        f.write("5. Solvent summary\n")
        f.write("-" * 60 + "\n")
        if len(solvent_summary) > 0:
            f.write(solvent_summary.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No solvent summary.\n\n")

        f.write("6. Peptide length statistics\n")
        f.write("-" * 60 + "\n")
        length_stats = numeric_summary(pep_feat["length"])
        for k, v in length_stats.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("7. Sequence QC\n")
        f.write("-" * 60 + "\n")
        f.write(f"Canonical sequences: {canonical_count}\n")
        f.write(f"Noncanonical or missing sequences: {noncanonical_count}\n")

        noncanonical_df = seq_qc[seq_qc["is_canonical_aa"] != True]
        if len(noncanonical_df) > 0:
            f.write("\nNoncanonical examples:\n")
            f.write(noncanonical_df.head(30).to_string(index=False))
            f.write("\n")
        f.write("\n")

        f.write("8. Solvent-conditioned label variability\n")
        f.write("-" * 60 + "\n")
        f.write(f"Sequences tested in multiple solvents: {multi_solvent_seq}\n")
        f.write(f"Sequences with discordant solvent labels: {discordant_seq}\n\n")

        f.write("9. Duplicate / conflict QC\n")
        f.write("-" * 60 + "\n")
        f.write(f"Duplicate sequence-solvent groups: {duplicate_groups}\n")
        f.write(f"Conflicting sequence-solvent groups: {conflict_groups}\n")

        if conflict_groups > 0:
            f.write("\nTop conflict rows:\n")
            f.write(
                duplicate_conflicts[duplicate_conflicts["has_conflict"] == 1]
                .head(30)
                .to_string(index=False)
            )
            f.write("\n")
        f.write("\n")

        f.write("10. Difficulty / Mw / quantity / HPLC summaries\n")
        f.write("-" * 60 + "\n")

        if difficulty_counts:
            f.write("Difficulty prediction counts:\n")
            for k, v in difficulty_counts.items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("Difficulty prediction counts: not detected\n")

        f.write("\nMw input stats:\n")
        mw_stats = numeric_summary(df["mw_num"]) if "mw_num" in df.columns else None
        f.write(str(mw_stats) + "\n")

        f.write("\nQuantity total mg stats:\n")
        quantity_stats = numeric_summary(df["quantity_total_mg"]) if "quantity_total_mg" in df.columns else None
        f.write(str(quantity_stats) + "\n")

        f.write("\nHPLC percent stats:\n")
        hplc_stats = numeric_summary(df["hplc_percent"]) if "hplc_percent" in df.columns else None
        f.write(str(hplc_stats) + "\n\n")

        f.write("11. Unknown label values\n")
        f.write("-" * 60 + "\n")
        f.write(f"Unknown label values count: {len(unknown_labels)}\n")
        if len(unknown_labels) > 0:
            f.write(unknown_labels.head(30).to_string(index=False))
            f.write("\n")
        f.write("\n")

        f.write("12. Output files\n")
        f.write("-" * 60 + "\n")
        for k, v in summary["output_files"].items():
            f.write(f"{k}: {v}\n")
        f.write(f"summary_json: {summary_path}\n\n")

        f.write("13. Draft paper sentence\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"The dataset contains {n_long} labeled peptide-solvent observations "
            f"derived from {n_unique_pep_long} unique peptide sequences and "
            f"{n_solvents_labeled} solvent/formulation conditions. "
            f"It includes {n_soluble} soluble and {n_insoluble} insoluble observations. "
            f"{multi_solvent_seq} peptide sequences were tested in multiple solvent conditions, "
            f"and {discordant_seq} sequences showed discordant solubility labels across solvents.\n"
        )

    # ------------------------------------------------------------
    # 14. 终端输出
    # ------------------------------------------------------------
    print("\nDone.")
    print("=" * 60)
    print(f"Output folder: {OUT_DIR}")
    print(f"Most important file: {paper_path}")
    print("=" * 60)
    print(f"Wide peptide rows: {n_wide}")
    print(f"Labeled peptide-solvent records: {n_long}")
    print(f"Unique peptides in long table: {n_unique_pep_long}")
    print(f"Solvents with labels: {n_solvents_labeled}")
    print(f"Soluble: {n_soluble}")
    print(f"Insoluble: {n_insoluble}")
    print(f"Positive rate: {positive_rate}")
    print(f"Sequences tested in multiple solvents: {multi_solvent_seq}")
    print(f"Discordant solvent-label sequences: {discordant_seq}")
    print(f"Duplicate sequence-solvent groups: {duplicate_groups}")
    print(f"Conflicting sequence-solvent groups: {conflict_groups}")
    print("=" * 60)
    print("Please send me this file:")
    print(paper_path)


if __name__ == "__main__":
    main()