#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "metadata" / "croissant.json"

ANON_URL = "https://anonymous.4open.science/r/PepSol2000-93DC/"
VERSION = "0.1.0-anonymous"

CANDIDATE_FILES = [
    "README.md",
    "requirements-minimal.txt",
    "LICENSE",
    "LICENSE-DATA",

    "data/raw/pep.csv",

    "data/processed/long_peptide_solvent.csv",
    "data/processed/features_for_model.csv",
    "data/processed/peptide_features.csv",
    "data/processed/solvent_summary.csv",
    "data/processed/sequence_label_variability.csv",
    "data/processed/duplicate_conflicts.csv",
    "data/processed/sequence_qc.csv",
    "data/processed/mw_discrepancies.csv",
    "data/processed/unknown_label_values.csv",
    "data/processed/summary.json",

    "data/metadata/dataset_metadata.json",
    "data/metadata/data_dictionary.csv",
    "data/metadata/checksums.sha256",

    "splits/README.md",
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
    "results/baselines/baseline_errors.csv",
    "results/baselines/baseline_table.tex",

    "scripts/build_dataset.py",
    "scripts/make_official_splits.py",
    "scripts/run_reference_baselines.py",
    "scripts/make_croissant.py",
]


FIELD_DESCRIPTIONS = {
    "observation_id": "Stable identifier for a peptide--solvent observation.",
    "source_row_index": "Row index in the raw wide-format input table.",
    "sequence_id": "Identifier for the cleaned peptide sequence.",
    "sequence_raw": "Raw peptide sequence string before cleaning.",
    "sequence_clean": "Cleaned peptide sequence used for modeling and grouping.",
    "solvent_name": "Solvent or formulation condition name.",
    "solvent_id": "Identifier for the solvent or formulation condition.",
    "label_raw": "Raw solubility label value before binary standardization.",
    "y": "Binary solubility label; 1 denotes soluble and 0 denotes insoluble under the specified solvent condition.",
    "difficulty_prediction": "Source-provided or derived difficulty prediction field, if available.",
    "difficulty_score": "Source-provided or derived difficulty score field as text, if available.",
    "difficulty_score_num": "Numeric difficulty score, if parseable.",
    "mw": "Molecular weight field as represented in the source table, if available.",
    "mw_num": "Numeric molecular weight, if parseable.",
    "quantity_delivered": "Quantity delivered field from the source table, if available.",
    "quantity_total_mg": "Numeric total quantity in milligrams, if parseable.",
    "hplc_result": "HPLC result field from the source table, if available.",
    "hplc_percent": "Numeric HPLC percentage, if parseable."
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encoding_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    if suffix == ".sha256":
        return "text/plain"
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".py":
        return "text/x-python"
    if suffix == ".tex":
        return "text/x-tex"
    return "application/octet-stream"


def data_type_for_column(col: str) -> str:
    if col in {"source_row_index", "y"}:
        return "sc:Integer"
    if col.endswith("_num") or col in {"quantity_total_mg", "hplc_percent"}:
        return "sc:Float"
    return "sc:Text"


def file_description(rel: str) -> str:
    if rel == "data/processed/long_peptide_solvent.csv":
        return "Main long-format peptide--solvent solubility table."
    if rel.startswith("splits/"):
        return "Official split file or split manifest for SolPepBench."
    if rel.startswith("results/baselines/"):
        return "Reference baseline result file for SolPepBench."
    if rel.startswith("scripts/"):
        return "Reproducibility script included in the anonymous review artifact."
    if rel.startswith("data/metadata/"):
        return "Dataset metadata, dictionary, checksum, or Croissant file."
    if rel.startswith("data/processed/"):
        return "Processed data table or quality-control output."
    if rel.startswith("data/raw/"):
        return "Raw wide-format input data table."
    return "Repository file included in the anonymous review artifact."


def make_distribution():
    dist = []
    for rel in CANDIDATE_FILES:
        path = ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        dist.append({
            "@type": "cr:FileObject",
            "@id": rel,
            "name": path.name,
            "description": file_description(rel),
            "contentUrl": rel,
            "encodingFormat": encoding_format(path),
            "sha256": sha256(path),
        })
    return dist


def make_long_table_recordset():
    rel = "data/processed/long_peptide_solvent.csv"
    path = ROOT / rel
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        columns = next(reader)

    fields = []
    for col in columns:
        fields.append({
            "@type": "cr:Field",
            "name": col,
            "description": FIELD_DESCRIPTIONS.get(col, f"Column `{col}` in the long peptide--solvent table."),
            "dataType": data_type_for_column(col),
            "source": {
                "fileObject": {"@id": rel},
                "extract": {"column": col}
            }
        })

    return [{
        "@type": "cr:RecordSet",
        "@id": "long_peptide_solvent_records",
        "name": "long_peptide_solvent",
        "description": (
            "Rows of the main SolPepBench table. Each record is a peptide--solvent "
            "observation with a binary solubility label."
        ),
        "field": fields
    }]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "conformsTo": "dct:conformsTo",
            "citeAs": "cr:citeAs",
            "recordSet": "cr:recordSet",
            "field": "cr:field",
            "dataType": {
                "@id": "cr:dataType",
                "@type": "@vocab"
            },
            "source": "cr:source",
            "fileObject": "cr:fileObject",
            "extract": "cr:extract",
            "column": "cr:column",
            "sha256": "cr:sha256"
        },
        "@type": "sc:Dataset",
        "name": "SolPepBench",
        "description": (
            "SolPepBench is a solvent-conditioned peptide solubility dataset and "
            "benchmark definition organized at the peptide--solvent observation level. "
            "The anonymous review artifact includes raw and processed tables, official "
            "split files, reference baseline results, quality-control summaries, "
            "metadata, checksums, and reproducibility scripts."
        ),
        "url": ANON_URL,
        "version": VERSION,
        "dateCreated": "2026-05-06",
        "datePublished": "2026-05-06",
        "dateModified": date.today().isoformat(),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "sdLicense": "https://creativecommons.org/licenses/by/4.0/",
        "creator": [
            {
                "@type": "sc:Organization",
                "name": "Anonymous Authors"
            }
        ],
        "publisher": {
            "@type": "sc:Organization",
            "name": "Anonymous Authors"
        },
        "inLanguage": "en",
        "keywords": [
            "peptide solubility",
            "solvent-conditioned prediction",
            "peptide benchmark",
            "molecular machine learning",
            "dataset",
            "evaluation"
        ],
        "isLiveDataset": False,
        "conformsTo": [
            "http://mlcommons.org/croissant/1.0",
            "http://mlcommons.org/croissant/RAI/1.0"
        ],
        "citeAs": (
            "Anonymous Authors. SolPepBench: A Solvent-Conditioned Dataset and "
            "Benchmark Definition for Peptide Solubility Prediction. Anonymous "
            "NeurIPS 2026 E&D review artifact, 2026."
        ),
        "distribution": make_distribution(),
        "recordSet": make_long_table_recordset(),

        "rai:dataCollection": (
            "The dataset is derived from a raw wide-format peptide table included in "
            "the review artifact. The processing pipeline converts solvent-specific "
            "label fields into long-format peptide--solvent observations and reports "
            "quality-control outputs for unknown labels, duplicates, conflicts, "
            "sequence checks, solvent summaries, and molecular-weight discrepancies."
        ),
        "rai:dataCollectionType": [
            "Secondary Data Analysis",
            "Experiments",
            "Manual Human Curation"
        ],
        "rai:dataCollectionRawData": (
            "The raw wide-format input table is included at data/raw/pep.csv."
        ),
        "rai:dataCollectionMissingData": (
            "The current release does not contain complete formulation metadata such "
            "as pH, ionic strength, peptide concentration, counterion, temperature, "
            "incubation time, mixing procedure, or full analytical protocol for every "
            "record."
        ),
        "rai:dataAnnotationProtocol": (
            "No new crowdsourced annotation was performed for this release. Binary "
            "labels are derived from solvent-specific fields in the raw table and "
            "standardized by scripts/build_dataset.py. Unrecognized label values are "
            "reported in quality-control outputs rather than silently interpreted."
        ),
        "rai:dataAnnotationPlatform": (
            "No external annotation platform was used in the construction of this "
            "anonymous review artifact."
        ),
        "rai:dataAnnotationAnalysis": (
            "The release reports duplicate sequence--solvent groups, conflicting "
            "sequence--solvent groups, unknown label values, sequence quality-control "
            "summaries, molecular-weight discrepancies, solvent-level summaries, and "
            "sequence-level label variability."
        ),
        "rai:annotationsPerItem": (
            "Not applicable; the release does not introduce a new crowdsourced "
            "annotation process."
        ),
        "rai:machineAnnotationTools": [
            "scripts/build_dataset.py",
            "scripts/make_official_splits.py",
            "scripts/run_reference_baselines.py"
        ],
        "rai:dataUseCases": [
            "Dataset analysis of solvent-conditioned peptide solubility labels.",
            "Development and evaluation of solvent-aware peptide solubility predictors.",
            "Benchmarking under pair-stratified, sequence-disjoint, and solvent-held-out split protocols.",
            "Studying when sequence-only label collapse discards condition-dependent variation."
        ],
        "rai:dataLimitations": [
            "Binary labels simplify concentration-dependent solubility behavior.",
            "Solubility outcomes may depend on unreported protocol variables such as pH, ionic strength, concentration, temperature, counterion, and incubation procedure.",
            "The dataset should not be interpreted as a complete physical formulation model.",
            "Performance on SolPepBench is not evidence of clinical, manufacturing, or regulatory readiness.",
            "Modified peptides and noncanonical chemistries may require richer representations than the cleaned sequence string."
        ],
        "rai:dataBiases": [
            "The solvent distribution is limited to seven labeled solvent conditions in the current processed release.",
            "The dataset may reflect source-table selection biases toward peptides and solvents that were measured, synthesized, reported, or retained.",
            "Positive and negative labels are moderately imbalanced, and label distributions differ across solvent conditions."
        ],
        "rai:dataSocialImpact": (
            "Potential benefits include more transparent peptide solubility benchmarking "
            "and reduced unnecessary experimental screening. Potential risks include "
            "overinterpretation of benchmark predictions as definitive formulation "
            "decisions. The release is intended for research use and hypothesis "
            "generation, not clinical, regulatory, or manufacturing decision-making."
        ),
        "rai:personalSensitiveInformation": (
            "The dataset does not contain human-subject data, patient records, personal "
            "identifiers, or clinical outcome data."
        ),
        "rai:dataSensitive": (
            "The dataset does not contain human-subject data, patient records, personal "
            "identifiers, or clinical outcome data."
        ),
        "rai:dataReleaseMaintenance": (
            "The intended maintenance model is conservative. Future releases should "
            "preserve raw inputs where possible, document changes to label mappings "
            "or preprocessing, and version official split files rather than silently "
            "redefining them."
        ),
        "rai:dataMaintenance": (
            "Future releases should preserve raw inputs where possible, document "
            "changes, and version official split files and benchmark results."
        )
    }

    OUT.write_text(json.dumps(croissant, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"distribution_files={len(croissant['distribution'])}")
    print(f"record_sets={len(croissant['recordSet'])}")


if __name__ == "__main__":
    main()
