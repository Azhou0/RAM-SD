"""Strict CSV input, train/test provenance, and leakage checks; no implicit splitting."""

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DATASETS, PAPER_COUNTS
from .io import digest, file_hash


@dataclass(frozen=True)
class Query:
    row_id: str
    text: str
    context: str = ""


@dataclass(frozen=True)
class Row:
    query: Query
    label: int


def normalize(text):
    return " ".join(text.split()).casefold()


def read_rows(path, dataset, split):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {"Text", "Label"}.issubset(fields):
            raise ValueError(f"{path}: expected Text,Label columns")
        if len(set(fields)) != len(fields):
            raise ValueError(f"{path}: duplicate CSV column names")
        if dataset == "mustard" and "Context" not in fields:
            raise ValueError(f"{path}: MUSTARD requires Context")
        rows = []
        seen = set()
        for index, raw in enumerate(reader):
            if None in raw or any(raw.get(k) is None for k in fields):
                raise ValueError(f"{path}: malformed record {index + 1}")
            text = raw["Text"]
            label = raw["Label"].strip()
            context = raw.get("Context", "")
            if not text.strip() or label not in ("0", "1"):
                raise ValueError(f"{path}: invalid text/label at record {index + 1}; rows are never silently dropped")
            if dataset == "mustard" and not context.strip():
                raise ValueError(f"{path}: empty MUSTARD context at record {index + 1}")
            row_id = raw.get("row_id") or f"{dataset}:{split}:{index:06d}:{digest([text, context])[:12]}"
            if row_id in seen:
                raise ValueError(f"{path}: duplicate row_id {row_id}")
            seen.add(row_id)
            rows.append(Row(Query(row_id, text, context), int(label)))
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def audit(dataset, train, test, paths, strict_counts=False):
    expected_train, expected_test = PAPER_COUNTS[dataset]
    warnings = []
    if len(train) != expected_train:
        warnings.append(f"train_count={len(train)}; paper={expected_train}")
    if len(test) != expected_test:
        warnings.append(f"test_count={len(test)}; paper={expected_test}")
    # Text-only datasets retrieve by text. MUSTARD has repeated utterances with distinct contexts.
    def key(row):
        return (normalize(row.query.text), normalize(row.query.context) if dataset == "mustard" else "")
    overlap = set(map(key, train)) & set(map(key, test))
    if overlap:
        raise ValueError(f"{dataset}: {len(overlap)} normalized train/test overlaps; refusing evaluation")
    if strict_counts and warnings:
        raise ValueError("Paper count mismatch: " + "; ".join(warnings))
    return {
        "dataset": dataset,
        "split_identity": "unverified_against_original_paper",
        "warnings": warnings,
        "normalized_train_test_overlap": len(overlap),
        "train_records_sha256": digest([asdict(r) for r in train]),
        "test_records_sha256": digest([asdict(r) for r in test]),
        "train": {"rows": len(train), "labels": dict(Counter(r.label for r in train)), "unique_normalized": len(set(map(key, train)))},
        "test": {"rows": len(test), "labels": dict(Counter(r.label for r in test)), "unique_normalized": len(set(map(key, test)))},
        "files": {name: {"name": Path(p).name, "sha256": file_hash(p)} for name, p in paths.items()},
    }


def load_data(root, dataset, strict_counts=False):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    directory = Path(root) / dataset
    paths = {"test": directory / "test.csv"}
    train_path = directory / "train.csv"
    train = []
    if train_path.exists():
        train = read_rows(train_path, dataset, "train")
        paths["train"] = train_path
    elif dataset != "mustard":
        raise ValueError(f"Missing {train_path}; retrieval requires a training split")
    test = read_rows(paths["test"], dataset, "test")
    report = audit(dataset, train, test, paths, strict_counts)
    return train, test, report
