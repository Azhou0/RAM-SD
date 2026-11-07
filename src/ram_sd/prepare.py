"""Format conversion only; never creates or guesses an experimental split."""

import csv
from pathlib import Path

from .io import read_json


def write_rows(path, records):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    if not records:
        raise ValueError("No records selected")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [name for name in ("row_id", "Text", "Label", "Context") if any(name in row for row in records)]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def convert_csv(source, destination, text_column, label_column, context_column=None,
                id_column=None, positive="1", negative="0", delimiter=","):
    if positive == negative:
        raise ValueError("Positive and negative labels must differ")
    records = []
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = [text_column, label_column, *([context_column] if context_column else []), *([id_column] if id_column else [])]
        if any(c not in (reader.fieldnames or []) for c in columns):
            raise ValueError(f"Input columns do not match {columns}")
        for index, row in enumerate(reader):
            raw_label = (row.get(label_column) or "").strip()
            text = row.get(text_column)
            if None in row or not isinstance(text, str) or not text.strip() or raw_label not in (positive, negative):
                raise ValueError(f"Invalid record {index + 1}; conversion never drops records")
            record = {"Text": text, "Label": int(raw_label == positive)}
            if context_column:
                record["Context"] = row[context_column]
            if id_column:
                record["row_id"] = row[id_column]
            records.append(record)
    write_rows(destination, records)


def convert_mustard(source, ids_path, destination):
    raw = read_json(source)
    ids = read_json(ids_path)
    if not isinstance(raw, dict) or not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
        raise ValueError("Expected MUStARD JSON object and an explicit JSON array of sample-ID strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in the requested split")
    records = []
    for key in ids:
        item = raw[key]
        context = item.get("context")
        if (not isinstance(context, list) or not context or any(not isinstance(c, str) for c in context)
                or type(item.get("sarcasm")) is not bool or not isinstance(item.get("utterance"), str)
                or not item["utterance"].strip()):
            raise ValueError(f"Invalid MUStARD record: {key}")
        # Preserve dialogue ordering. Speaker-name augmentation is not silently introduced.
        records.append({"row_id": key, "Text": item["utterance"], "Label": int(item["sarcasm"]),
                        "Context": "\n".join(context)})
    write_rows(destination, records)
