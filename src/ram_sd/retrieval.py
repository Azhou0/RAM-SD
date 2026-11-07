"""Label-separated FAISS cosine retrieval with dataset and source-file binding."""

from pathlib import Path
import random

import faiss
import numpy as np

from .io import digest, file_hash, read_json, write_json


def records_for(train):
    return [{"row_id": r.query.row_id, "text": r.query.text, "label": r.label} for r in train]


def index_identity(dataset, train, report, config, client):
    if not config.retrieves(dataset):
        raise ValueError("MUSTARD disables Stage 1 and must not build or load retrieval indexes")
    if not train or "train" not in report["files"]:
        raise ValueError("No training split for retrieval")
    return {"dataset": dataset, "split": "train", "train_sha256": report["files"]["train"]["sha256"],
            "records_sha256": digest(records_for(train)), "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions, "endpoint": client.embedding_endpoint,
            "retrieval_order_seed": config.retrieval_order_seed}


def location(root, identity):
    return Path(root) / identity["dataset"] / digest(identity)[:20]


def normalized(vectors):
    values = np.array(vectors, dtype="float32", order="C", copy=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Invalid or zero embedding vectors")
    return np.ascontiguousarray(values / norms)


def build_index(root, dataset, train, report, config, client, batch_size=64):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    identity = index_identity(dataset, train, report, config, client)
    directory = location(root, identity)
    if (directory / "manifest.json").exists():
        Retriever(root, dataset, train, report, config, client)
        return directory
    records = records_for(train)
    for label in (0, 1):
        if sum(r["label"] == label for r in records) < config.k:
            raise ValueError(f"Training subset {label} has fewer than k={config.k} records")
    # Embedding order is independent of insertion-order seeds. A saved response is reused exactly.
    vectors = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        scope = f'{dataset}:train:{identity["train_sha256"]}:embeddings:{start}'
        vectors.append(client.embed([r["text"] for r in batch], config, scope))
    vectors = normalized(np.concatenate(vectors))
    directory.mkdir(parents=True, exist_ok=True)
    stored_records = {}
    for label in (0, 1):
        indices = [i for i, r in enumerate(records) if r["label"] == label]
        random.Random(config.retrieval_order_seed + label).shuffle(indices)
        index = faiss.IndexFlatIP(config.embedding_dimensions)
        index.add(np.ascontiguousarray(vectors[indices]))
        faiss.write_index(index, str(directory / f"label_{label}.faiss"))
        stored_records[str(label)] = [records[i] for i in indices]
    write_json(directory / "records.json", stored_records)
    files = {name: file_hash(directory / name) for name in ("records.json", "label_0.faiss", "label_1.faiss")}
    write_json(directory / "manifest.json", {"identity": identity, "files": files,
               "normalization": "L2", "metric": "cosine_via_IndexFlatIP", "faiss_version": faiss.__version__})
    return directory


class Retriever:
    def __init__(self, root, dataset, train, report, config, client):
        self.config = config
        self.client = client
        identity = index_identity(dataset, train, report, config, client)
        self.directory = location(root, identity)
        path = self.directory / "manifest.json"
        if not path.exists():
            raise ValueError(f"No matching train-only index. Run build-index first: {self.directory}")
        self.manifest = read_json(path)
        if self.manifest.get("identity") != identity:
            raise ValueError("Index dataset, source, model, dimensions, or insertion seed mismatch")
        expected_files = {"records.json", "label_0.faiss", "label_1.faiss"}
        if set(self.manifest.get("files", {})) != expected_files:
            raise ValueError("Incomplete index file manifest")
        for name, sha in self.manifest["files"].items():
            if file_hash(self.directory / name) != sha:
                raise ValueError(f"Index file changed: {name}")
        self.records = read_json(self.directory / "records.json")
        self.indexes = {}
        flat = []
        for label in (0, 1):
            rows = self.records[str(label)]
            index = faiss.read_index(str(self.directory / f"label_{label}.faiss"))
            if index.d != config.embedding_dimensions or index.ntotal != len(rows) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise ValueError("Index shape or metric mismatch")
            if len(rows) < config.k or any(r["label"] != label for r in rows):
                raise ValueError("Incomplete or mislabeled retrieval subset")
            self.indexes[label] = index
            flat.extend(rows)
        if sorted(flat, key=lambda r: r["row_id"]) != sorted(records_for(train), key=lambda r: r["row_id"]):
            raise ValueError("Stored index records are not the current training split")

    def retrieve(self, query):
        vector = normalized(self.client.embed([query.text], self.config, f"query:{query.row_id}"))
        exemplars = []
        for label in (0, 1):
            scores, indexes = self.indexes[label].search(vector, self.config.k)
            for score, index in zip(scores[0], indexes[0]):
                if index < 0 or index >= len(self.records[str(label)]) or not np.isfinite(score):
                    raise ValueError("FAISS returned an invalid neighbor")
                record = self.records[str(label)][int(index)]
                exemplars.append({**record, "similarity": float(score)})
        return exemplars
