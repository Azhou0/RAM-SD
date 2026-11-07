"""Small live integration check; never a reproduction of the paper's scores.

Uses 16 local training examples for retrieval and evaluates only 2 IAC-V1 +
1 MUSTARD records. Input copies, indexes, responses and predictions stay in ignored
local directories. The release builder includes none of these artifacts.
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil

from ram_sd.client import RecordedClient, load_env_file
from ram_sd.config import Config
from ram_sd.data import load_data, read_rows
from ram_sd.io import file_hash, write_json
from ram_sd.retrieval import Retriever, build_index
from ram_sd.runner import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", type=Path, required=True, help="Local legacy CSV directory; no downloads")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_terra.json"))
    parser.add_argument("--mode", choices=("live", "replay"), default="replay")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/openai_smoke_terra"))
    args = parser.parse_args()
    load_env_file(args.env_file)
    config = Config.load(args.config)
    if config.k != 3:
        raise ValueError("This smoke check uses k=3")
    data_root = Path(".cache/openai_smoke_data")
    selected, counts = [], Counter()
    source_train = args.source_data_dir / "train_iacv1.csv"
    for row in read_rows(source_train, "iacv1", "train"):
        if counts[row.label] < 8:
            selected.append(row)
            counts[row.label] += 1
    if counts != {0: 8, 1: 8}:
        raise ValueError("Need eight valid training records per class for this smoke test")
    (data_root / "iacv1").mkdir(parents=True, exist_ok=True)
    with (data_root / "iacv1/train.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("row_id", "Text", "Label"))
        writer.writeheader()
        writer.writerows({"row_id": r.query.row_id, "Text": r.query.text, "Label": r.label} for r in selected)
    for dataset in ("iacv1", "mustard"):
        target = data_root / dataset / "test.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source_data_dir / f"test_{dataset}.csv", target)

    summary = {"scope": "live_integration_smoke_test", "not_paper_score_reproduction": True,
               "requested_model": config.model, "requested_samples": 3, "datasets": {},
               "train_subset": {"source_sha256": file_hash(source_train), "counts": dict(counts),
                                "selection": "first eight source rows per label, original order"}}
    for dataset, limit in (("iacv1", 2), ("mustard", 1)):
        print(f"Starting {dataset}: at most {limit} test sample(s)", flush=True)
        train, test, report = load_data(data_root, dataset)
        report["experiment_scope"] = "small_integration_smoke_test"
        report["retrieval_training_subset"] = dataset == "iacv1"
        client = RecordedClient(".cache/openai_smoke_responses", args.mode,
                                chat_endpoint="https://api.openai.com/v1", embedding_endpoint="https://api.openai.com/v1")
        try:
            retriever = None
            if config.retrieves(dataset):
                build_index("artifacts/openai_smoke_indexes", dataset, train, report, config, client)
                retriever = Retriever("artifacts/openai_smoke_indexes", dataset, train, report, config, client)
            result = evaluate(dataset, test, report, config, client, retriever,
                              args.out_root / dataset, resume=True, limit=limit)
            usage = [event.get("usage") or {} for event in client.events if not event["cached"]]
            summary["datasets"][dataset] = {
                "status": result["status"], "successful_samples": result["successful_rows"],
                "failed_samples": result["failed_rows"], "full_dataset": result["full_dataset"],
                "requests": result["requests_this_invocation"],
                "actual_returned_models": sorted({e["model"] for e in client.events if e.get("model")}),
                "new_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                "new_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            }
            write_json(args.out_root / "summary.json", summary)
            if result["status"] != "complete":
                print("Smoke check stopped after a failed sample. Inspect its trace; no fallback label was inserted.", flush=True)
                return 1
        finally:
            client.close()
    summary["status"] = "passed"
    summary["successful_samples"] = sum(r["successful_samples"] for r in summary["datasets"].values())
    write_json(args.out_root / "summary.json", summary)
    print(f'Completed {summary["successful_samples"]} integration smoke samples; this is not a benchmark result.', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
