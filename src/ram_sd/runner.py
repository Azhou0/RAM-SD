"""Sequential, resumable evaluation bound to code, config, data, index and provider."""

from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform

from . import __version__, prompts
from .client import safe_error
from .io import digest, file_hash, read_json, write_json
from .metrics import binary_metrics
from .parsing import appendix_verdict, maintext_verdict
from .pipeline import Pipeline


def runtime_identity():
    packages = {}
    for name in ("openai", "numpy", "faiss-cpu"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "implementation_version": __version__,
            "source_files": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "prompts_sha256": prompts.fingerprint()}


@contextmanager
def run_lock(directory):
    path = directory / ".run.lock"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(f"Output directory is locked: {path}. Check the owning process before removing a stale lock.") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def validate_saved(record, input_fingerprint, query, config):
    trace = record["trace"]
    if record.get("input_fingerprint") != input_fingerprint or record.get("trace_sha256") != digest(trace):
        raise ValueError("Saved trace checksum or input mismatch")
    if trace.get("row_id") != query.row_id or trace.get("status") != "ok":
        raise ValueError("Invalid successful saved trace")
    final_stage = "integrator" if config.profile == "appendix" else "judger"
    outputs = [s["output"] for s in trace["steps"] if s["stage"] == final_stage]
    if len(outputs) != 1:
        raise ValueError("Saved trace has missing or duplicated final stage")
    parsed = (appendix_verdict if config.profile == "appendix" else maintext_verdict)(outputs[0])
    if any(trace.get(key) != value for key, value in parsed.items()):
        raise ValueError("Saved prediction disagrees with its recorded model response")
    return trace


def evaluate(dataset, test, report, config, client, retriever, out, resume=False, limit=None, keep_going=False):
    if report.get("dataset") != dataset or report.get("test_records_sha256") != digest([asdict(r) for r in test]):
        raise ValueError("Dataset audit is not bound to these evaluation records")
    if config.retrieves(dataset) and retriever is None:
        raise ValueError("A validated train-only retriever is required for this dataset")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    selected = test[:limit] if limit is not None else test
    if not selected:
        raise ValueError("No test records")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        return _evaluate(dataset, test, selected, report, config, client, retriever, directory, resume, keep_going)


def _evaluate(dataset, all_test, selected, report, config, client, retriever, directory, resume, keep_going):
    runtime = runtime_identity()
    identity = {"dataset": dataset, "config": config.as_dict(), "data": report,
                "selected_row_ids": [r.query.row_id for r in selected], "runtime": runtime,
                "chat_endpoint": client.chat_endpoint,
                "embedding_endpoint": client.embedding_endpoint if config.retrieves(dataset) else None,
                "index": retriever.manifest if retriever else None}
    fingerprint = digest(identity)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if not resume:
            raise ValueError("Output already contains a run. Use a new directory or --resume.")
        if previous.get("experiment_fingerprint") != fingerprint:
            raise ValueError("Cannot resume: code, config, data, selected rows, runtime, provider or index changed")
    elif any(p.name != ".run.lock" for p in directory.iterdir()):
        raise ValueError("Refusing to write into a nonempty output directory without a matching manifest")
    manifest = {"experiment_fingerprint": fingerprint, "identity": identity,
                "invocation_utc": datetime.now(timezone.utc).isoformat(),
                "mode": client.mode, "status": "running", "data_split_recovered": False,
                "claim": "method_reimplementation; historical paper scores not verified"}
    write_json(manifest_path, manifest)
    pipeline = Pipeline(config, client, retriever)
    attempted = {}
    failed = 0
    traces_dir = directory / "traces"
    for index, row in enumerate(selected):
        query = row.query
        input_fingerprint = digest(asdict(row))
        path = traces_dir / f"{digest(query.row_id)}.json"
        try:
            if path.exists() and resume:
                saved = read_json(path)
                if saved.get("trace", {}).get("status") == "ok":
                    trace = validate_saved(saved, input_fingerprint, query, config)
                else:
                    trace = pipeline.predict(query, dataset)
            else:
                trace = pipeline.predict(query, dataset)
        except Exception as error:
            # Do not overwrite a previously completed trace that failed integrity validation.
            if path.exists() and read_json(path).get("trace", {}).get("status") == "ok":
                raise
            trace = dict(pipeline.last_trace or {"row_id": query.row_id})
            trace.update({"status": "failed", "error_type": type(error).__name__, "error": safe_error(error)})
            failed += 1
        write_json(path, {"input_fingerprint": input_fingerprint, "trace": trace, "trace_sha256": digest(trace)})
        attempted[query.row_id] = trace
        print(f'[{index + 1}/{len(selected)}] {query.row_id}: {trace["status"]}', flush=True)
        if trace["status"] == "failed" and not keep_going:
            break
    predictions_path = directory / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "label", "prediction", "confidence", "p_sarcasm", "plan_type", "status"])
        writer.writeheader()
        for row in selected:
            trace = attempted.get(row.query.row_id, {})
            writer.writerow({"row_id": row.query.row_id, "label": row.label, "status": trace.get("status", "not_attempted"),
                             **{key: trace.get(key, "") for key in ("prediction", "confidence", "p_sarcasm", "plan_type")}})
    complete = len(attempted) == len(selected) and failed == 0
    comparison_config = config.as_dict()
    comparison_config.pop("retrieval_order_seed")
    result = {"dataset": dataset, "status": "complete" if complete else "failed",
              "full_dataset": len(selected) == len(all_test), "requested_rows": len(selected),
              "successful_rows": sum(t["status"] == "ok" for t in attempted.values()),
              "failed_rows": failed, "not_attempted_rows": len(selected) - len(attempted),
              "experiment_fingerprint": fingerprint, "data_split_recovered": False,
              "comparison_signature": digest({"dataset": dataset, "config": comparison_config, "data": report,
                  "runtime": runtime, "chat_endpoint": client.chat_endpoint,
                  "embedding_endpoint": identity["embedding_endpoint"]}),
              "config": config.as_dict(), "metrics": None,
              "requests_this_invocation": {"total": len(client.events),
                   "cache_hits": sum(e["cached"] for e in client.events),
                   "live_requests": sum(not e["cached"] for e in client.events)}}
    if complete:
        result["metrics"] = binary_metrics([r.label for r in selected], [attempted[r.query.row_id]["prediction"] for r in selected])
    write_json(directory / "requests.json", client.events)
    write_json(directory / "metrics.json", result)
    manifest["status"] = result["status"]
    write_json(manifest_path, manifest)
    return result
