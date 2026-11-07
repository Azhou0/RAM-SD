import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

from .client import RecordedClient, load_env_file, safe_error
from .config import Config, DATASETS
from .data import load_data
from .io import read_json, write_json
from .metrics import paired_mcnemar, summarize_runs
from .prepare import convert_csv, convert_mustard
from .retrieval import Retriever, build_index
from .runner import evaluate


def parser():
    result = argparse.ArgumentParser(prog="ram-sd", description="RAM-SD method reproduction; original paper scores are not certified")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("audit-data", "build-index", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--data-dir", type=Path, required=True, help="Root containing DATASET/train.csv and DATASET/test.csv")
        command.add_argument("--strict-counts", action="store_true", help="Require Table 1 counts; this does not prove original split identity")
        if name == "audit-data":
            command.add_argument("--output", type=Path)
            continue
        command.add_argument("--config", type=Path, help="Default: Config() matches configs/appendix.json")
        command.add_argument("--env-file", type=Path, help="Explicitly load supported variables from this file; overrides those variables in the current process")
        command.add_argument("--retrieval-order-seed", type=int, help="Override training-vector insertion-order seed only")
        command.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes"))
        command.add_argument("--cache-dir", type=Path, default=Path(".cache/responses"))
        command.add_argument("--mode", choices=("replay", "live"), default="replay", help="Replay never calls the API; live uses saved responses before making new requests")
        if name == "build-index":
            command.add_argument("--batch-size", type=int, default=64)
        else:
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--limit", type=int, help="First N test rows, marked as a partial evaluation")
            command.add_argument("--keep-going", action="store_true", help="Continue after failed samples; failed runs never get a headline score")
    command = sub.add_parser("prepare-csv", help="Convert an existing split; never resplit or silently drop records")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--text-column", required=True)
    command.add_argument("--label-column", required=True)
    command.add_argument("--context-column")
    command.add_argument("--id-column")
    command.add_argument("--positive-label", default="1")
    command.add_argument("--negative-label", default="0")
    command.add_argument("--delimiter", choices=("comma", "tab"), default="comma")
    command = sub.add_parser("prepare-mustard", help="Convert official JSON using explicit split IDs; no fold is guessed")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--ids", type=Path, required=True, help="JSON array of sample-ID strings for one chosen split")
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("summarize", help="Mean/sample std across compatible completed runs, then mean over four datasets")
    command.add_argument("metrics", type=Path, nargs="+")
    command.add_argument("--output", type=Path)
    command = sub.add_parser("compare", help="Paired exact McNemar test using imported baseline predictions")
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--baseline", type=Path, required=True)
    command.add_argument("--output", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    client = None
    try:
        if args.command == "prepare-csv":
            convert_csv(args.input, args.output, args.text_column, args.label_column, args.context_column,
                        args.id_column, args.positive_label, args.negative_label,
                        "\t" if args.delimiter == "tab" else ",")
            result = {"output": str(args.output), "operation": "format_conversion_only"}
        elif args.command == "prepare-mustard":
            convert_mustard(args.input, args.ids, args.output)
            result = {"output": str(args.output), "operation": "format_conversion_only"}
        elif args.command == "summarize":
            result = summarize_runs([read_json(p) for p in args.metrics])
        elif args.command == "compare":
            def rows(path):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    return list(csv.DictReader(stream))
            result = paired_mcnemar(rows(args.predictions), rows(args.baseline))
        else:
            train, test, report = load_data(args.data_dir, args.dataset, args.strict_counts)
            if args.command == "audit-data":
                result = report
            else:
                if args.env_file:
                    load_env_file(args.env_file)
                config = Config.load(args.config) if args.config else Config()
                if args.retrieval_order_seed is not None:
                    config = replace(config, retrieval_order_seed=args.retrieval_order_seed)
                client = RecordedClient(args.cache_dir, args.mode)
                for warning in report["warnings"]:
                    print(f"Data note: {warning}", file=sys.stderr)
                if args.command == "build-index":
                    directory = build_index(args.index_dir, args.dataset, train, report, config, client, args.batch_size)
                    result = {"index_directory": str(directory), "data_audit": report}
                else:
                    retriever = Retriever(args.index_dir, args.dataset, train, report, config, client) if config.retrieves(args.dataset) else None
                    result = evaluate(args.dataset, test, report, config, client, retriever, args.out,
                                      args.resume, args.limit, args.keep_going)
        if getattr(args, "output", None) and args.command in ("audit-data", "compare", "summarize"):
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("status") == "failed" else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f"ram-sd: {safe_error(error)}", file=sys.stderr)
        return 2
    finally:
        if client:
            client.close()
