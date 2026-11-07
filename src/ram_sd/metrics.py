"""Binary metrics and paired exact McNemar test, without inventing missing predictions."""

import math
import statistics


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require nonempty aligned labels and predictions")
    if any(type(x) is not int or x not in (0, 1) for x in [*labels, *predictions]):
        raise ValueError("Metrics accept integer 0/1 values only")
    matrix = [[0, 0], [0, 0]]
    for y, p in zip(labels, predictions):
        matrix[y][p] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    def ratio(a, b):
        return a / b if b else 0.0
    f1_0 = ratio(2 * tn, 2 * tn + fp + fn)
    f1_1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {"n": len(labels), "accuracy": (tn + tp) / len(labels),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "f1": f1_1,
            "macro_f1": (f1_0 + f1_1) / 2,
            "confusion_matrix": matrix, "confusion_matrix_order": [0, 1], "scale": "fraction_0_to_1"}


def paired_mcnemar(records, baseline):
    def keyed(rows):
        result = {}
        for row in rows:
            if "status" in row and row["status"] != "ok":
                raise ValueError("Failed or unattempted predictions cannot enter a paired test")
            if row["row_id"] in result:
                raise ValueError("Duplicate row_id in paired predictions")
            if str(row["label"]) not in ("0", "1") or str(row["prediction"]) not in ("0", "1"):
                raise ValueError("Only complete binary predictions can be compared")
            result[row["row_id"]] = (int(row["label"]), int(row["prediction"]))
        return result
    ours, theirs = keyed(records), keyed(baseline)
    if not ours or ours.keys() != theirs.keys():
        raise ValueError("McNemar comparison requires exactly matching row_id sets")
    b = c = 0
    for key, (label, prediction) in ours.items():
        other_label, other_prediction = theirs[key]
        if label != other_label:
            raise ValueError("Gold labels differ between the paired files")
        b += int(prediction == label and other_prediction != label)
        c += int(prediction != label and other_prediction == label)
    n = b + c
    probability = 1.0 if not n else min(1.0, 2 * math.fsum(
        math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2))
        for k in range(min(b, c) + 1)))
    return {"n_pairs": len(ours), "ours_only_correct": b, "baseline_only_correct": c,
            "p_value": probability, "test": "exact_two_sided_conditional_binomial_McNemar"}


def summarize_runs(runs):
    if not runs:
        raise ValueError("No metrics to summarize")
    groups = {}
    seen = set()
    for run in runs:
        if run.get("status") != "complete" or not run.get("full_dataset"):
            raise ValueError("Only complete, full-dataset evaluations can be summarized")
        if run["experiment_fingerprint"] in seen:
            raise ValueError("Duplicate run fingerprint; replay is not another independent run")
        seen.add(run["experiment_fingerprint"])
        groups.setdefault(run["dataset"], []).append(run)
    output = {}
    for dataset, entries in groups.items():
        signatures = {r["comparison_signature"] for r in entries}
        if len(signatures) != 1:
            raise ValueError(f"Incompatible configurations or datasets for {dataset}")
        output[dataset] = {"runs": len(entries)}
        for metric in ("accuracy", "macro_f1"):
            values = [r["metrics"][metric] for r in entries]
            output[dataset][metric] = {"mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    result = {"datasets": output, "std_definition": "sample standard deviation, ddof=1; null for n=1",
              "data_splits_recovered": all(r.get("data_split_recovered", False) for r in runs)}
    if set(groups) == {"iacv1", "iacv2", "mustard", "semeval"}:
        result["unweighted_four_dataset_mean"] = {m: statistics.mean(v[m]["mean"] for v in output.values())
                                                 for m in ("accuracy", "macro_f1")}
    return result
