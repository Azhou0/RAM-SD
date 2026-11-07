"""Synthetic fixtures only. No author dataset, credentials, or network requests."""

from contextlib import redirect_stdout, redirect_stderr
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ram_sd.cli import main
from ram_sd.client import CacheMiss, RecordedClient, load_env_file
from ram_sd.config import Config
from ram_sd.data import Query, Row, audit, load_data, read_rows
from ram_sd.io import digest, read_json, write_json
from ram_sd.metrics import binary_metrics, paired_mcnemar, summarize_runs
from ram_sd.parsing import appendix_verdict, maintext_verdict, planner_output
from ram_sd.pipeline import Pipeline, agents_for, validate_plan
from ram_sd.prepare import convert_csv, convert_mustard
from ram_sd.retrieval import Retriever, build_index, normalized
from ram_sd.runner import evaluate


class FixtureTransport:
    def __init__(self, plan="expectation_violation", fail_stage=None, finish_reason="stop", zero_vectors=False):
        self.plan = plan
        self.calls = []
        self.fail_stage = fail_stage
        self.finish_reason = finish_reason
        self.zero_vectors = zero_vectors

    def __call__(self, kind, payload, scope):
        self.calls.append((kind, payload, scope))
        stage = "rationale" if ":rationale:" in scope else scope.rsplit(":", 1)[-1]
        if self.fail_stage == stage:
            raise RuntimeError("Synthetic transport failure")
        if kind == "embedding":
            vectors = []
            for i, text in enumerate(payload["input"]):
                vector = [0.0] * payload["dimensions"]
                if not self.zero_vectors:
                    vector[int(text.startswith("one"))] = 2.0
                vectors.append({"index": i, "embedding": vector})
            # Deliberately reorder the server response: the client must use its index field.
            return {"model": "synthetic-embedding-fixture", "data": list(reversed(vectors))}
        if stage == "planner":
            content = json.dumps({"selected_plan": self.plan, "confidence": 0.8,
                                  "contextual_analysis": "Fixture inferred context", "reasoning": "Fixture plan reasoning",
                                  "knowledge_entities": []})
        elif stage == "integrator":
            content = '<<LABEL>> 0\n{"label":0,"conf":0.9,"reasoning":"Fixture literal expression"}'
        elif stage == "judger":
            content = '{"p_sarcasm":0.7,"explanation":"Fixture probability judgment"}'
        else:
            content = "Fixture analysis for " + stage
        return {"id": "synthetic-response", "model": "synthetic-chat-fixture", "system_fingerprint": "fixture",
                "choices": [{"finish_reason": self.finish_reason, "message": {"content": content}}]}


def csv_file(path, rows, fields=("Text", "Label")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = Config(embedding_dimensions=4, k=1)
        directory = self.root / "data" / "iacv1"
        csv_file(directory / "train.csv", [{"Text": "zero train A", "Label": 0}, {"Text": "zero train B", "Label": 0},
                                           {"Text": "one train A", "Label": 1}, {"Text": "one train B", "Label": 1}])
        csv_file(directory / "test.csv", [{"Text": "query A", "Label": 0}, {"Text": "query B", "Label": 1}])
        self.train, self.test, self.report = load_data(self.root / "data", "iacv1")
        mustard = self.root / "data" / "mustard"
        csv_file(mustard / "test.csv", [{"Text": "utterance A", "Label": 0, "Context": "dialogue A"},
                                       {"Text": "utterance B", "Label": 1, "Context": "dialogue B"}],
                 ("Text", "Label", "Context"))
        _, self.mustard_test, self.mustard_report = load_data(self.root / "data", "mustard")

    def tearDown(self):
        self.temporary.cleanup()

    def client(self, fixture=None, mode="live", cache="cache"):
        return RecordedClient(self.root / cache, mode, fixture or FixtureTransport(),
                              chat_endpoint="https://fixture.invalid/v1", embedding_endpoint="https://fixture.invalid/v1")

    def indexed(self, config=None, fixture=None):
        config = config or self.config
        client = self.client(fixture)
        build_index(self.root / "indexes", "iacv1", self.train, self.report, config, client)
        retriever = Retriever(self.root / "indexes", "iacv1", self.train, self.report, config, client)
        return client, retriever

    def run_quiet(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return evaluate(*args, **kwargs)

    def mustard_run(self, client, out, config=None, **kwargs):
        return self.run_quiet("mustard", self.mustard_test, self.mustard_report,
                              config or self.config, client, None, out, **kwargs)

    def test_labels_invalid_or_blank_rows_never_dropped(self):
        for row in [{"Text": "bad", "Label": 2}, {"Text": " ", "Label": 0}]:
            csv_file(self.root / "bad.csv", [row])
            with self.assertRaises(ValueError):
                read_rows(self.root / "bad.csv", "iacv1", "test")

    def test_train_test_overlap_normalizes_case_and_whitespace(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            audit("iacv1", [Row(Query("train", " A  TEXT "), 0)], [Row(Query("test", "a text"), 1)], {})

    def test_counts_are_reported_without_resplitting(self):
        self.assertEqual(self.report["train"]["rows"], 4)
        self.assertEqual(self.report["split_identity"], "unverified_against_original_paper")
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            load_data(self.root / "data", "iacv1", strict_counts=True)

    def test_mustard_requires_dialogue_context(self):
        csv_file(self.root / "mustard.csv", [{"Text": "hello", "Label": 0}])
        with self.assertRaisesRegex(ValueError, "Context"):
            read_rows(self.root / "mustard.csv", "mustard", "test")

    def test_mustard_repeated_text_with_distinct_context_is_not_overlap(self):
        result = audit("mustard", [Row(Query("a", "hello", "before A"), 0)],
                       [Row(Query("b", "hello", "before B"), 1)], {})
        self.assertEqual(result["normalized_train_test_overlap"], 0)

    def test_duplicate_row_ids_rejected(self):
        csv_file(self.root / "bad.csv", [{"row_id": "a", "Text": "a", "Label": 0},
                                         {"row_id": "a", "Text": "b", "Label": 1}], ("row_id", "Text", "Label"))
        with self.assertRaisesRegex(ValueError, "duplicate row_id"):
            read_rows(self.root / "bad.csv", "iacv1", "test")

    def test_config_rejects_nonfinite_or_invalid_counts(self):
        for values in ({"temperature": float("nan")}, {"k": 0}, {"n": 2}, {"use_context_agent": 1}):
            with self.assertRaises(ValueError):
                Config(**values)

    def test_terra_adapter_keeps_paper_sampling_values_separate(self):
        paper = Config()
        terra = replace(paper, model="gpt-6.1-terra", output_token_parameter="max_completion_tokens", reasoning_effort="none")
        self.assertEqual(paper.model, "gpt-4o")
        self.assertEqual(paper.generation()["max_tokens"], 512)
        self.assertNotIn("reasoning_effort", paper.generation())
        self.assertNotIn("max_tokens", terra.generation())
        self.assertEqual(terra.generation()["max_completion_tokens"], 512)
        self.assertEqual(terra.generation()["temperature"], 0.1)
        self.assertEqual(terra.generation()["top_p"], 1)
        self.assertEqual(terra.generation()["reasoning_effort"], "none")

    def test_env_loading_is_explicit_and_restricts_variables(self):
        import os
        from unittest.mock import patch
        path = self.root / ".env"
        path.write_text('# fixture only\nOPENAI_API_KEY="test-value"\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            load_env_file(path)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-value")
        path.write_text('OPENAI_API_KEY=test-value\nPATH=unwanted\n')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported env"):
                load_env_file(path)
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_k_is_applied_to_both_label_subsets(self):
        config = replace(self.config, k=2)
        client, retriever = self.indexed(config)
        neighbors = retriever.retrieve(self.test[0].query)
        self.assertEqual([e["label"] for e in neighbors], [0, 0, 1, 1])
        self.assertTrue(all(abs(e["similarity"]) <= 1.00001 for e in neighbors))

    def test_too_few_neighbors_fails_before_any_api_request(self):
        transport = FixtureTransport()
        with self.assertRaisesRegex(ValueError, "fewer than k"):
            build_index(self.root / "indexes", "iacv1", self.train, self.report, replace(self.config, k=3), self.client(transport))
        self.assertEqual(transport.calls, [])

    def test_zero_vectors_are_not_silently_inserted(self):
        with self.assertRaisesRegex(ValueError, "Zero embedding"):
            build_index(self.root / "indexes", "iacv1", self.train, self.report, self.config, self.client(FixtureTransport(zero_vectors=True)))
        self.assertEqual(list((self.root / "indexes").rglob("manifest.json")), [])

    def test_index_rejects_wrong_train_source(self):
        client, _ = self.indexed()
        report = {**self.report, "files": {**self.report["files"], "train": {"sha256": "changed"}}}
        with self.assertRaisesRegex(ValueError, "No matching train-only index"):
            Retriever(self.root / "indexes", "iacv1", self.train, report, self.config, client)

    def test_index_integrity_checked_before_load(self):
        client, retriever = self.indexed()
        with (retriever.directory / "records.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Index file changed"):
            Retriever(self.root / "indexes", "iacv1", self.train, self.report, self.config, client)

    def test_negative_faiss_neighbor_not_python_negative_index(self):
        _, retriever = self.indexed()
        class InvalidIndex:
            def search(self, query, k):
                return np.array([[0.0]]), np.array([[-1]])
        retriever.indexes[0] = InvalidIndex()
        with self.assertRaisesRegex(ValueError, "invalid neighbor"):
            retriever.retrieve(self.test[0].query)

    def test_l2_normalization_and_zero_rejection(self):
        np.testing.assert_allclose(normalized([[3.0, 4.0]]), [[0.6, 0.8]])
        with self.assertRaises(ValueError):
            normalized([[0.0, 0.0]])

    def test_replay_never_calls_transport(self):
        original = self.client()
        first = original.chat("fixture prompt", self.config, "fixture:semantic")
        def forbidden(*args):
            self.fail("Replay attempted a network request")
        replay = self.client(forbidden, mode="replay")
        self.assertEqual(replay.chat("fixture prompt", self.config, "fixture:semantic"), first)
        with self.assertRaises(CacheMiss):
            replay.chat("different prompt", self.config, "fixture:semantic")

    def test_cache_tampering_is_detected(self):
        client = self.client()
        client.chat("fixture", self.config, "fixture:semantic")
        path = next((self.root / "cache" / "chat").glob("*.json"))
        value = read_json(path)
        value["response"]["model"] = "changed"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            client.chat("fixture", self.config, "fixture:semantic")

    def test_truncated_chat_is_failure(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.client(FixtureTransport(finish_reason="length")).chat("fixture", self.config, "fixture:semantic")

    def test_pipeline_all_three_plans_with_real_faiss(self):
        for plan in ("expectation_violation", "knowledge_dependent", "simple_irony"):
            with self.subTest(plan=plan):
                fixture = FixtureTransport(plan)
                client = self.client(fixture, cache="cache_" + plan)
                index_dir = self.root / ("index_" + plan)
                build_index(index_dir, "iacv1", self.train, self.report, self.config, client)
                retriever = Retriever(index_dir, "iacv1", self.train, self.report, self.config, client)
                result = Pipeline(self.config, client, retriever).predict(self.test[0].query, "iacv1")
                stages = [s["stage"] for s in result["steps"]]
                self.assertEqual(result["plan_type"], plan)
                self.assertLess(stages.index("similarity"), stages.index("planner"))
                self.assertEqual(len(result["exemplars"]), 2)
                self.assertTrue(all(e["rationale"] for e in result["exemplars"]))
                self.assertIn("rhetoric", result["agents_executed"])
                self.assertIsNone(result["p_sarcasm"])

    def test_mustard_has_no_retrieval_and_context_reaches_all_stages(self):
        fixture = FixtureTransport("knowledge_dependent")
        client = self.client(fixture)
        class ForbiddenRetriever:
            def retrieve(self, query):
                raise AssertionError("MUSTARD must not retrieve")
        result = Pipeline(self.config, client, ForbiddenRetriever()).predict(Query("m1", "fixture utterance", "CONTEXT_SENTINEL"), "mustard")
        self.assertFalse(result["retrieval_enabled"])
        self.assertEqual(result["exemplars"], [])
        self.assertTrue(all(kind == "chat" for kind, _, _ in fixture.calls))
        self.assertTrue(all("CONTEXT_SENTINEL" in payload["messages"][0]["content"] for _, payload, _ in fixture.calls))

    def test_gold_label_does_not_change_model_requests(self):
        query = Query("gold-test", "fixture utterance", "fixture context")
        first = self.client(cache="gold")
        report0 = audit("mustard", [], [Row(query, 0)], {})
        report1 = audit("mustard", [], [Row(query, 1)], {})
        result1 = self.run_quiet("mustard", [Row(query, 0)], report0, self.config, first, None, self.root / "gold0")
        second = self.client(mode="replay", cache="gold")
        result2 = self.run_quiet("mustard", [Row(query, 1)], report1, self.config, second, None, self.root / "gold1")
        self.assertEqual(result1["metrics"]["accuracy"], 1)
        self.assertEqual(result2["metrics"]["accuracy"], 0)
        self.assertTrue(all(e["cached"] for e in second.events))

    def test_maintext_uses_separate_judger_and_probability_threshold(self):
        config = replace(self.config, profile="maintext", use_context_agent=False,
                         planner_rule_validation=False, simple_irony_semantic_support=False)
        result = Pipeline(config, self.client()).predict(Query("m1", "fixture", "context"), "mustard")
        self.assertEqual(result["steps"][-1]["stage"], "judger")
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["p_sarcasm"], 0.7)
        self.assertEqual(maintext_verdict('{"p_sarcasm":0.5,"explanation":"boundary"}')["prediction"], 1)

    def test_appendix_confidence_is_not_class_probability(self):
        result = appendix_verdict('<<LABEL>> 0\n{"label":0,"conf":0.99,"reasoning":"literal"}')
        self.assertEqual(result["prediction"], 0)
        self.assertIsNone(result["p_sarcasm"])

    def test_conflicting_label_and_nonfinite_probabilities_rejected(self):
        for text in ('<<LABEL>> 1\n{"label":0,"conf":0.9,"reasoning":"x"}',
                     '{"label":true,"conf":0.9,"reasoning":"x"}',
                     '{"label":0,"conf":NaN,"reasoning":"x"}'):
            with self.assertRaises(ValueError):
                appendix_verdict(text)

    def test_malformed_outer_json_does_not_salvage_inner_label(self):
        with self.assertRaises(ValueError):
            appendix_verdict('{"broken": {"label":0,"conf":0.9,"reasoning":"inner"}')

    def test_duplicate_fields_or_invalid_label_tags_are_rejected(self):
        for value in ('{"label":0,"label":1,"conf":0.9,"reasoning":"x"}',
                      '<<LABEL>> 2\n{"label":0,"conf":0.9,"reasoning":"x"}'):
            with self.assertRaises(ValueError):
                appendix_verdict(value)

    def test_real_openai_sdk_round_trip_with_mock_http(self):
        import httpx
        from openai import OpenAI
        payloads = []
        def handler(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            if request.url.path.endswith("embeddings"):
                return httpx.Response(200, json={"object": "list", "model": "fixture-embedding",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
            return httpx.Response(200, json={"id": "fixture-sdk", "created": 1, "object": "chat.completion",
                "model": "fixture-chat", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Fixture SDK output"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        client = RecordedClient(self.root / "sdk", "live", chat_endpoint="https://fixture.invalid/v1",
                                embedding_endpoint="https://fixture.invalid/v1")
        for kind in ("chat", "embedding"):
            client._sdk_clients[kind] = OpenAI(api_key="fixture-credential-never-save",
                base_url="https://fixture.invalid/v1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        try:
            self.assertEqual(client.chat("fixture", self.config, "sdk:semantic"), "Fixture SDK output")
            self.assertEqual(client.embed(["fixture"], self.config, "sdk:embedding").shape, (1, 4))
            self.assertEqual(payloads[0]["temperature"], 0.1)
            self.assertEqual(payloads[0]["max_tokens"], 512)
            self.assertEqual(payloads[0]["top_p"], 1)
            self.assertEqual(payloads[0]["n"], 1)
            self.assertEqual(payloads[1]["dimensions"], 4)
            self.assertTrue(all("fixture-credential-never-save" not in p.read_text() for p in (self.root / "sdk").rglob("*.json")))
        finally:
            client.close()

    def test_audit_must_match_evaluation_dataset_and_records(self):
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.run_quiet("mustard", self.test, self.report, self.config, self.client(), None, self.root / "mismatch")

    def test_rule_validation_follows_appendix_and_logs_override(self):
        raw = {"selected_plan": "simple_irony", "knowledge_entities": []}
        selected, changes = validate_plan(raw, "word " * 51, True)
        self.assertEqual(selected, "expectation_violation")
        self.assertTrue(changes)
        selected, _ = validate_plan({"selected_plan": "expectation_violation", "knowledge_entities": ["A", "B", "C"]}, "short", True)
        self.assertEqual(selected, "knowledge_dependent")
        # The undocumented short-text/exaggeration override from the local code is not reintroduced.
        self.assertEqual(validate_plan({"selected_plan": "expectation_violation", "knowledge_entities": []}, "short!", True)[0], "expectation_violation")

    def test_model_failure_is_not_a_negative_prediction_or_metric(self):
        client = self.client(FixtureTransport(fail_stage="planner"))
        result = self.mustard_run(client, self.root / "failure")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["failed_rows"], 1)
        self.assertEqual(result["not_attempted_rows"], 1)
        with (self.root / "failure" / "predictions.csv").open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(records[0]["prediction"], "")

    def test_resume_reuses_verified_successful_traces(self):
        client = self.client()
        out = self.root / "resume"
        first = self.mustard_run(client, out)
        replay = self.client(mode="replay")
        resumed = self.mustard_run(replay, out, resume=True)
        self.assertEqual(first["metrics"], resumed["metrics"])
        self.assertEqual(replay.events, [])
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.mustard_run(replay, out, config=replace(self.config, k=2), resume=True)

    def test_saved_prediction_checked_against_raw_final_response(self):
        client = self.client()
        out = self.root / "tamper"
        self.mustard_run(client, out)
        path = next((out / "traces").glob("*.json"))
        record = read_json(path)
        record["trace"]["prediction"] = 1
        record["trace_sha256"] = digest(record["trace"])
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.mustard_run(client, out, resume=True)

    def test_output_reuse_requires_resume(self):
        out = self.root / "existing"
        client = self.client()
        self.mustard_run(client, out)
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.mustard_run(client, out)

    def test_binary_metrics_known_confusion_matrix(self):
        result = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["macro_f1"], 0.5)

    def test_paired_exact_mcnemar_aligns_ids_and_rejects_missing_pairs(self):
        ours = [{"row_id": str(i), "label": 0, "prediction": 0} for i in range(5)]
        baseline = [{"row_id": str(i), "label": 0, "prediction": 1} for i in reversed(range(5))]
        self.assertAlmostEqual(paired_mcnemar(ours, baseline)["p_value"], 0.0625)
        with self.assertRaises(ValueError):
            paired_mcnemar(ours, baseline[:4])

    def test_partial_or_duplicate_runs_not_aggregated_as_replicates(self):
        run = {"dataset": "iacv1", "status": "complete", "full_dataset": True,
               "experiment_fingerprint": "a", "comparison_signature": "same", "metrics": {"accuracy": 0.7, "macro_f1": 0.6}}
        with self.assertRaises(ValueError):
            summarize_runs([run, run])
        with self.assertRaises(ValueError):
            summarize_runs([{**run, "full_dataset": False}])
        result = summarize_runs([run, {**run, "experiment_fingerprint": "b", "metrics": {"accuracy": 0.9, "macro_f1": 0.8}}])
        self.assertAlmostEqual(result["datasets"]["iacv1"]["accuracy"]["sample_std"], 0.1414213562)

    def test_format_conversion_preserves_rows_and_maps_labels(self):
        csv_file(self.root / "raw.csv", [{"Text": "fixture", "Label": "sarc"}])
        convert_csv(self.root / "raw.csv", self.root / "converted.csv", "Text", "Label", positive="sarc", negative="notsarc")
        result = read_rows(self.root / "converted.csv", "iacv1", "test")
        self.assertEqual(result[0].label, 1)
        self.assertEqual(result[0].query.text, "fixture")

    def test_mustard_conversion_requires_explicit_ids(self):
        write_json(self.root / "raw.json", {"idA": {"utterance": "fixture", "context": ["first", "second"], "sarcasm": True}})
        write_json(self.root / "ids.json", ["idA"])
        convert_mustard(self.root / "raw.json", self.root / "ids.json", self.root / "mustard.csv")
        result = read_rows(self.root / "mustard.csv", "mustard", "test")
        self.assertEqual(result[0].query.context, "first\nsecond")

    def test_cli_audit_is_offline(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = main(["audit-data", "--dataset", "iacv1", "--data-dir", str(self.root / "data")])
        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
