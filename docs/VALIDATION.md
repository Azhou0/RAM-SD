# Validation record

Date: November 7, 2026. Environment: macOS arm64, CPython 3.12.11. Runtime dependencies are pinned in uv.lock; the direct versions used were openai 2.54.0, numpy 2.5.4, and faiss-cpu 1.15.1.

## Installation and offline tests

The project was reinstalled as a regular wheel. Every installed Python module was compared with `src/ram_sd` and matched its source content.

```text
uv sync --frozen --no-editable --reinstall-package ram-sd-repro
uv run --frozen --no-editable python -m unittest discover -s tests -q
Ran 40 tests ... OK
uv build --wheel --out-dir dist/wheels
Successfully built dist/wheels/ram_sd_repro-0.1.0-py3-none-any.whl
```

Tests cover real FAISS retrieval, all three plans, MUSTARD context propagation without retrieval, gold-label isolation, data/index/response integrity, invalid outputs, failed-run scoring, resume, replay, statistics, format conversion, env loading, and the real OpenAI SDK against mock HTTP.

The release instructions use a regular wheel installation, which was verified on the test host, rather than relying on editable path hooks.

## Live OpenAI integration check

The service endpoint was `https://api.openai.com/v1`. Model discovery confirmed access to `gpt-6.1-terra`, `gpt-4o`, and `text-embedding-3-large`. An initial Terra request using `max_tokens` returned HTTP 400. The separate smoke configuration was adapted to `max_completion_tokens=512` and `reasoning_effort=none`, after which the request succeeded. Temperature 0.1, top_p 1, n 1, and zero penalties were retained. The GPT-4o paper configurations were not replaced.

| Dataset | Position in local test file | Gold label | Prediction | Execution |
|---|---|---:|---:|---|
| IAC-V1 | First record | 0 | 0 | Completed |
| IAC-V1 | Second record | 0 | 0 | Completed |
| MUSTARD | First record | 1 | 0 | Completed; prediction disagreed |

The temporary IAC-V1 index contained only the first eight training examples per label, for 16 examples total, with top-3 retrieval per label. MUSTARD performed no retrieval. Returned model names were `gpt-6.1-terra` and, for embeddings, `text-embedding-3-large`.

The sample checks produced 34 successful new logical requests, including embeddings and excluding separate parameter probes, plus four cache hits. All three pipelines completed; two predictions agreed with the labels and one did not.

**This is an integration check, not a benchmark or a reproduction of the paper's Macro-F1/Accuracy.** Source data, full prompts, responses, and individual predictions remain in ignored local directories. Credentials are excluded from source, index metadata, and response records.

When generated locally, detailed results are available under `outputs/openai_smoke_terra/`, including `summary.json`, per-dataset manifests, predictions, metrics, and traces. Parameter probes were recorded in `outputs/openai_probe*.json`. These runtime artifacts are not distributed in the repository.

## Offline replay

The three samples were replayed into a separate output directory with the same inputs and configuration. Every recorded stage output and final prediction matched the live records. The replay made zero new API requests. Local replay outputs were saved under `outputs/openai_smoke_replayed/`.

## Publication contents

Source releases use an explicit file allowlist. They contain code, configurations, pinned dependencies, documentation, synthetic tests, and source-file hashes. Credentials, datasets, vector indexes, model responses, historical results, virtual environments, and runtime outputs are excluded. The MIT license and paper citation are retained.

Original processed splits, the historical model snapshot, and three-run retrieval ordering remain unavailable. Complete benchmarks, baseline reruns, ablations, and human explanation evaluation have not been performed as part of this validation.
