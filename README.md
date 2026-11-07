# RAM-SD

## Retrieval-Augmented Multi-agent Framework for Sarcasm Detection

Code for the ACL 2026 paper by Ziyang Zhou, Ziqi Liu, Yan Wang, Yiming Lin, and Yangbin Chen.

[Paper](https://aclanthology.org/2026.acl-long.64/) | [Dataset sources](docs/DATASETS.md) | [Paper alignment](docs/PAPER_ALIGNMENT.md) | [Validation](docs/VALIDATION.md)

RAM-SD combines balanced example retrieval, rationale generation, a meta-planner, specialized reasoning agents, and evidence synthesis. This repository reconstructs the method from the authors' original local code and the published paper, with pinned dependencies, recorded model responses, resumable evaluation, and offline replay.

**Reproduction status:** the implementation has passed 40 offline tests and a three-sample live integration check. The paper's complete benchmark scores have not been reproduced. The original processed splits, dated model snapshot, and three-run retrieval ordering have not been recovered. Implementation choices are documented in [Paper alignment](docs/PAPER_ALIGNMENT.md).

## Method

1. **Contextual retrieval:** retrieve the top-k sarcastic and top-k non-sarcastic training examples from separate FAISS cosine-similarity indexes; generate a rationale for each example and compare their patterns with the query.
2. **Meta-planning:** use the retrieved evidence to select an expectation-violation, knowledge-dependent, or simple-irony plan and infer contextual guidance.
3. **Multi-view reasoning:** execute the selected Semantic, Expectation, Knowledge, Alignment, Incongruity, and Rhetoric roles as applicable.
4. **Synthesis:** integrate the evidence into a final judgment and explanation.

Retrieval uses only the same dataset's training split. MUSTARD disables retrieval and supplies dialogue context directly to downstream stages.

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Azhou0/RAM-SD.git
cd RAM-SD
uv sync --frozen --no-editable
uv run --frozen --no-editable ram-sd --help
uv run --frozen --no-editable python -m unittest discover -s tests -v
```

`uv.lock` pins runtime dependencies. Non-editable installation avoids reliance on editable path hooks. Reinstall after changing source files; during development, use `PYTHONPATH=src` to test the source tree explicitly. Without uv, create a Python 3.12 environment, install `requirements.lock.txt`, then run `python -m pip install --no-deps .`.

## Datasets

Datasets are distributed by their original providers and are not included here. See [Dataset sources and formats](docs/DATASETS.md) for official links and conversion commands.

```text
data/
  iacv1/train.csv     iacv1/test.csv
  iacv2/train.csv     iacv2/test.csv
  semeval/train.csv   semeval/test.csv
  mustard/test.csv    # train.csv is optional; no retrieval
```

CSV files require `Text,Label`, with `0` for non-sarcastic and `1` for sarcastic. MUSTARD also requires a nonempty `Context` column. An optional `row_id` column supplies stable, unique IDs.

Audit the inputs before running:

```bash
uv run --frozen --no-editable ram-sd audit-data \
  --dataset iacv1 --data-dir data --output outputs/iacv1_data_audit.json
```

Add `--strict-counts` to require the sample counts in Table 1. Matching counts do not establish that a split is the paper's original split. The program never silently removes rows or creates a new split to match those counts.

## API configuration

Set `OPENAI_API_KEY` in your shell, or create a local `.env` using [.env.example](.env.example) and pass `--env-file .env`. On Unix, restrict its permissions with `chmod 600 .env`.

Optional variables are `RAM_SD_API_KEY`, `RAM_SD_BASE_URL`, `RAM_SD_EMBEDDING_API_KEY`, and `RAM_SD_EMBEDDING_BASE_URL`. Env-file loading is explicit, accepts only the documented variables, and performs no shell expansion or execution. Keys are never serialized into indexes or response records.

The paper configurations use GPT-4o, temperature `0.1`, top_p `1`, max_tokens `512`, n `1`, zero penalties, and `text-embedding-3-large` embeddings with `3072` dimensions. The default retrieval setting is `k=3` per label.

## Run an experiment

Build the training-only index, then evaluate:

```bash
uv run --frozen --no-editable ram-sd build-index \
  --config configs/appendix.json --dataset iacv1 --data-dir data \
  --env-file .env --mode live

uv run --frozen --no-editable ram-sd run \
  --config configs/appendix.json --dataset iacv1 --data-dir data \
  --env-file .env --mode live --out outputs/appendix/iacv1
```

For MUSTARD, run `run --dataset mustard` directly and skip `build-index`. Use `--limit 2` for an initial interface check; the result is explicitly marked as a partial evaluation.

`--mode live` uses saved responses when available and sends new requests to the configured model service. `--mode replay` reads saved responses only and fails on a cache miss.

Run all four datasets with the helper script:

```bash
bash scripts/run_main.sh /absolute/path/to/data appendix live outputs/main 0 .env
```

The final two arguments are the retrieval insertion-order seed and the optional env-file path.

## Implementation profiles

| Configuration | Basis | Final decision |
|---|---|---|
| `configs/appendix.json` (default) | Published Appendix E prompts, with the original code's Context helper | Integrator returns `label`, confidence in that label, and an explanation |
| `configs/maintext.json` | Architecture in Sections 3.2-3.4 | A separate reconstructed Judger returns `p_sarcasm`; the label is `1` when the probability is at least `0.5` |

The appendix's `conf` field is not treated as the probability of the positive class. The independent Judger prompt is not published in the paper and is explicitly marked as reconstructed. Planner rules, optional Semantic support, shared grounding, and helper-agent choices are documented in [Paper alignment](docs/PAPER_ALIGNMENT.md). Keep the profiles separate when reporting results.

## Outputs, resume, and replay

Each evaluation writes:

- `manifest.json`: code, configuration, runtime, data and index hashes, selected IDs, and provider endpoints.
- `predictions.csv`: per-sample labels, predictions, confidence, selected plan, and execution status.
- `metrics.json`: binary metrics, coverage, and request statistics.
- `requests.json` and `traces/`: invocation metadata and per-sample reasoning traces.

Raw prompts and responses are stored under `.cache/responses/`. Data, outputs, caches, indexes, and credentials are excluded from version control and source releases.

Use `--resume` with the same output directory to reuse verified successful traces. Changes to code, configuration, inputs, selected rows, runtime, provider, or index prevent resume. For a fresh replay, use `--mode replay --out outputs/replayed` with the original configuration and cache.

Evaluation stops on the first failure by default. `--keep-going` collects additional failures. A run with failed or unattempted samples receives no aggregate score; failures never become default negative predictions. Truncated or malformed responses remain available for diagnosis. If replacing a saved invalid response, use a new cache and output directory and record the new generation.

## Repeated runs and significance

Pass the same `--retrieval-order-seed` to `build-index` and `run`. It changes per-label vector insertion order, without randomizing similarity ranks or replacing retrieved examples. The original three-run ordering procedure is unavailable; new seeds are explicit reproduction conditions, not recovered historical settings. Changing insertion order may have no effect when nearest neighbors have no boundary ties. MUSTARD has no retrieval-order variation.

```bash
uv run --frozen --no-editable ram-sd summarize \
  outputs/run0/iacv1/metrics.json outputs/run1/iacv1/metrics.json \
  --output outputs/summary.json

uv run --frozen --no-editable ram-sd compare \
  --predictions outputs/main/iacv1/predictions.csv \
  --baseline /path/to/baseline_predictions.csv \
  --output outputs/mcnemar.json
```

Aggregation uses sample standard deviation (`ddof=1`) and reports an unweighted mean when all four datasets are present. Partial, failed, or duplicate replay runs cannot serve as repeated full evaluations. The paired test uses the exact two-sided conditional-binomial McNemar test and requires matching `row_id,label,prediction` records in the imported baseline file.

## Small OpenAI integration check

`configs/smoke_terra.json` is a separate integration-test configuration using `gpt-6.1-terra`. It retains temperature `0.1`, top_p `1`, and an output limit of `512`, while using the model's `max_completion_tokens` parameter and `reasoning_effort=none`. The GPT-4o paper configurations remain separate. See [OpenAI parameter compatibility](https://developers.openai.com/api/docs/guides/latest-model).

```bash
uv run --frozen --no-editable python scripts/smoke_openai.py \
  --source-data-dir /path/to/local/legacy/datasets/sarcasm \
  --env-file .env --mode live
```

This checks two IAC-V1 samples and one MUSTARD sample, using only eight IAC-V1 training examples per label for the temporary index. It is not a benchmark. The completed live check ran all three pipelines; two predictions agreed with the labels and one did not. Offline replay reproduced every recorded stage output with zero new API requests. See [Validation](docs/VALIDATION.md).

## Repository layout

```text
src/ram_sd/          Pipeline, retrieval, API recording, evaluation, and CLI
configs/            Paper profiles and a separate Terra smoke-test profile
docs/               Dataset guide, paper alignment, and validation record
scripts/            Batch execution, small integration check, and source packaging
tests/              Offline tests with synthetic fixtures and mock HTTP
references/         Source-file hashes and paper provenance
```

The implemented scope covers the main method, format conversion, input audit, evaluation, replay, and aggregation. Baseline predictions can be imported for paired tests. Full baseline reruns, ablations, human explanation evaluation, and the paper's other supplementary experiments are outside the validated scope.

Create a source archive with `uv run --frozen --no-editable python scripts/build_release.py`.

## Citation

```bibtex
@inproceedings{zhou-etal-2026-ram,
  title = "{RAM-SD}: Retrieval-Augmented Multi-agent framework for Sarcasm Detection",
  author = "Zhou, Ziyang and Liu, Ziqi and Wang, Yan and Lin, Yiming and Chen, Yangbin",
  booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2026",
  pages = "1434--1448",
  url = "https://aclanthology.org/2026.acl-long.64/"
}
```

## License

The implementation is distributed under the [MIT License](LICENSE). Dataset licenses and provider terms apply separately.
