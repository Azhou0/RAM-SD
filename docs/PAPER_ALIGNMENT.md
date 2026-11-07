# Paper alignment and implementation choices

Source: the [published ACL 2026 PDF](https://aclanthology.org/2026.acl-long.64.pdf), especially Sections 3.1-3.4 and 4.1.3, and Appendices A.1, C.3, and E. The prompt layouts on PDF pages 13-15 were checked against the rendered pages. Original source-file hashes are recorded in [source_manifest.json](../references/source_manifest.json).

This document distinguishes published specifications, behavior inherited from the authors' original code, and explicit reconstruction choices. It does not certify recovery of the historical experiment environment or scores.

## Changes made to align the implementation

| Component | Original local code | Reconstructed implementation and basis |
|---|---|---|
| Models and decoding | Multiple hardcoded `gpt-4.1-mini` calls, temperature 0.2, and token limits of 1500 or 256 | Shared GPT-4o configuration: temperature 0.1, top_p 1, max_tokens 512, n 1, zero penalties; Section 4.1.3 / A.1 |
| Embeddings | MiniLM by default, with optional OpenAI support | `text-embedding-3-large`, 3072 dimensions, with strict shape validation; Section 4.1.3 |
| Stage order | `select_plan` runs before the plan's similarity agent | Retrieval, individual rationales, similarity analysis, Planner, specialists, synthesis; E.1 / E.2 |
| Rationales | One similarity prompt explains six examples together | One E.1.1 call per retrieved example; training labels and rationales form C_aug |
| Retrieval isolation | Task name can change while `train_iacv1` and a shared index directory remain defaults | Index identity binds the dataset, training file and record hashes, embedding model/dimensions, provider, and insertion order; A.1 |
| Retrieval count | CLI k is not forwarded to the similarity agent | Exactly k examples per label, six in total by default; insufficient subsets fail explicitly; Section 3.1 |
| Planner inputs | Primarily query/context; inferred context is not a complete downstream interface | Similarity analysis and C_aug precede planning; return selected plan, contextual analysis, reasoning, and confidence; E.2 |
| Knowledge-dependent plan | Rhetoric is missing from the local v2 plan | Semantic, Knowledge, Alignment, and Rhetoric core roles; Section 3.2 |
| Agent grounding | Some agents receive only local dependencies | Every specialist can access dialogue context, O_plan, S_analysis, and C_aug; E.3 |
| MUSTARD | Initializes retrieval like other tasks | Stage 1 is disabled, and dialogue context reaches Planner and downstream stages; A.1 |
| Failure handling | API/parsing failures may become `0,0.5`; failed embeddings may become zero vectors | Record failures and withhold aggregate metrics; no default labels or zero-vector padding |
| Dependencies | Requirements include other experiments and do not fully specify this pipeline | Pinned runtime dependencies in uv.lock and an exported requirements.lock.txt |
| Traceability | Timestamped directories without a complete run identity | Recorded calls, code/config/data/index hashes, checked resume, and offline replay |

## Differences between the main text and appendix

### Final decision

Section 3.4 describes an Integrator followed by a separate Judger that returns a sarcasm probability, thresholded at 0.5. Appendix E.4 publishes only an Integrator template with a two-line label and JSON `label,conf,reasoning`; no independent Judger template is provided.

The default `appendix` profile uses the published complete template and follows its label. Its `conf` is confidence in the selected label, not P(y=1). The `maintext` profile adds the separate Judger and threshold rule. Its evidence-summary and Judger prompts are explicitly reconstructed; they are not claimed to be recovered historical prompts.

### Planner validation rules

The E.2 Implementation Note specifies overriding Simple Irony for text longer than 50 words and triggering Knowledge-Dependent for at least three specific entities. Local v2 also contains a short-text/exaggeration override to Simple Irony; that additional unpublished rule is omitted.

The appendix profile counts whitespace-separated words and deduplicates stripped entity names case-insensitively. It applies the long-text rule first, then the three-entity rule. The local code's entity rule applies only to an initial Simple Irony recommendation; this implementation follows the appendix's more general wording. This is a disclosed behavior change. The main-text profile disables these rules by default because the main text does not specify them.

### Planner output schema

The appendix requires entity analysis but does not include the entity list in its example output JSON. The appendix prompt therefore explicitly requests an additional `knowledge_entities` array so rule validation can execute. This is a documented serialization extension.

### Context helper

The main text lists six core specialists. The Expectation and Knowledge prompts in the appendix also reference a Context Agent, without publishing its standalone template. The default appendix profile reuses the original local `_context_agent` prompt as a helper for EV/KD plans. The main-text profile uses O_plan directly without that helper. Configuration and traces distinguish the two choices.

### Simple Irony

The main text permits Rhetoric alone, with optional Semantic/Incongruity support, but gives no activation criteria. The original code always includes Semantic. The appendix profile keeps Semantic plus Rhetoric; the main-text profile defaults to Rhetoric alone. No new activation rule for optional Incongruity support is invented.

### Shared grounding and prompt serialization

Appendix E.3 states that specialists receive O_plan and can reference C_aug and S_analysis. A stable JSON grounding block supplies that information, along with dialogue context, to each role. Example lists expand with k, and PDF layout line breaks are normalized. These are executable templates rather than byte-for-byte recovery of historical API requests. Actual requests and responses are retained for exact replay.

## Repeated runs, API-call counts, and randomness

The paper attributes three-run variation to FAISS ordering, without providing seeds, insertion sequences, tie handling, or historical responses. This implementation records per-label insertion-order seeds while retaining L2 normalization and exact cosine search. It does not randomly reorder ranked neighbors or replace examples. Without ties, insertion-order changes may have no effect. MUSTARD has no retrieval-order variation.

Appendix C.3 does not explain how individual rationale generation, similarity analysis, and the Context helper enter its API-call table. With k=3, the default appendix profile has 11 logical generation calls for SI and 14 for EV/KD before cache reuse. MUSTARD has 4 for SI and 7 for EV/KD. Identical rationale requests can be reused, reducing new requests. The main-text profile has the same totals under its default helper/support settings, with two final calls for summary and judgment. Traces retain every logical call rather than removing published stages to match the table.

`requests_this_invocation` counts successful logical responses, including cache hits. SDK-internal retries and calls that fail before returning are not counted as successful requests.

Low temperature and n=1 do not guarantee identical service outputs. No extra generation seed or JSON-mode API parameter is added to the paper profiles. The model is asked to produce JSON in the prompt; malformed, conflicting, refused, or truncated outputs fail explicitly. Returned model names, fingerprints, response IDs, and usage are recorded. The original dated GPT-4o snapshot is unavailable; a current alias is not a historical snapshot. See the [Chat Completions reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) and [reproducibility guidance](https://developers.openai.com/api/docs/guides/advanced-usage).

## Validated scope

The main method, offline logic, real FAISS retrieval, evaluation, replay, and packaging have been implemented and checked. Datasets are not redistributed or automatically resplit. Manifests explicitly mark original split identity as unverified. Complete main experiments, baseline reruns, ablations, and human explanation evaluation have not been validated.

A separately authorized small live check used OpenAI `gpt-6.1-terra`. Its token-parameter adaptation and reasoning setting exist only in `smoke_terra.json`; the paper profiles remain GPT-4o. See [Validation](VALIDATION.md) for the observed results, which are not evidence of the published benchmark scores.
