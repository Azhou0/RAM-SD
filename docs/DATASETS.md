# Dataset sources and input formats

Datasets, historical predictions, and vector indexes are not redistributed in this repository. The following are the original providers' release locations, checked on November 7, 2026.

| Benchmark | Official page | Author-maintained repository |
|---|---|---|
| IAC-V1 sarcasm subset | [UCSC Sarcasm Corpus V1](https://nlds.soe.ucsc.edu/node/32) | [slukin/nlds-sarcasm-corpus-v1](https://github.com/slukin/nlds-sarcasm-corpus-v1) |
| IAC-V2 sarcasm subset | [UCSC Sarcasm Corpus V2](https://nlds.soe.ucsc.edu/node/43) | [soraby/sarcasm2](https://github.com/soraby/sarcasm2) |
| MUSTARD | [Official MUStARD repository](https://github.com/soujanyaporia/MUStARD) | [Text and dialogue JSON](https://github.com/soujanyaporia/MUStARD/blob/master/data/sarcasm_data.json) |
| SemEval-2018 Task 3 A | [Task organizers' repository](https://github.com/Cyvhee/SemEval2018-Task3) | [Official datasets directory](https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets) |

Some legacy UCSC pages redirect or are intermittently unavailable; the linked author repositories provide an alternative. IAC-V2 includes GEN, HYP, and RQ subsets: combining them does not recover the paper's processed benchmark. SemEval uses Task A binary labels; Task B labels are not interchangeable with 0/1. Follow each provider's data-use terms.

## Split provenance

**Original raw corpora are not the same as the processed splits used in the paper.** Available local records do not establish the filtering rules for the three training sets or identify which of five candidate MUSTARD folds was used. This implementation does not select a fold, invent a split seed, or truncate files to match a published count. When supplying another valid split, record its source revision, preprocessing, selection rules, and IDs.

Table 1 in the [RAM-SD paper](https://aclanthology.org/2026.acl-long.64.pdf) reports:

| Dataset | Train | Test |
|---|---:|---:|
| IAC-V1 | 1,595 | 320 |
| IAC-V2 | 5,216 | 1,042 |
| SemEval | 3,634 | 784 |
| MUSTARD | 552 | 138 |

## Unified CSV schema

Use UTF-8 CSV files with these columns:

| Column | Required | Meaning |
|---|---|---|
| `Text` | Yes | Input utterance, preserved as supplied |
| `Label` | Yes | `0` = non-sarcastic; `1` = sarcastic |
| `Context` | MUSTARD only | Nonempty dialogue context; quoted multiline fields are supported |
| `row_id` | Optional | Unique stable sample ID |

If no ID is supplied, the loader derives one from row position and content. It does not silently lowercase text, drop missing values, or deduplicate inputs. Case folding and whitespace normalization are used separately for overlap auditing.

```text
data/iacv1/train.csv
data/iacv1/test.csv
data/iacv2/train.csv
data/iacv2/test.csv
data/semeval/train.csv
data/semeval/test.csv
data/mustard/test.csv
```

MUSTARD's `train.csv` is optional and used only for split auditing. No retrieval index is built or queried for MUSTARD.

## Format conversion

The conversion commands map columns and labels for an already chosen split. They do not create a new split.

For a SemEval Task A TSV with the corresponding columns:

```bash
uv run --frozen --no-editable ram-sd prepare-csv \
  --input /path/to/a_chosen_split.tsv --output data/semeval/test.csv \
  --delimiter tab --text-column "Tweet text" --label-column "Label" --id-column "Tweet index"
```

For IAC files using `sarc/notsarc`, add `--positive-label sarc --negative-label notsarc`. Use the actual downloaded column names. Any subset filtering and split-ID selection must be documented explicitly.

The official MUStARD JSON stores `utterance`, `context`, and `sarcasm` under sample IDs. Supply a JSON array of the IDs in your explicitly selected split:

```bash
uv run --frozen --no-editable ram-sd prepare-mustard \
  --input /path/to/sarcasm_data.json --ids /path/to/test_ids.json \
  --output data/mustard/test.csv
```

The converter preserves dialogue order and joins context utterances with newlines, without adding speaker names. The paper does not specify this serialization detail, so it is an explicit implementation convention. The complete 690-record source file must not be described as the paper's 138-record test split.

## Input audit

```bash
uv run --frozen --no-editable ram-sd audit-data \
  --dataset iacv1 --data-dir data --output outputs/iacv1_data_audit.json
```

The report records counts, label distributions, SHA-256 hashes, normalized duplicates, and train/test overlap. Count mismatches are reported; `--strict-counts` makes them fatal. Even when counts match, `split_identity` remains unverified against the original paper split.
