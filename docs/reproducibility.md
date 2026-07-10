# Reproducibility Notes

KoDiMARC separates lightweight structural checks from the full GPU workflow. Source code, executable YAML configurations, sample schemas, and manuscript table summaries are included. Raw third-party datasets, pretrained weights, and trained checkpoints must be obtained or produced locally.

## Included Materials

| Material | Location |
| --- | --- |
| Step1 and Step2 implementation | `src/kodimarc/` |
| Data, training, and evaluation commands | `scripts/` |
| Executable experiment settings | `configs/manuscript/` |
| Input schema examples | `data/sample/` |
| Reported Tables 5-9 | `artifacts/results/` |
| Run and table provenance | `docs/EXPERIMENT_MANIFEST.md` |

## External Inputs

Full reproduction requires:

- extracted Korean Wikipedia text under `data/raw/kowiki/extracted/`
- KorNLI files under `data/raw/kornli/`
- AI Malpyeong files under `data/raw/ai_malpyeong/`
- access to `kakaocorp/kanana-1.5-8b-instruct-2505`

AI Malpyeong is a three-label LOGIC task: `forward`, `contrastive`, and `compatible`. The seven labels `ADD`, `CONTRAST`, `CAUSAL`, `EXPLAN`, `CONCESS`, `COND`, and `EXAMPLE` apply only to Step1 KoWiki-derived marker supervision.

## Structural Validation

```bash
python scripts/step1/build_sft_data.py \
  --input-jsonl data/sample/step1_sample.jsonl \
  --train-output /tmp/kodimarc_step1_train.jsonl \
  --valid-output /tmp/kodimarc_step1_valid.jsonl \
  --test-output /tmp/kodimarc_step1_test.jsonl

python scripts/validate_reproducibility.py
```

The first command checks sample preprocessing. The validator parses every YAML file, checks the executable manuscript config schemas, parses sample JSONL and result CSV files, verifies required repository files, and checks literal script/config paths in README commands. Neither command loads model weights.

## Full Workflow

The root README provides ordered commands for:

1. creating the data layout;
2. constructing KoWiki sentence pairs and weak labels;
3. building Step1 SFT splits;
4. training the Step1 generator;
5. preparing Step2 data with top-k markers;
6. training Step2; and
7. evaluating NO, WITH, and WRONG modes.

Every literal script option shown there is exposed by the corresponding CLI. User-specific checkpoint and run paths are derived from the outputs of the preceding commands.

## Numerical Scope

The CSV files under `artifacts/results/` preserve the values and precision shown in the manuscript. Tables 6 and 8 combine task/view values that match different saved runs; a single saved run summary containing every displayed cell was not available. The commercial-LLM evaluation driver for all Table 5 rows is also not included. `docs/EXPERIMENT_MANIFEST.md` records these boundaries and the available run-level evidence.

Numerical reruns remain sensitive to external dataset revisions, model/tokenizer revisions, CUDA and GPU behavior, quantization kernels, random seed, early stopping, and checkpoint selection. The dependency list contains package names because the saved records do not provide a complete package lock for the training environment.
