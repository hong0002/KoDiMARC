# PeerJ Computer Science AI Application Reproducibility Notes

This document mirrors the reviewer-facing reproducibility information in `README.md`. It is intended to make the KoDiMARC public repository easier to audit for the PeerJ Computer Science AI Application requirements.

## PeerJ Computer Science AI Application Reproducibility Checklist
| No. | PeerJ item | Repository response |
| --- | --- | --- |
| 1 | Algorithms and code used to implement them | KoDiMARC is implemented as a two-stage pipeline. The algorithm-to-code map below links each component to concrete scripts and modules. |
| 2 | README file | `README.md` is the primary entry point and this file provides expanded reproducibility notes. |
| 3 | Title | `KoDiMARC: Two-view multi-task learning with discourse markers for robust Korean sentence relation classification` |
| 4 | Description | KoDiMARC first trains a discourse marker generator from weakly supervised KoWiki sentence pairs, then trains a marker-aware multi-task relation classifier over KorNLI and AI Malpyeong data. |
| 5 | Dataset Information | Full datasets are external. The repository includes only sample JSONL schemas under `data/sample/`. |
| 6 | Code Information | Implementation files are under `src/kodimarc/`, workflow scripts are under `scripts/`, and example configs are under `configs/`. |
| 7 | Usage Instructions | The full reproduction workflow below gives ordered commands for Step1, Step2, and evaluation. |
| 8 | Requirements | Python dependencies are listed in `requirements.txt`; full training requires a CUDA-capable GPU environment. |
| 9 | Methodology | Step1 and Step2 methodology is summarized below and expanded in `docs/step1.md` and `docs/step2.md`. |
| 10 | Citations | Users should cite the KoDiMARC manuscript after publication and follow citation requirements for KoWiki, KorNLI, AI Malpyeong, and the base model. |
| 11 | License & Contribution Guidelines | Repository code is MIT licensed; external datasets and model weights retain their own terms. Contribution guidance is provided in `README.md`. |

## Algorithm-to-Code Map
| Component | Purpose | Concrete paths |
| --- | --- | --- |
| Discourse marker lexicon and category mappings | Defines marker categories, label IDs, and relation/category mappings. | `src/kodimarc/common/markers.py` |
| KoWiki sentence-pair construction | Builds adjacent sentence pairs from extracted Korean Wikipedia text. | `scripts/step1/build_sentence_pairs.py` |
| Rule-based discourse marker detection/removal | Detects explicit markers and writes marker-removed hypotheses. | `scripts/step1/detect_and_remove_markers.py`, `src/kodimarc/step1/marker_detection.py` |
| Step1 SFT data construction | Converts weak supervision into response-only SFT splits. | `scripts/step1/build_sft_data.py`, `src/kodimarc/step1/dataset.py` |
| Step1 generator training | Fine-tunes the discourse marker generator. | `scripts/step1/train_step1_generator.py`, `src/kodimarc/step1/modeling.py`, `configs/step1/step1_sft_example.yaml` |
| Step1 top-k marker scoring | Attaches top-k discourse marker candidates and scores. | `scripts/step1/score_topk_markers.py` |
| Final Step2 data construction | Prepares KorNLI/AI Malpyeong splits and attaches Step1 markers. | `scripts/step2/prepare_step2_data.py`, `scripts/step2/build_final_step2_data.py` |
| Step2 multi-task dataset loading | Builds no-marker, predicted-marker, and wrong-marker examples. | `src/kodimarc/step2/dataset.py`, `src/kodimarc/step2/loader.py`, `src/kodimarc/step2/prompt.py` |
| Step2 model and objectives | Implements marker-aware classification, base-delta heads, losses, and memory bank support. | `src/kodimarc/step2/model.py`, `src/kodimarc/step2/losses.py`, `src/kodimarc/step2/memory_bank.py` |
| Step2 training | Runs multi-task training and checkpointing. | `scripts/step2/train_step2.py`, `src/kodimarc/step2/trainer.py`, `src/kodimarc/step2/checkpointing.py` |
| Step2 evaluation | Evaluates NO, WITH, and WRONG marker views. | `scripts/step2/evaluate_step2.py`, `src/kodimarc/step2/evaluate.py`, `src/kodimarc/step2/eval_utils.py`, `src/kodimarc/step2/metrics.py` |
| Config files | Provide editable settings for data paths, models, losses, training, and ablations. | `configs/step1/step1_sft_example.yaml`, `configs/step2/step2_full_example.yaml`, `configs/step2/step2_ablation_example.yaml` |
| Sample JSONL files | Document expected schemas only. | `data/sample/step1_sample.jsonl`, `data/sample/step2_sample.jsonl` |

## Data Availability and External Data Access
The repository includes code, configs, documentation, and small sample JSONL files. It does not include full raw datasets or trained checkpoints.

Included files:
- `src/kodimarc/`: implementation modules.
- `scripts/`: command-line reproduction scripts.
- `configs/`: example YAML configs.
- `docs/`: workflow and reproducibility documentation.
- `data/sample/`: tiny JSONL schema examples.
- `requirements.txt` and `LICENSE`.

External resources required for full reproduction:
- Korean Wikipedia / KoWiki extracted text at `data/raw/kowiki/extracted/`.
- KorNLI TSV files at `data/raw/kornli/`.
- AI Malpyeong logical-relation JSON files at `data/raw/ai_malpyeong/`.
- A compatible Korean instruction-tuned base LLM, such as `kakaocorp/kanana-1.5-8b-instruct-2505`.

The full raw datasets and trained checkpoints are not redistributed because they are subject to external access terms, licensing restrictions, and storage constraints. Users must obtain external datasets and model weights from their original providers, then run the documented scripts locally.

The sample files under `data/sample/` are schema examples only. They are not sufficient to reproduce the full manuscript results.

## Recommended Local Layout
Create the expected local directory layout with:
```bash
python scripts/setup_data_layout.py --root .
```

This prepares:
```text
data/
├── raw/
│   ├── ai_malpyeong/
│   ├── kornli/
│   └── kowiki/extracted/
└── processed/
    ├── kowiki/
    └── multitask_nli/final/
```

## Minimal Smoke Test
Run the included Step1 sample conversion without downloading external datasets:
```bash
python3 scripts/step1/build_sft_data.py \
  --input-jsonl data/sample/step1_sample.jsonl \
  --train-output /tmp/kodimarc_step1_train.jsonl \
  --valid-output /tmp/kodimarc_step1_valid.jsonl \
  --test-output /tmp/kodimarc_step1_test.jsonl
```

Expected output:
```text
[done] train=2, valid=0, test=0
```

This validates that the sample JSONL schema can be read and converted into Step1 SFT format. It does not train a model or reproduce manuscript results.

## Full Reproduction Workflow
Run the workflow in explicit stages after preparing the external datasets and checking local paths in the configs.

### 1. Create the local data layout
```bash
python scripts/setup_data_layout.py --root .
```

### 2. Build Step1 weak supervision data
```bash
python scripts/step1/build_sentence_pairs.py \
  --input-dir data/raw/kowiki/extracted \
  --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py \
  --input-jsonl data/processed/kowiki/wiki_pairs.jsonl \
  --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl
```

### 3. Build Step1 SFT data
```bash
python scripts/step1/build_sft_data.py \
  --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl \
  --train-output data/processed/kowiki/dp_sft_train.jsonl \
  --valid-output data/processed/kowiki/dp_sft_valid.jsonl \
  --test-output data/processed/kowiki/dp_sft_test.jsonl
```

### 4. Train the Step1 generator
```bash
python scripts/step1/train_step1_generator.py \
  --config configs/step1/step1_sft_example.yaml
```

### 5. Build Step2 JSONL files with Step1 top-k markers
```bash
python scripts/step2/build_final_step2_data.py \
  --kornli-dir data/raw/kornli \
  --ai-malpyeong-dir data/raw/ai_malpyeong \
  --step1-model-name-or-path /path/to/step1_checkpoint \
  --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505 \
  --output-dir data/processed/multitask_nli/final \
  --top-k 5 \
  --fallback
```

Expected files:
- `data/processed/multitask_nli/final/train.jsonl`
- `data/processed/multitask_nli/final/dev_nli.jsonl`
- `data/processed/multitask_nli/final/dev_logic.jsonl`
- `data/processed/multitask_nli/final/test_nli.jsonl`
- `data/processed/multitask_nli/final/test_logic.jsonl`
- `data/processed/multitask_nli/final/build_manifest.json`

### 6. Train Step2
```bash
python scripts/step2/train_step2.py \
  --config configs/step2/step2_full_example.yaml
```

### 7. Evaluate Step2
```bash
python scripts/step2/evaluate_step2.py \
  --config /path/to/run_dir/config_used.yaml \
  --exp marker_sensitive
```

Evaluation modes:
- `NO`: no-marker view.
- `WITH`: predicted-marker view using Step1 candidates.
- `WRONG`: corrupted or forbidden-category wrong-marker view when enabled.

## Reproducibility Limits
Exact numerical reproduction depends on:
- External dataset versions and preprocessing state.
- Base model and tokenizer version.
- Local CUDA/GPU environment.
- Quantization and numerical precision.
- Random seed and stochastic training behavior.
- Step1 checkpoint quality and Step2 checkpoint selection.

The public release reconstructs the code path from locally obtained raw data to processed inputs, training outputs, and evaluation artifacts. It does not guarantee bit-for-bit reproduction across hardware or independently trained checkpoints.

## Version for PeerJ Review
- Repository URL: `https://github.com/hong0002/KoDiMARC`
- Release tag: TO BE FILLED BY AUTHOR BEFORE RESUBMISSION
- Commit hash: TO BE FILLED BY AUTHOR BEFORE RESUBMISSION
- Archive DOI/persistent URL: TO BE FILLED BY AUTHOR BEFORE RESUBMISSION
- Archive date: TO BE FILLED BY AUTHOR BEFORE RESUBMISSION

## Practical Caveat
The public repository provides the code path needed for raw-data-to-results reproduction, but the final manuscript submission system must separately verify authorship, funding, affiliations, competing interests, ethics statements, and other manuscript metadata.
