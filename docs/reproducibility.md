# KoDiMARC Reproducibility Notes

This document mirrors the reviewer-facing reproducibility information in `README.md` and provides a compact audit trail for the PeerJ Computer Science AI Application requirements.

## PeerJ Computer Science AI Application Reproducibility Checklist
| No. | PeerJ item | Repository response |
| --- | --- | --- |
| 1 | Algorithms and code used to implement them | KoDiMARC is implemented as a two-stage pipeline. The algorithm-to-code map below links each component to concrete scripts and modules. |
| 2 | README file | `README.md` is the primary entry point. This document, `docs/MANUSCRIPT_RUN_MANIFEST.md`, and `docs/results/README.md` provide supporting reproducibility details. |
| 3 | Title | `KoDiMARC: Two-view multi-task learning with discourse markers for robust Korean sentence relation classification` |
| 4 | Description | Step1 constructs weakly supervised KoWiki discourse-marker data and trains a marker generator. Step2 trains a marker-aware multi-task classifier over KorNLI and AI Malpyeong sentence pairs. |
| 5 | Dataset Information | Full datasets are external. The repository includes source code, configs, documentation, and small JSONL schema examples under `data/sample/`. |
| 6 | Code Information | Implementation files are under `src/kodimarc/`, workflow scripts are under `scripts/`, public configs are under `configs/step1/` and `configs/step2/`, and copied local run configs are under `configs/peerj_review/`. |
| 7 | Usage Instructions | The full workflow below gives ordered commands for data layout creation, Step1, Step2, and evaluation. |
| 8 | Requirements | Python dependencies are listed in `requirements.txt`; full training requires a CUDA-capable GPU environment. |
| 9 | Methodology | Step1 and Step2 methodology is summarized below and expanded in `docs/step1.md` and `docs/step2.md`. |
| 10 | Citations | Users should cite the KoDiMARC manuscript after publication and follow citation requirements for KoWiki, KorNLI, AI Malpyeong, and the selected base model. |
| 11 | License & Contribution Guidelines | Repository code is MIT licensed; external datasets and model weights retain their own terms. Contribution guidance is provided in `README.md`. |

## Algorithm-to-Code Map
| Component | Purpose | Concrete paths |
| --- | --- | --- |
| Discourse marker lexicon and category mappings | Defines marker categories, label IDs, and relation/category mappings. | `src/kodimarc/common/markers.py` |
| KoWiki sentence-pair construction | Builds adjacent sentence pairs from extracted Korean Wikipedia text. | `scripts/step1/build_sentence_pairs.py` |
| Rule-based discourse marker detection/removal | Detects explicit markers and writes marker-removed hypotheses. | `scripts/step1/detect_and_remove_markers.py`, `src/kodimarc/step1/marker_detection.py` |
| Step1 SFT data construction | Converts weak supervision into response-only SFT splits. | `scripts/step1/build_sft_data.py`, `src/kodimarc/step1/dataset.py` |
| Step1 generator training | Fine-tunes the discourse marker generator. | `scripts/step1/train_step1_generator.py`, `src/kodimarc/step1/modeling.py`, `configs/step1/step1_sft_example.yaml`, `configs/peerj_review/step1_sft_local_artifact.yaml` |
| Step1 top-k marker scoring | Attaches top-k discourse marker candidates and scores. | `scripts/step1/score_topk_markers.py` |
| Final Step2 data construction | Prepares KorNLI/AI Malpyeong splits and attaches Step1 markers. | `scripts/step2/prepare_step2_data.py`, `scripts/step2/build_final_step2_data.py` |
| Step2 multi-task dataset loading | Builds no-marker, predicted-marker, and wrong-marker examples. | `src/kodimarc/step2/dataset.py`, `src/kodimarc/step2/loader.py`, `src/kodimarc/step2/prompt.py` |
| Step2 model and objectives | Implements marker-aware classification, base-delta heads, losses, and memory bank support. | `src/kodimarc/step2/model.py`, `src/kodimarc/step2/losses.py`, `src/kodimarc/step2/memory_bank.py` |
| Step2 training | Runs multi-task training and checkpointing. | `scripts/step2/train_step2.py`, `src/kodimarc/step2/trainer.py`, `src/kodimarc/step2/checkpointing.py` |
| Step2 evaluation | Evaluates NO, WITH, and WRONG marker views. | `scripts/step2/evaluate_step2.py`, `src/kodimarc/step2/evaluate.py`, `src/kodimarc/step2/eval_utils.py`, `src/kodimarc/step2/metrics.py` |
| Config files | Provide editable settings for data paths, models, losses, training, and ablations. | `configs/step1/`, `configs/step2/`, `configs/peerj_review/` |
| Sample JSONL files | Document expected row schemas. | `data/sample/step1_sample.jsonl`, `data/sample/step2_sample.jsonl` |

## Data Availability and External Data Access
This repository includes code, configs, documentation, and small sample JSONL files. It does not include full raw datasets or trained checkpoints.

Included files:
- `src/kodimarc/`: implementation modules.
- `scripts/`: command-line reproduction scripts.
- `configs/step1/` and `configs/step2/`: public example YAML configs.
- `configs/peerj_review/`: verified local run configs copied from searched experiment artifacts.
- `docs/`: workflow, manifest, results, and reproducibility documentation.
- `data/sample/`: toy JSONL schema examples.
- `requirements.txt` and `LICENSE`.

External resources required for full reproduction:
- Korean Wikipedia / KoWiki extracted text at `data/raw/kowiki/extracted/`.
- KorNLI TSV files at `data/raw/kornli/`.
- AI Malpyeong logical-relation files at `data/raw/ai_malpyeong/`.
- A compatible Korean instruction-tuned base LLM, such as `kakaocorp/kanana-1.5-8b-instruct-2505`.

The full raw datasets and trained checkpoints are not redistributed because they are subject to external provider terms, access restrictions, licensing constraints, and storage constraints. Users must obtain external datasets and model weights from their original providers, then run the documented scripts locally.

The sample files under `data/sample/` are schema examples. They validate input fields and conversion paths, but they are too small to produce manuscript-scale metrics.

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

## Manuscript Experiment Environment
The following values were verified from the searched local artifacts and machine information summarized in `docs/MANUSCRIPT_RUN_MANIFEST.md`.

| Item | Value |
| --- | --- |
| Operating system | Linux `5.15.0-139-generic`, `x86_64 GNU/Linux` |
| CPU | Intel Core i9-10900X @ 3.70 GHz, 10 cores / 20 threads |
| GPU | 4 x NVIDIA GeForce RTX 3090, 24 GiB VRAM each |
| NVIDIA driver | `535.183.01` |
| System memory | 188 GiB RAM |
| Step2 backbone | `kakaocorp/kanana-1.5-8b-instruct-2505` in verified Kanana configs |
| Step2 precision / quantization | `bf16` mixed precision and `8bit` quantization |
| Step2 LoRA | rank `64`, alpha `128`, dropout `0.0` |
| Main seeds found | Step1 SFT local artifact: `42`; Step2 full result run: `42`; Step2 marker-sensitive run: `43`; no-MREL and no-SupCon ablations: `42` |

The searched experiment artifacts did not contain a complete `pip freeze`, conda environment export, or immutable package-lock file. Public dependencies are therefore documented in `requirements.txt`, while copied configs preserve the model, precision, quantization, LoRA, training, and seed settings available in saved artifacts.

## Methodology
Step1:
1. Build adjacent sentence pairs from extracted KoWiki text.
2. Detect explicit Korean discourse markers using the rule-based lexicon.
3. Remove detected marker spans from the second sentence.
4. Convert labeled pairs into response-only SFT examples.
5. Fine-tune a Korean language model to generate the missing discourse marker.
6. Score downstream sentence pairs with top-k marker candidates and scores.

Step2:
1. Convert KorNLI and AI Malpyeong into common JSONL splits.
2. Attach Step1 top-k marker candidates to each sentence pair.
3. Train a multi-task classifier over NLI and LOGIC examples.
4. Use no-marker, predicted-marker, and wrong-marker views for training and diagnostics.
5. Evaluate NO, WITH, and WRONG modes using accuracy, macro precision, macro recall, macro F1, confusion matrices, transition summaries, and marker-category reports.

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

The local Step1 artifact config found in searched experiment folders is preserved as `configs/peerj_review/step1_sft_local_artifact.yaml`.

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

Verified local artifact configs are preserved under `configs/peerj_review/`.

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

## Smoke Test
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

The smoke test validates the JSONL schema and executable data-conversion path using the included toy files. Full manuscript metrics require the external datasets, the manuscript configs, and the full Step1/Step2 workflow described above.

## Reproducing the Reported Tables
Use `docs/results/README.md` as the compact result index and `docs/MANUSCRIPT_RUN_MANIFEST.md` as the detailed artifact manifest.

| Manuscript table type | Reproduction artifact or command |
| --- | --- |
| Step1 marker prediction table | `/home/jihong/Multi-Task_NLI/outputs/marker_only_top1_logic/20260525_170242/summary.csv`, summarized in `docs/results/README.md`. |
| Main end-to-end AI Malpyeong and KorNLI table | `configs/peerj_review/step2_marker_sensitive_run_20260406_seed43.yaml` and the saved metrics indexed in `docs/results/README.md`. |
| Ablation table | `configs/peerj_review/step2_ablation_no_mrel_run_20260413.yaml`, `configs/peerj_review/step2_ablation_no_supcon_run_20260413.yaml`, and additional local summaries listed in `docs/MANUSCRIPT_RUN_MANIFEST.md`. |
| NO / WITH / WRONG diagnostics | Saved `metrics.json`, transition summaries, and marker-category reports in the run directories listed in `docs/results/README.md`. |
| Marker distribution table | `/data1/jihong/multi-task_NLI3/outputs/marker_distribution/label_marker_counts_summary.json`, summarized in `docs/results/README.md`. |

## Reproducibility Scope
This repository reconstructs the raw-data-to-results code path when external datasets and local model weights are available. Exact numerical values can vary with:
- External dataset version and preprocessing state.
- Base model and tokenizer revision.
- Step1 checkpoint used for top-k marker generation.
- GPU hardware, CUDA runtime, quantization kernels, bf16 behavior, and random seed.
- Early stopping and checkpoint selection.

Saved local artifacts show the manuscript-facing seeds, model identifiers, precision, quantization, LoRA settings, and result files available for this release.

## Code Availability
- Repository: `https://github.com/hong0002/KoDiMARC`
- License: MIT License

The repository contains the source code, configs, sample schemas, and reproduction instructions used for review. No release archive DOI was found in the searched local artifacts, so no DOI is stated here.

## Citation
If this repository is used in research, cite the accompanying KoDiMARC manuscript after publication. Users must also follow the citation and license requirements of Korean Wikipedia / KoWiki, KorNLI, AI Malpyeong, and the selected base model.

## License
This repository is released under the MIT License. See `LICENSE` for details. External datasets and pretrained model weights are governed by their original providers' access terms and licenses.

## Contribution Guidelines
This repository is maintained as a reproducibility release for the accompanying manuscript. Contributions that improve documentation, fix reproducibility bugs, or clarify setup instructions are welcome.

Useful issue reports include the command, config path, dataset split, observed behavior, expected behavior, Python version, package versions, GPU model, and CUDA/driver information.
