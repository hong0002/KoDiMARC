# KoDiMARC

## Title
KoDiMARC: Two-view multi-task learning with discourse markers for robust Korean sentence relation classification

## Description
KoDiMARC is a two-stage framework for Korean sentence-pair relation classification. Step1 constructs weakly supervised Korean discourse-marker data from Korean Wikipedia / KoWiki sentence pairs, trains a discourse marker generator, and produces top-k discourse marker candidates. Step2 trains a marker-aware multi-task classifier using KorNLI for natural language inference labels and AI Malpyeong for Korean logical-relation labels.

This repository is the public reproducibility release for the PeerJ Computer Science submission. It contains source code, reproducibility scripts, configuration files, documentation, and small JSONL schema examples. Full raw datasets and trained checkpoints are not redistributed because they must be obtained from their original providers and may have access, licensing, and storage constraints.

## PeerJ Computer Science AI Application Reproducibility Checklist
| No. | PeerJ item | Repository response |
| --- | --- | --- |
| 1 | Algorithms and code used to implement them | The two-stage algorithm is summarized in `Methodology`; implementation files are mapped in `Algorithm-to-Code Map`. |
| 2 | README file | This file is the reviewer-facing entry point. Expanded notes are in `docs/reproducibility.md`, `docs/MANUSCRIPT_RUN_MANIFEST.md`, and `docs/results/README.md`. |
| 3 | Title | The manuscript title is stated in `Title`. |
| 4 | Description | `Description` summarizes the public reproducibility release and the Step1/Step2 design. |
| 5 | Dataset Information | `Dataset Information` and `Data Availability and External Data Access` describe included files, external datasets, local paths, and redistribution limits. |
| 6 | Code Information | `Code Information` and `Algorithm-to-Code Map` identify source files, scripts, configs, metrics utilities, and sample schemas. |
| 7 | Usage Instructions | `Full Reproduction Workflow` lists ordered commands for layout creation, Step1, Step2, evaluation, and locating metrics. |
| 8 | Requirements | `Requirements` and `Manuscript Experiment Environment` document dependencies, hardware, precision, quantization, and seed settings found in local artifacts. |
| 9 | Methodology | `Methodology` describes weak-supervised marker generation, two-view marker-aware classification, and NO/WITH/WRONG evaluation. |
| 10 | Citations | `Citation` identifies the manuscript and external resources that require citation under their own terms. |
| 11 | License & Contribution Guidelines | `License` and `Contribution Guidelines` state the MIT license for repository code and clarify that external datasets/model weights retain their own terms. |

## Dataset Information
The full KoDiMARC workflow uses four external resources:
- Korean Wikipedia / KoWiki dump or extracted Korean Wikipedia text for Step1 weak supervision.
- KorNLI TSV files for NLI supervision with `entailment`, `neutral`, and `contradiction` labels.
- AI Malpyeong Korean sentence-pair logical-relation data for LOGIC supervision, including labels such as `순접(forward)`, `역접(contrastive)`, `양립(compatible)`, `인과(causal)`, `양보(concessive)`, `조건(conditional)`, `설명(explanatory)`, and `예시(example)`.
- A Korean base LLM or encoder backbone obtained under its provider terms. The manuscript-facing Kanana runs use `kakaocorp/kanana-1.5-8b-instruct-2505`.

Files included in this repository:
- Source code under `src/kodimarc/`.
- Reproducibility scripts under `scripts/`.
- Public example configs under `configs/step1/` and `configs/step2/`.
- Verified local run configs and the manuscript-reconstructed Step1 Kanana config under `configs/peerj_review/`.
- Workflow documentation under `docs/`.
- Toy JSONL schema examples under `data/sample/`.
- `requirements.txt` and `LICENSE`.

The `data/sample/` files are toy schema examples:
- `data/sample/step1_sample.jsonl` shows Step1 weak-supervision rows.
- `data/sample/step2_sample.jsonl` shows Step2 rows with top-k marker fields.

## Data Availability and External Data Access
The repository does not redistribute full raw datasets or trained checkpoints. The raw datasets must be obtained from their original providers and placed under the expected local paths before running the full workflow.

| Resource | Use in KoDiMARC | Expected local path | Access notes |
| --- | --- | --- | --- |
| Korean Wikipedia / KoWiki | Step1 adjacent sentence-pair construction and weak marker supervision | `data/raw/kowiki/extracted/` | Obtain a Korean Wikipedia dump from [Wikimedia Dumps](https://dumps.wikimedia.org/kowiki/) and extract plain text locally. |
| KorNLI | Step2 NLI supervision and NLI evaluation | `data/raw/kornli/` | KorNLI is distributed through the [KorNLU dataset repository](https://github.com/kakaobrain/KorNLUDatasets). |
| AI Malpyeong logical-relation data | Step2 LOGIC supervision and LOGIC evaluation | `data/raw/ai_malpyeong/` | Obtain the Korean sentence-pair logical-relation task data through the National Institute of Korean Language [AI Malpyeong platform](https://kli.korean.go.kr/benchmark/home.do). |
| `kakaocorp/kanana-1.5-8b-instruct-2505` or compatible model | Step1 marker generation and Step2 backbone | Local Hugging Face cache or model path | The manuscript-facing Kanana configs use the [Hugging Face model identifier](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505). |

The sample JSONL files validate schemas and executable conversion paths. Full manuscript metrics require the external datasets, trained Step1/Step2 checkpoints, and the full workflow below.

Recommended local layout:
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

## Code Information
Repository structure:
```text
KoDiMARC/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── peerj_review/
│   ├── step1/
│   └── step2/
├── data/sample/
├── docs/
├── scripts/
│   ├── step1/
│   └── step2/
└── src/kodimarc/
    ├── common/
    ├── step1/
    └── step2/
```

Important documentation:
- `docs/step1.md`: detailed Step1 workflow.
- `docs/step2.md`: detailed Step2 workflow.
- `docs/reproducibility.md`: expanded PeerJ reproducibility notes.
- `docs/MANUSCRIPT_RUN_MANIFEST.md`: searched local artifact folders, verified configs, result files, and stored environment evidence.
- `docs/results/README.md`: compact index of manuscript table values and local result artifacts used for cross-checking.

## Algorithm-to-Code Map
| Manuscript component | Purpose | Concrete repository paths |
| --- | --- | --- |
| Discourse marker lexicon and category mapping | Defines Korean discourse marker inventory, categories, label mappings, and wrong-category mappings. | `src/kodimarc/common/markers.py` |
| KoWiki sentence-pair construction | Builds adjacent sentence pairs from extracted KoWiki text. | `scripts/step1/build_sentence_pairs.py` |
| Rule-based marker detection/removal | Detects discourse markers and writes marker-removed hypotheses. | `scripts/step1/detect_and_remove_markers.py`, `src/kodimarc/step1/marker_detection.py` |
| Step1 SFT data construction | Converts weakly supervised marker rows into response-only SFT JSONL splits. | `scripts/step1/build_sft_data.py`, `src/kodimarc/step1/dataset.py` |
| Step1 generator training | Fine-tunes the discourse marker generator. | `scripts/step1/train_step1_generator.py`, `src/kodimarc/step1/modeling.py`, `configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml`, `configs/step1/step1_sft_example.yaml`, `configs/peerj_review/step1_sft_local_artifact.yaml` |
| Step1 top-k scoring / Step2 data construction | Attaches top-k marker candidates to KorNLI and AI Malpyeong examples. | `scripts/step1/score_topk_markers.py`, `scripts/step2/build_final_step2_data.py` |
| KorNLI / AI Malpyeong preparation | Converts external raw datasets into common Step2 JSONL splits. | `scripts/step2/prepare_step2_data.py` |
| Step2 dataset loading | Builds NLI/LOGIC minibatches and no-marker, with-marker, and wrong-marker views. | `src/kodimarc/step2/dataset.py`, `src/kodimarc/step2/loader.py`, `src/kodimarc/step2/prompt.py` |
| Step2 model | Implements the marker-aware encoder classifier. | `src/kodimarc/step2/model.py` |
| Base-delta head | Separates base no-marker prediction from marker-induced delta prediction when enabled. | `src/kodimarc/step2/model.py`, `configs/step2/step2_full_example.yaml`, `configs/peerj_review/step2_marker_sensitive_run_20260406_seed43.yaml` |
| Marker dropout | Regularizes predicted-marker views during training. | `src/kodimarc/step2/dataset.py`, `src/kodimarc/step2/trainer.py` |
| Marker corruption / wrong-marker construction | Samples corrupted or forbidden-category markers for wrong-marker views. | `src/kodimarc/step2/dataset.py`, `src/kodimarc/common/markers.py` |
| MREL | Marker-relation compatibility supervision. | `src/kodimarc/step2/model.py`, `src/kodimarc/step2/trainer.py` |
| Supervised contrastive learning | Adds supervised contrastive loss with optional memory queues. | `src/kodimarc/step2/losses.py`, `src/kodimarc/step2/memory_bank.py`, `src/kodimarc/step2/trainer.py` |
| WITH-WRONG margin objective | Encourages predicted-marker views to outperform wrong-marker views. | `src/kodimarc/step2/losses.py`, `src/kodimarc/step2/trainer.py` |
| Step2 training | Trains and checkpoints the multi-task classifier. | `scripts/step2/train_step2.py`, `src/kodimarc/step2/trainer.py`, `src/kodimarc/step2/checkpointing.py` |
| Evaluation under NO / WITH / WRONG | Evaluates saved checkpoints under no-marker, predicted-marker, and wrong-marker modes. | `scripts/step2/evaluate_step2.py`, `src/kodimarc/step2/evaluate.py`, `src/kodimarc/step2/eval_utils.py` |
| Metrics calculation | Computes accuracy, macro precision, macro recall, macro F1, confusion matrices, transition summaries, and marker-category reports. | `src/kodimarc/step2/metrics.py`, `src/kodimarc/step2/eval_utils.py` |
| YAML configs | Stores public examples, copied manuscript-facing local artifact configs, and the manuscript-reconstructed Step1 Kanana config. | `configs/step1/`, `configs/step2/`, `configs/peerj_review/` |
| Sample JSONL schemas | Shows expected Step1 and Step2 row formats. | `data/sample/step1_sample.jsonl`, `data/sample/step2_sample.jsonl` |

## Requirements
Install the public dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists:
```text
torch
transformers
peft
bitsandbytes
accelerate
datasets
PyYAML
tqdm
ujson
kss
sentencepiece
tokenizers
numpy
pandas
scikit-learn
scipy
unsloth
```

`requirements.txt` contains the repository dependencies used by the public scripts and modules. Full Step1/Step2 training uses large language models and requires a CUDA-capable GPU environment. The manuscript-facing configs use 8-bit quantization and LoRA for Step1, and bf16 mixed precision, 8-bit quantization, and LoRA for the Kanana 8B Step2 backbone.

## Manuscript Experiment Environment
The following environment values were verified from the local machine and searched experiment artifacts described in `docs/MANUSCRIPT_RUN_MANIFEST.md`.

| Item | Value |
| --- | --- |
| Operating system | Linux `5.15.0-139-generic`, `x86_64 GNU/Linux` |
| CPU | Intel Core i9-10900X @ 3.70 GHz, 10 cores / 20 threads |
| GPU | 4 x NVIDIA GeForce RTX 3090, 24 GiB VRAM each |
| NVIDIA driver | `535.183.01` |
| System memory | 188 GiB RAM |
| Python | `3.12.4` in the released validation environment; searched local run artifacts did not store a separate interpreter version |
| CUDA runtime version | Not stored in the released logs; NVIDIA driver `535.183.01` was used |
| Step1 manuscript generator | `kakaocorp/kanana-1.5-8b-instruct-2505`, documented in `configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml` |
| Step2 backbone | `kakaocorp/kanana-1.5-8b-instruct-2505` in the manuscript-facing Kanana configs |
| Precision / quantization | `bf16` mixed precision and `8bit` quantization in verified Step2 configs |
| LoRA | rank `64`, alpha `128`, dropout `0.0` in verified Kanana Step2 configs |
| Main seeds found | Step1 manuscript config: `42`; auxiliary EXAONE Step1 local artifact: `42`; Step2 full result run: `42`; Step2 marker-sensitive run: `43`; no-MREL and no-SupCon ablations: `42` |

The searched experiment folders did not contain a `pip freeze`, conda environment export, or equivalent package-lock file. The public dependency list is therefore provided by `requirements.txt`, and the copied configs preserve the model, precision, quantization, LoRA, training, and seed settings available in the saved artifacts. All files under `configs/peerj_review/*.yaml` are valid YAML and can be parsed with PyYAML.

## Methodology
KoDiMARC contains two connected stages.

### Step1: Discourse Marker Generator
1. Build adjacent sentence pairs from extracted KoWiki text.
2. Detect explicit Korean discourse markers using the rule-based lexicon.
3. Remove detected marker spans from the second sentence.
4. Convert labeled pairs into response-only SFT examples.
5. Fine-tune a Korean language model to generate the missing discourse marker.
6. Score downstream sentence pairs with top-k marker candidates and scores.

### Step2: Marker-aware Multi-task Relation Classifier
1. Convert KorNLI and AI Malpyeong into common JSONL splits.
2. Attach Step1 top-k marker candidates to each pair.
3. Train a multi-task classifier over NLI and LOGIC examples.
4. Use no-marker, predicted-marker, and wrong-marker views for training and diagnostics.
5. Evaluate NO, WITH, and WRONG modes using accuracy, macro precision, macro recall, macro F1, confusion matrices, transition summaries, and marker-category reports.

## Full Reproduction Workflow
Commands below use scripts that exist in this repository. Replace `/path/to/step1_checkpoint` with a locally trained Step1 checkpoint or adapter.

### 1. Create the data layout
```bash
python scripts/setup_data_layout.py --root .
```

### 2. Build Step1 weak-supervision data
```bash
python scripts/step1/build_sentence_pairs.py \
  --input-dir data/raw/kowiki/extracted \
  --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py \
  --input-jsonl data/processed/kowiki/wiki_pairs.jsonl \
  --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl
```

### 3. Build Step1 SFT JSONL data
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
  --config configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml
```

The manuscript Step1 generator config is `configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml`. The exact original Kanana Step1 YAML was not found in the searched local artifacts, so this file is reconstructed from manuscript PDF Table 10 and verified local scripts/artifacts. The EXAONE file `configs/peerj_review/step1_sft_local_artifact.yaml` is retained as an auxiliary smaller Step1 local artifact, not as the manuscript Step1 generator used for final Step2 results.

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

Expected outputs:
- `data/processed/multitask_nli/final/train.jsonl`
- `data/processed/multitask_nli/final/dev_nli.jsonl`
- `data/processed/multitask_nli/final/dev_logic.jsonl`
- `data/processed/multitask_nli/final/test_nli.jsonl`
- `data/processed/multitask_nli/final/test_logic.jsonl`
- `data/processed/multitask_nli/final/build_manifest.json`

### 6. Train Step2
Use the public normalized config:
```bash
python scripts/step2/train_step2.py \
  --config configs/step2/step2_full_example.yaml
```

Verified local artifact configs are also preserved for review:
```bash
python scripts/step2/train_step2.py \
  --config configs/peerj_review/step2_marker_sensitive_run_20260406_seed43.yaml
```

The copied `configs/peerj_review/*.yaml` files preserve saved local run settings or manuscript-reconstructed settings. They are valid YAML parseable with PyYAML, and their paths and source artifacts are documented in `docs/MANUSCRIPT_RUN_MANIFEST.md`.

### 7. Evaluate Step2 under NO / WITH / WRONG modes
```bash
python scripts/step2/evaluate_step2.py \
  --config /path/to/run_dir/config_used.yaml \
  --exp marker_sensitive
```

Output metrics are written under the saved run directory. Typical files include:
- `summary.csv`
- `metrics.json`
- `test/summary.csv`
- `test/metrics.json`
- `test/test_nli_transition_summary.json`
- `test/test_logic_transition_summary.json`
- `test/test_*_marker_category_report.csv`

## Smoke Test
Run a minimal no-GPU schema and conversion check using the included toy Step1 data:
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
The searched local experiment folders contain saved configs, summaries, metrics, and diagnostic reports for the manuscript-facing runs. The compact index is in `docs/results/README.md`; the detailed artifact manifest is in `docs/MANUSCRIPT_RUN_MANIFEST.md`.

The numerical summaries in this repository follow the manuscript tables. Some local result files may store additional decimal places or intermediate diagnostic outputs; the manuscript-facing tables are rounded as reported in the PDF.

| Manuscript table | Headline values |
| --- | --- |
| Table 5 Step1 generator | Kanana-1.5-8B-instruct-2505: marker accuracy `0.480`, label accuracy `0.772`, marker F1 `0.147`, label F1 `0.502`. |
| Table 6 KoDiMARC WITH | AI-M Acc `0.874`, AI-M Macro-F1 `0.674`, KorNLI Acc `0.873`, KorNLI Macro-F1 `0.873`. |
| Table 7 full KoDiMARC | KorNLI Acc `0.8730`, KorNLI F1 `0.8730`, AI-M Acc `0.8736`, AI-M F1 `0.6743`. |
| Table 8 WRONG diagnostic | AI-M Acc `0.862`, AI-M Macro-F1 `0.647`, KorNLI Acc `0.861`, KorNLI Macro-F1 `0.861`. |
| Table 9 marker distribution | AI Malpyeong Forward `N=125`, Compatible `N=12`, Contrastive `N=132`; KorNLI Contradiction `N=1652`, Entailment `N=1651`, Neutral `N=1651`. |

| Manuscript table type | Reproduction artifact or command |
| --- | --- |
| Step1 marker prediction table | Inspect manuscript Table 5 values in `docs/results/README.md` and the manuscript Step1 config `configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml`. |
| Main end-to-end AI Malpyeong and KorNLI table | Inspect manuscript Table 6 values in `docs/results/README.md`; run Step2 training/evaluation with `configs/peerj_review/step2_marker_sensitive_run_20260406_seed43.yaml` for the corresponding local code path. |
| Ablation table | Inspect manuscript Table 7 values in `docs/results/README.md`; copied ablation configs are `configs/peerj_review/step2_ablation_no_mrel_run_20260413.yaml` and `configs/peerj_review/step2_ablation_no_supcon_run_20260413.yaml`. |
| NO / WITH / WRONG diagnostic table | Inspect manuscript Table 8 values in `docs/results/README.md`; local diagnostic files are listed in `docs/MANUSCRIPT_RUN_MANIFEST.md`. |
| Marker distribution table | Inspect manuscript Table 9 values in `docs/results/README.md`; the local marker-distribution artifact is listed in `docs/MANUSCRIPT_RUN_MANIFEST.md`. |

The copied configs and documented result paths provide the bridge between public scripts and the local manuscript artifacts without redistributing raw datasets or large checkpoints.

## Reproducibility Scope
This repository reconstructs the raw-data-to-results code path when external datasets and local model weights are available. Exact numerical values can vary with:
- External dataset version and preprocessing state.
- Base model and tokenizer revision.
- Step1 checkpoint used for top-k marker generation.
- GPU hardware, CUDA runtime, quantization kernels, bf16 behavior, and random seed.
- Early stopping and checkpoint selection.

Saved local artifacts show the manuscript-facing seeds, model identifiers, precision, quantization, LoRA settings, and result files that were available for this release.

## Code Availability
- Repository: `https://github.com/hong0002/KoDiMARC`
- License: MIT License

The PeerJ submission cites the repository URL above. The repository contains the source code, configs, sample schemas, and reproduction instructions used for review.

## Citation
If this repository is used in research, cite the accompanying KoDiMARC manuscript after publication. Users must also follow the citation and license requirements of Korean Wikipedia / KoWiki, KorNLI, AI Malpyeong, and the selected base model.

## License
This repository is released under the MIT License. See `LICENSE` for details.

External datasets and pretrained model weights are not covered by this repository's MIT License. They remain governed by their original providers' access terms and licenses.

## Contribution Guidelines
This repository is maintained as a reproducibility release for the accompanying manuscript. Contributions that improve documentation, fix reproducibility bugs, or clarify setup instructions are welcome.

Useful issue reports include the command, config path, dataset split, observed behavior, expected behavior, Python version, package versions, GPU model, and CUDA/driver information.
