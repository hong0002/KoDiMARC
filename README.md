# KoDiMARC

**KoDiMARC: Two-View Multi-Task Learning with Discourse Markers for Robust Korean Sentence Relation Classification**

## Overview

KoDiMARC is a two-stage framework for Korean sentence-pair relation classification. Step1 constructs a weakly supervised discourse-marker corpus from adjacent Korean Wikipedia (KoWiki) sentences, trains a marker generator, and ranks top-k marker candidates. Step2 uses those candidates in a marker-aware multi-task classifier trained jointly on KorNLI and AI Malpyeong.

Step2 has a decoder-based causal LLM backbone with last-valid-token pooling and marker-span pooling. Its base-delta prediction head feeds task-specific NLI and LOGIC heads, while a 2-way marker-reliability (MREL) head, marker dropout, marker corruption, supervised contrastive learning, and a WITH-WRONG margin objective reduce sensitivity to absent or misleading markers.

## Repository Structure

```text
KoDiMARC/
├── artifacts/results/       # CSV summaries of manuscript Tables 5-9
├── configs/manuscript/      # executable experiment configurations
├── configs/step1/           # editable Step1 example
├── configs/step2/           # editable Step2 examples
├── data/sample/             # small JSONL schema examples
├── docs/                    # detailed workflow and experiment notes
├── scripts/                 # data, training, evaluation, and validation CLIs
└── src/kodimarc/            # Step1 and Step2 implementation modules
```

The main implementation paths are:

| Component | Code |
| --- | --- |
| Marker lexicon and seven Step1 categories | `src/kodimarc/common/markers.py` |
| KoWiki sentence-pair construction | `scripts/step1/build_sentence_pairs.py` |
| Marker detection and removal | `scripts/step1/detect_and_remove_markers.py`, `src/kodimarc/step1/marker_detection.py` |
| Step1 SFT preparation and training | `scripts/step1/build_sft_data.py`, `scripts/step1/train_step1_generator.py`, `src/kodimarc/step1/` |
| Top-k marker scoring and Step2 data construction | `scripts/step1/score_topk_markers.py`, `scripts/step2/build_final_step2_data.py` |
| Step2 data loading and views | `src/kodimarc/step2/dataset.py`, `src/kodimarc/step2/prompt.py` |
| Step2 backbone, pooling, and heads | `src/kodimarc/step2/loader.py`, `src/kodimarc/step2/model.py` |
| Step2 objectives and training | `src/kodimarc/step2/losses.py`, `src/kodimarc/step2/trainer.py`, `scripts/step2/train_step2.py` |
| NO/WITH/WRONG evaluation | `src/kodimarc/step2/evaluate.py`, `src/kodimarc/step2/eval_utils.py`, `scripts/step2/evaluate_step2.py` |

Detailed descriptions are available in [docs/step1.md](docs/step1.md), [docs/step2.md](docs/step2.md), and [docs/reproducibility.md](docs/reproducibility.md).

## Datasets

| Resource | Role | Local directory | Access |
| --- | --- | --- | --- |
| Korean Wikipedia / KoWiki | Step1 weak supervision from adjacent sentences containing explicit discourse markers | `data/raw/kowiki/extracted/` | [Wikimedia Dumps](https://dumps.wikimedia.org/kowiki/) |
| KorNLI | Step2 NLI supervision: `entailment`, `neutral`, `contradiction` | `data/raw/kornli/` | [Kakao Brain KorNLU repository](https://github.com/kakaobrain/kor-nlu-datasets) |
| AI Malpyeong | Step2 LOGIC supervision: `forward`, `contrastive`, `compatible` | `data/raw/ai_malpyeong/` | [NIKL report/data page](https://www.korean.go.kr/front/reportData/reportDataView.do?mn_id=207&pageIndex=1&report_seq=1192&searchOrder=years); dataset access requires authentication through the [AI Malpyeong platform](https://kli.korean.go.kr/benchmark/home.do) |
| Kanana-1.5-8B-instruct-2505 | Decoder-based causal LLM backbone for the reported Step1 and Step2 settings | Hugging Face cache or a local model directory | [Model page](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505) |

The AI Malpyeong provider labels `순접`, `역접`, and `양립` are reported as `forward`, `contrastive`, and `compatible`, respectively. They are distinct from the seven KoWiki-derived Step1 discourse categories: `ADD`, `CONTRAST`, `CAUSAL`, `EXPLAN`, `CONCESS`, `COND`, and `EXAMPLE`.

Full raw datasets, pretrained weights, adapters, and trained checkpoints are not redistributed. They remain subject to provider access conditions, licenses, and storage constraints. The files under `data/sample/` illustrate the expected JSONL schemas and are not sufficient to reproduce manuscript results.

## Installation

Python 3.12 and a CUDA-capable GPU environment are recommended for full training. Create an environment and install the package-name dependency list:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Step1 and Step2 training use 8-bit model loading and LoRA. CPU-only execution is suitable for the validation and preprocessing checks, but not for the reported 8B-model training workflow.

## Configuration

Executable configurations associated with the experiments are stored under `configs/manuscript/`:

| Config | Purpose |
| --- | --- |
| `step1_kanana_8b_instruct_2505.yaml` | Step1 Kanana generator settings from Appendix Table A1, completed with documented implementation defaults |
| `step1_exaone_1.2b_auxiliary.yaml` | Auxiliary EXAONE Step1 comparison |
| `step2_full.yaml` | Full marker-sensitive Step2 configuration |
| `step2_ablation_no_mrel.yaml` | MREL ablation |
| `step2_ablation_no_supcon.yaml` | supervised contrastive learning ablation |
| `step2_ablation_no_marker_corruption.yaml` | marker-corruption ablation |
| `step2_ablation_no_marker_dropout.yaml` | marker-dropout ablation |
| `step2_legacy_full.yaml` | earlier full-run configuration retained for result provenance |

Each file is directly parseable by the corresponding training script. Paths point to the local data layout created below. Provenance and the relationship between saved runs and manuscript tables are summarized in [docs/EXPERIMENT_MANIFEST.md](docs/EXPERIMENT_MANIFEST.md).

## Methodology

### Step1: Discourse-Marker Generation

1. Split extracted KoWiki documents into adjacent sentence pairs.
2. Detect explicit markers with the predefined Korean lexicon and assign one of seven discourse categories.
3. Remove the detected marker from the second sentence.
4. Convert the weakly supervised pairs to response-only SFT records.
5. Fine-tune the Step1 causal LLM generator.
6. Score the marker lexicon by conditional likelihood and retain top-k candidates and scores.

### Step2: Multi-Task Relation Classification

Step2 combines KorNLI NLI labels with the three AI Malpyeong LOGIC labels. The shared decoder-based causal LLM produces a last-valid-token sentence-pair representation and a marker-span representation. Task-specific NLI and LOGIC predictions are computed with the base-delta head. Training uses a no-marker view and a predicted-marker view, with marker dropout, marker corruption, MREL, supervised contrastive learning, and the WITH-WRONG margin objective. Evaluation reports NO, WITH, and WRONG modes separately.

## Usage

### Lightweight Sample Check

This command runs immediately after installing the dependencies and does not load a model:

```bash
python scripts/step1/build_sft_data.py \
  --input-jsonl data/sample/step1_sample.jsonl \
  --train-output /tmp/kodimarc_step1_train.jsonl \
  --valid-output /tmp/kodimarc_step1_valid.jsonl \
  --test-output /tmp/kodimarc_step1_test.jsonl

python scripts/validate_reproducibility.py
```

The sample check verifies JSONL parsing and preprocessing. It does not train a model or reproduce paper metrics.

### Full Reproduction Workflow

The remaining commands require the external datasets and model access described above.

1. Create the local directory layout.

```bash
python scripts/setup_data_layout.py --root .
```

2. Build the Step1 weak-supervision corpus.

```bash
python scripts/step1/build_sentence_pairs.py \
  --input-dir data/raw/kowiki/extracted \
  --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py \
  --input-jsonl data/processed/kowiki/wiki_pairs.jsonl \
  --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl
```

3. Build response-only Step1 SFT splits.

```bash
python scripts/step1/build_sft_data.py \
  --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl \
  --train-output data/processed/kowiki/dp_sft_train.jsonl \
  --valid-output data/processed/kowiki/dp_sft_valid.jsonl \
  --test-output data/processed/kowiki/dp_sft_test.jsonl
```

4. Train the Step1 generator and select the resulting run directory.

```bash
python scripts/step1/train_step1_generator.py \
  --config configs/manuscript/step1_kanana_8b_instruct_2505.yaml

STEP1_CHECKPOINT=$(find outputs/step1_kanana_8b_instruct_2505 \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
test -n "$STEP1_CHECKPOINT"
```

5. Convert KorNLI and AI Malpyeong and attach Step1 top-k markers.

```bash
python scripts/step2/build_final_step2_data.py \
  --kornli-dir data/raw/kornli \
  --ai-malpyeong-dir data/raw/ai_malpyeong \
  --step1-model-name-or-path "$STEP1_CHECKPOINT" \
  --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505 \
  --output-dir data/processed/multitask_nli/final \
  --top-k 5 \
  --fallback
```

The command writes `train.jsonl`, `dev_nli.jsonl`, `dev_logic.jsonl`, `test_nli.jsonl`, `test_logic.jsonl`, and `build_manifest.json` under `data/processed/multitask_nli/final/`.

6. Train Step2.

```bash
python scripts/step2/train_step2.py \
  --config configs/manuscript/step2_full.yaml
```

7. Evaluate the trained checkpoint under NO, WITH, and WRONG modes.

```bash
STEP2_RUN_CONFIG=$(find outputs/marker_sensitive \
  -mindepth 2 -maxdepth 2 -name config_used.yaml | sort | tail -n 1)
test -n "$STEP2_RUN_CONFIG"

python scripts/step2/evaluate_step2.py \
  --config "$STEP2_RUN_CONFIG" \
  --exp marker_sensitive \
  --output-subdir test_eval
```

Evaluation writes `metrics.json`, confusion matrices, transition summaries, and marker-category reports under the selected run directory.

## Reproducing the Reported Results

The exact rounded values from manuscript Tables 5-9 are provided in [artifacts/results/](artifacts/results/). Step2 evaluation produces unrounded per-run metrics; the table CSVs retain the precision used in the manuscript.

Tables 6 and 8 aggregate task/view values that match different saved runs. No single saved run summary available for this release contains every displayed cell. Table 5 also includes commercial-LLM rows whose evaluation driver is not part of the public code. These boundaries are documented in [docs/EXPERIMENT_MANIFEST.md](docs/EXPERIMENT_MANIFEST.md) so that table transcription is not presented as single-run regeneration.

Exact reruns can vary with external dataset revisions, base-model and tokenizer revisions, CUDA and GPU behavior, quantization kernels, random seed, early stopping, and checkpoint selection.

## Computational Environment

| Item | Configuration |
| --- | --- |
| Operating system | Linux `5.15.0-139-generic`, `x86_64 GNU/Linux` |
| CPU | Intel Core i9-10900X @ 3.70 GHz, 10 cores / 20 threads |
| GPU | 4 x NVIDIA GeForce RTX 3090, 24 GiB VRAM each |
| NVIDIA driver | `535.183.01` |
| System memory | 188 GiB RAM |
| Python | `3.12.4` |
| Main backbone | `kakaocorp/kanana-1.5-8b-instruct-2505` |
| Precision and quantization | bf16 mixed precision; 8-bit model loading |
| LoRA | rank 64, alpha 128, dropout 0.0 |

Package names are listed in `requirements.txt`. The saved experiment records do not provide a complete package lock, so no lock file is labeled as the exact training environment.

## Data and Code Availability

Implementation code, executable configurations, sample schemas, result summaries, and reproduction instructions are available at https://github.com/hong0002/KoDiMARC. Versioned source archives are listed on the [GitHub Releases page](https://github.com/hong0002/KoDiMARC/releases). The code is distributed under the MIT License.

External datasets and pretrained models must be obtained from their original providers. The repository samples are schema examples only; full-scale results require the source datasets and locally trained Step1 and Step2 checkpoints.

## Citation

Please cite the accompanying manuscript:

> Ji-Hong Park, Juyoung Kim, Sang-Min Choi, and Gun-Woo Kim. “KoDiMARC: Two-View Multi-Task Learning with Discourse Markers for Robust Korean Sentence Relation Classification.” Manuscript submitted to PeerJ Computer Science.

KorNLI, AI Malpyeong, Korean Wikipedia, and the selected pretrained model should also be cited according to their providers' instructions.

## License

Repository code is available under the [MIT License](LICENSE). External datasets and model weights are excluded from that license and retain their original terms.

## Contributing

Contributions that correct reproducibility bugs, documentation errors, or command/config mismatches are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the information needed in an issue or pull request.
