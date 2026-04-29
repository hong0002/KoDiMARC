# KoDiMARC

## Project Overview
KoDiMARC (Korean Discourse Marker-aware Relation Classifier) is a two-stage framework for Korean sentence-pair relation classification. The repository accompanies a paper submission and is organized as a public reproducibility release rather than a full data release. It provides the core implementation for weakly supervised discourse marker generation in Step1 and marker-aware multi-task relation classification in Step2.

## Method Summary
KoDiMARC consists of two connected stages. Step1 detects explicit discourse markers from Korean Wikipedia sentence pairs, removes the marker from the hypothesis, and trains a response-only discourse marker generator under weak supervision. The trained Step1 model is then used to attach top-k discourse marker candidates and scores to downstream sentence pairs. Step2 consumes these candidate markers together with Korean NLI and AI Malpyeong relation labels, and trains a two-view multi-task classifier with no-marker, with-marker, and wrong-marker evaluation modes.

## Repository Structure
```text
KoDiMARC/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/
├── scripts/
├── src/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
└── docs/
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation
This repository does not include KorNLI, AI Malpyeong, or KoWiki data because of licensing and redistribution constraints. Users should prepare the raw datasets separately and point the scripts or configs to their local data locations. The `data/sample/` directory only provides small toy examples that illustrate the expected JSONL format.

To initialize the recommended local folder structure, run:
```bash
python scripts/setup_data_layout.py --root .
```

This creates directories such as:
- `data/raw/kowiki/extracted`
- `data/raw/kornli`
- `data/raw/ai_malpyeong`
- `data/processed/kowiki`
- `data/processed/multitask_nli/final`
- `outputs/step1_sft_example`
- `outputs/marker_sensitive`

## Step1: Discourse Marker Generator
Step1 builds adjacent KoWiki sentence pairs, detects explicit discourse markers with a rule-based lexicon, creates marker-removed hypotheses, and converts the resulting weak supervision into response-only SFT JSONL files.

The recommended workflow is documented in detail in [`docs/step1.md`](docs/step1.md):
1. Build adjacent sentence pairs.
2. Detect and remove explicit discourse markers.
3. Build response-only SFT data.
4. Train the Step1 generator.
5. Score top-k candidate markers for downstream data.

## Step2: Marker-aware Multi-task Relation Classifier
Step2 trains a marker-aware relation classifier over Korean NLI labels (`entailment`, `neutral`, `contradiction`) and AI Malpyeong logic labels (`forward`, `contrastive`, `compatible`). It uses a no-marker view and a predicted-marker view during training, and can additionally sample wrong-marker views for corruption-based contrastive supervision and robustness evaluation.

The recommended workflow is documented in detail in [`docs/step2.md`](docs/step2.md):
1. Materialize final Step2 JSONL files under `data/processed/multitask_nli/final/`.
2. Train the marker-aware multi-task classifier.
3. Evaluate the saved run in `NO`, `WITH`, and optional `WRONG` modes.

The default public Step2 example config reflects the marker-sensitive main setting used for the paper-facing runs, with KL consistency disabled by default.

## Evaluation Modes: NO / WITH / WRONG
- `NO`: no-marker view
- `WITH`: predicted-marker view using Step1 candidates
- `WRONG`: wrong-marker view using forbidden-category or corrupted markers

## Recommended Usage Flow
The repository is easier to understand if the two stages are run step by step instead of through a single shell wrapper.

### Step1 walkthrough
```bash
python scripts/setup_data_layout.py --root .

python scripts/step1/build_sentence_pairs.py   --input-dir data/raw/kowiki/extracted   --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py   --input-jsonl data/processed/kowiki/wiki_pairs.jsonl   --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl

python scripts/step1/build_sft_data.py   --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl   --train-output data/processed/kowiki/dp_sft_0train.jsonl   --valid-output data/processed/kowiki/dp_sft_valid.jsonl   --test-output data/processed/kowiki/dp_sft_test.jsonl

python scripts/step1/train_step1_generator.py   --config configs/step1/step1_sft_example.yaml
```

### Build the final Step2 split files
```bash
python scripts/step2/build_final_step2_data.py   --kornli-dir data/raw/kornli   --ai-malpyeong-dir data/raw/ai_malpyeong   --step1-model-name-or-path /path/to/step1_checkpoint   --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505   --output-dir data/processed/multitask_nli/final   --top-k 5   --fallback
```

This command creates:
- `data/processed/multitask_nli/final/train.jsonl`
- `data/processed/multitask_nli/final/dev_nli.jsonl`
- `data/processed/multitask_nli/final/dev_logic.jsonl`
- `data/processed/multitask_nli/final/test_nli.jsonl`
- `data/processed/multitask_nli/final/test_logic.jsonl`

### Step2 walkthrough
```bash
python scripts/step2/train_step2.py   --config configs/step2/step2_full_example.yaml

python scripts/step2/evaluate_step2.py   --config /path/to/run_dir/config_used.yaml   --exp marker_sensitive
```

### Optional convenience wrapper
An optional helper script is still provided for users who already understand the full workflow:
```bash
bash scripts/run_example_pipeline.sh
```
For first-time users, reading [`docs/step1.md`](docs/step1.md), [`docs/step2.md`](docs/step2.md), and [`docs/reproducibility.md`](docs/reproducibility.md) first is recommended.

## Reproducibility Notes
- Raw datasets are not distributed in this repository.
- Sample JSONL files are provided only to illustrate the expected format.
- Example configs use relative paths or placeholders and should be adapted locally.
- Model checkpoints and experiment outputs are intentionally excluded.
- Large-scale reproduction of the paper results requires local access to the original datasets and sufficient GPU resources.
- The public workflow is designed so that users can reconstruct the processed `final/*.jsonl` Step2 inputs from raw data and a locally trained Step1 checkpoint.

## Citation
Citation information will be added after publication.

## License
This repository is released under the MIT License. See `LICENSE` for details.
