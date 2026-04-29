# Step2: Marker-aware Multi-task Relation Classifier

## Overview
Step2 is a multi-task Korean sentence-pair relation classifier that combines KorNLI labels and AI Malpyeong logic labels. It optionally uses Step1 top-k discourse marker candidates to construct marker-aware views during training and evaluation.

## Recommended Workflow
The public release is easiest to reproduce if Step2 is approached as a short sequence of explicit stages.

### 1. Build the final Step2 JSONL files
Create the processed Step2 split files and attach Step1 top-k discourse marker fields.

```bash
python scripts/step2/build_final_step2_data.py   --kornli-dir data/raw/kornli   --ai-malpyeong-dir data/raw/ai_malpyeong   --step1-model-name-or-path /path/to/step1_checkpoint   --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505   --output-dir data/processed/multitask_nli/final   --top-k 5   --fallback
```

Expected outputs:
- `data/processed/multitask_nli/final/train.jsonl`
- `data/processed/multitask_nli/final/dev_nli.jsonl`
- `data/processed/multitask_nli/final/dev_logic.jsonl`
- `data/processed/multitask_nli/final/test_nli.jsonl`
- `data/processed/multitask_nli/final/test_logic.jsonl`
- `data/processed/multitask_nli/final/build_manifest.json`

The generated JSONL files include Step1 marker fields such as:
- `step1_topk_markers`
- `step1_topk_scores`
- `step1_top1_marker`
- `step1_s2_top1`
- `step1_s2_topk`
- `step1_status`

### 2. Train the marker-aware classifier
Run the public example config. The default example reflects the marker-sensitive main setting with KL consistency disabled.

```bash
python scripts/step2/train_step2.py   --config configs/step2/step2_full_example.yaml
```

Expected output:
- a timestamped run directory under `outputs/marker_sensitive/`

### 3. Evaluate the saved run
Evaluate the trained run with the saved `config_used.yaml` file.

```bash
python scripts/step2/evaluate_step2.py   --config /path/to/run_dir/config_used.yaml   --exp marker_sensitive
```

Expected outputs include evaluation files under the saved run directory.

## Lower-level Data Preparation Helper
A lower-level helper is also provided:
```bash
python scripts/step2/prepare_step2_data.py   --kornli-dir data/raw/kornli   --ai-malpyeong-dir data/raw/ai_malpyeong   --output-dir data/processed/multitask_nli/base
```
This helper only builds the merged KorNLI and AI Malpyeong JSONL files. It does not attach Step1 top-k marker predictions by itself.

## Input Fields
The primary Step2 inputs are:
- `premise`
- `hypothesis`
- `nli_label` for KorNLI examples
- `logic_label` for AI Malpyeong examples
- `step1_topk_markers`
- `step1_topk_scores`

## Multi-task Setup
The classifier jointly learns:
- Korean NLI labels: `entailment`, `neutral`, `contradiction`
- Logic labels: `forward`, `contrastive`, `compatible` (corresponding to the Korean labels used in AI Malpyeong)

## Views
### No-marker View
The no-marker view always omits the discourse marker candidate.

### Predicted-marker View
The predicted-marker view uses the top-ranked Step1 candidate marker or another sampled candidate from the Step1 top-k list.

## Marker Dropout
Marker dropout stochastically removes the predicted marker during training, which regularizes the model against over-reliance on Step1 predictions.

## Marker Corruption
Marker corruption replaces the predicted marker with a mismatched candidate. This provides a counterfactual marker signal and is used in the WITH-WRONG margin objective.

## Forbidden-category Wrong Marker Sampling
For logic supervision, KoDiMARC uses forbidden category mappings to sample wrong markers from categories that should be inconsistent with the gold relation label.

## Base-delta Prediction Head
The classifier supports a marker-aware base-delta head. The base representation predicts the relation without marker information, while the delta representation captures the additional effect of the marker-aware view.

## Marker Relation Compatibility (MREL)
MREL provides an auxiliary signal for learning whether a discourse marker is compatible with the gold relation.

## Supervised Contrastive Loss
A supervised contrastive component can be added across NLI and logic minibatches, optionally using queue-based memory for more negatives.

## WITH-WRONG Margin Loss
The WITH-WRONG margin objective encourages the predicted-marker view to outperform the wrong-marker view by a configurable margin.

## Evaluation Modes
- `NO`: evaluate the no-marker view.
- `WITH`: evaluate the predicted-marker view.
- `WRONG`: evaluate the wrong-marker view.
