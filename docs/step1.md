# Step1: Discourse-Marker Generation

Step1 derives weak supervision from explicit Korean discourse markers in adjacent KoWiki sentences. It trains a response-only SFT generator and scores a fixed marker lexicon to provide top-k candidates for Step2.

## Labels and Data Schema

The Step1 labels are the seven discourse-function categories defined in `src/kodimarc/common/markers.py`:

`ADD`, `CONTRAST`, `CAUSAL`, `EXPLAN`, `CONCESS`, `COND`, and `EXAMPLE`.

These labels describe KoWiki-derived marker categories. They are not AI Malpyeong LOGIC labels.

Rule-based preprocessing produces records with:

- `s1`: first sentence
- `s2`: original second sentence
- `s2_no_marker`: second sentence after removing the detected marker
- `label`: one of the seven Step1 categories
- `marker`: detected marker surface form

`data/sample/step1_sample.jsonl` contains two schema examples.

## Data Construction

Extract a KoWiki dump to `data/raw/kowiki/extracted/`, then run:

```bash
python scripts/step1/build_sentence_pairs.py \
  --input-dir data/raw/kowiki/extracted \
  --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py \
  --input-jsonl data/processed/kowiki/wiki_pairs.jsonl \
  --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl

python scripts/step1/build_sft_data.py \
  --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl \
  --train-output data/processed/kowiki/dp_sft_train.jsonl \
  --valid-output data/processed/kowiki/dp_sft_valid.jsonl \
  --test-output data/processed/kowiki/dp_sft_test.jsonl
```

The SFT records contain `instruction`, `input`, `output`, `label`, and `marker`. The loss is applied to the response tokens that generate the marker.

## Generator Training

The executable Kanana configuration is `configs/manuscript/step1_kanana_8b_instruct_2505.yaml`:

```bash
python scripts/step1/train_step1_generator.py \
  --config configs/manuscript/step1_kanana_8b_instruct_2505.yaml
```

Appendix Table A1 reports 8-bit loading, maximum sequence length 256, one epoch, per-device batch size 1, gradient accumulation 16, AdamW with learning rate `1.0e-5`, 200 warmup steps, LoRA rank 64/alpha 128/dropout 0.0, early-stopping patience 3, and seed 42. The YAML identifies additional implementation defaults separately.

## Top-k Scoring

`scripts/step1/score_topk_markers.py` computes each lexicon candidate's conditional log-likelihood and writes:

- `step1_topk_markers`
- `step1_topk_scores`
- `top1_marker`

For the full pipeline, `scripts/step2/build_final_step2_data.py` prepares KorNLI and AI Malpyeong splits and invokes the scorer for each split. See the root README for the complete command.
