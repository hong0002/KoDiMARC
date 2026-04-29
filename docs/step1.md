# Step1: Discourse Marker Generator

## Overview
Step1 constructs weakly supervised Korean discourse marker data from adjacent KoWiki sentence pairs. It detects explicit discourse markers in the hypothesis, removes the marker span, and converts the result into response-only SFT training data.

## Recommended Workflow
The public release is easier to follow when Step1 is run as a sequence of small, explicit steps.

### 1. Build adjacent sentence pairs
Use the sentence-pair builder to convert extracted KoWiki documents into adjacent sentence pairs.

```bash
python scripts/step1/build_sentence_pairs.py   --input-dir /path/to/kowiki/extracted   --output-jsonl data/processed/kowiki/wiki_pairs.jsonl
```

Expected output:
- `data/processed/kowiki/wiki_pairs.jsonl`

### 2. Detect and remove discourse markers
Run the rule-based detector to find explicit discourse markers and create marker-removed hypotheses.

```bash
python scripts/step1/detect_and_remove_markers.py   --input-jsonl data/processed/kowiki/wiki_pairs.jsonl   --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl
```

Expected output:
- `data/processed/kowiki/wiki_pairs_labeled.jsonl`

### 3. Build response-only SFT data
Convert the weak supervision JSONL into train, validation, and test splits for response-only SFT.

```bash
python scripts/step1/build_sft_data.py   --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl   --train-output data/processed/kowiki/dp_sft_train.jsonl   --valid-output data/processed/kowiki/dp_sft_valid.jsonl   --test-output data/processed/kowiki/dp_sft_test.jsonl
```

Expected outputs:
- `data/processed/kowiki/dp_sft_train.jsonl`
- `data/processed/kowiki/dp_sft_valid.jsonl`
- `data/processed/kowiki/dp_sft_test.jsonl`

### 4. Train the Step1 generator
Train the discourse marker generator with the example Step1 config.

```bash
python scripts/step1/train_step1_generator.py   --config configs/step1/step1_sft_example.yaml
```

Expected output:
- a local Step1 checkpoint directory under the output path defined in the config

### 5. Score top-k candidate markers
Use the trained Step1 generator to attach top-k candidate markers and scores to downstream sentence pairs.

```bash
python scripts/step1/score_topk_markers.py   --input-jsonl data/processed/multitask_nli/train.jsonl   --output-jsonl data/processed/multitask_nli/train_with_step1_topk.jsonl   --model-name-or-path /path/to/step1_checkpoint   --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505   --top-k 5   --fallback
```

Expected output:
- `data/processed/multitask_nli/train_with_step1_topk.jsonl`

## Discourse Marker Lexicon
The rule-based detector uses a fixed discourse marker lexicon covering the following coarse categories:
- `ADD`
- `CONTRAST`
- `CAUSAL`
- `EXPLAN`
- `CONCESS`
- `COND`
- `EXAMPLE`

The Korean marker strings themselves are preserved because they are part of the actual model input and output space.

## Marker Detection on KoWiki Sentence Pairs
The public release follows a simple explicit-marker extraction pipeline:
1. Split Korean Wikipedia documents into adjacent sentence pairs.
2. Detect discourse markers at the beginning of the hypothesis.
3. Also detect comma-delimited intra-sentence discourse markers when they match the lexicon.
4. Keep only examples for which the marker can be mapped to one of the coarse categories.

## Marker Removal
When a marker is detected, the script writes a marker-removed hypothesis field:
- `s2`: original hypothesis
- `s2_no_marker`: marker-removed hypothesis

## Weak Supervision JSONL Format
The rule-based detection step writes JSONL rows of the form:
```json
{
  "s1": "문장1",
  "s2": "그러나 문장2",
  "s2_no_marker": "문장2",
  "label": "CONTRAST",
  "marker": "그러나"
}
```

## Response-only SFT Format
The SFT builder converts the weak supervision into response-only training examples:
```json
{
  "instruction": "...",
  "input": "...",
  "output": "그러나",
  "label": "CONTRAST",
  "marker": "그러나"
}
```
Only the output tokens are used for language-model loss.

## Top-k Marker Scoring
The public scoring script loads a trained Step1 generator, prompts it with a sentence pair, and aggregates multiple generations into top-k candidate markers. The output attaches the following fields:
- `step1_topk_markers`
- `step1_topk_scores`
- `step1_top1_marker`
- `step1_top1_score`

## Output Fields for Step2
After Step1 scoring, downstream Step2 data can include:
- `step1_topk_markers`
- `step1_topk_scores`
- `step1_top1_marker`
- `step1_top1_score`
- `premise_marked`
- `hypothesis_marked`
- `pair_marked`
