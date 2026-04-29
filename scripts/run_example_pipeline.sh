#!/usr/bin/env bash
set -euo pipefail

# Optional convenience wrapper.
# For first-time users, reading README.md, docs/step1.md, docs/step2.md, and docs/reproducibility.md first is recommended.

python scripts/setup_data_layout.py --root .

python scripts/step1/build_sentence_pairs.py   --input-dir data/raw/kowiki/extracted   --output-jsonl data/processed/kowiki/wiki_pairs.jsonl

python scripts/step1/detect_and_remove_markers.py   --input-jsonl data/processed/kowiki/wiki_pairs.jsonl   --output-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl

python scripts/step1/build_sft_data.py   --input-jsonl data/processed/kowiki/wiki_pairs_labeled.jsonl   --train-output data/processed/kowiki/dp_sft_train.jsonl   --valid-output data/processed/kowiki/dp_sft_valid.jsonl   --test-output data/processed/kowiki/dp_sft_test.jsonl

python scripts/step1/train_step1_generator.py   --config configs/step1/step1_sft_example.yaml

python scripts/step2/build_final_step2_data.py   --kornli-dir data/raw/kornli   --ai-malpyeong-dir data/raw/ai_malpyeong   --step1-model-name-or-path /path/to/step1_checkpoint   --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505   --output-dir data/processed/multitask_nli/final   --top-k 5   --fallback

python scripts/step2/train_step2.py   --config configs/step2/step2_full_example.yaml

python scripts/step2/evaluate_step2.py   --config /path/to/run_dir/config_used.yaml   --exp marker_sensitive
