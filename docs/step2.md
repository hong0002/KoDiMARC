# Step2: Marker-Aware Multi-Task Classification

Step2 jointly trains on KorNLI and AI Malpyeong using a shared decoder-based causal LLM backbone. KorNLI supplies `entailment`, `neutral`, and `contradiction`; AI Malpyeong supplies `forward`, `contrastive`, and `compatible`.

The source AI Malpyeong values `순접`, `역접`, and `양립` map to `forward`, `contrastive`, and `compatible`. Other discourse categories such as `CAUSAL`, `EXPLAN`, and `COND` belong to the Step1 marker lexicon, not the Step2 LOGIC label space.

## Input Schema

Each JSONL row contains:

- `id`, `source`, `premise`, and `hypothesis`
- either `nli_label` or `logic_label`; the unused task label is `null`
- `step1_topk_markers` and aligned `step1_topk_scores`

`data/sample/step2_sample.jsonl` illustrates NLI and LOGIC rows. Full data are written to `data/processed/multitask_nli/final/` by `scripts/step2/build_final_step2_data.py`.

## Architecture

`src/kodimarc/step2/loader.py` loads the decoder-based causal LLM backbone with 8-bit quantization and LoRA. `src/kodimarc/step2/model.py` applies:

- last-valid-token pooling for the sentence-pair representation
- marker-span pooling for the marker representation
- a base-delta prediction head
- task-specific 3-way NLI and 3-way LOGIC heads
- a 2-way MREL head

The input views are:

- **NO**: no candidate marker
- **WITH**: the Step1 predicted marker
- **WRONG**: a marker sampled from a label-specific forbidden category

Training combines clean View1/View2 classification, marker dropout, marker corruption, MREL, supervised contrastive learning, and a WITH-WRONG margin objective. The implementation is in `src/kodimarc/step2/trainer.py` and `src/kodimarc/step2/losses.py`.

## Training and Evaluation

```bash
python scripts/step2/train_step2.py \
  --config configs/manuscript/step2_full.yaml
```

Training creates a timestamped run directory beneath `outputs/marker_sensitive/`, saves `config_used.yaml`, and evaluates the selected checkpoint. To repeat evaluation:

```bash
STEP2_RUN_CONFIG=$(find outputs/marker_sensitive \
  -mindepth 2 -maxdepth 2 -name config_used.yaml | sort | tail -n 1)

python scripts/step2/evaluate_step2.py \
  --config "$STEP2_RUN_CONFIG" \
  --exp marker_sensitive \
  --output-subdir test_eval
```

The evaluator writes metrics, confusion matrices, transition analyses, and marker-category reports for NO, WITH, and WRONG modes.

## Ablations

Executable variants are provided for MREL, supervised contrastive learning, marker corruption, and marker dropout under `configs/manuscript/`. Their reported WITH-view metrics are preserved in `artifacts/results/table7_ablation_results.csv`.
