# Reported Result Tables

The CSV files in this directory preserve the values reported in manuscript Tables 5-9. Values use the precision shown in the manuscript; they are compact table summaries rather than raw prediction files.

| File | Manuscript table | Source and regeneration |
| --- | --- | --- |
| `table5_step1_marker_prediction.csv` | Table 5, Step1 marker prediction | Transcribed from the manuscript table. `scripts/step1/train_step1_generator.py` trains the SFT models, but the repository does not include the commercial-LLM evaluation driver used for every row. |
| `table6_main_results.csv` | Table 6, main results | Transcribed from the manuscript table. Per-run KoDiMARC metrics are produced by `scripts/step2/evaluate_step2.py`; baseline aggregation is not automated in this repository. |
| `table7_ablation_results.csv` | Table 7, ablations | Transcribed from the manuscript table. The corresponding executable configurations are under `configs/manuscript/step2_ablation_*.yaml`. |
| `table8_no_with_wrong_results.csv` | Table 8, diagnostic views | Transcribed from the manuscript table. NO, WITH, and WRONG metrics are written by `scripts/step2/evaluate_step2.py`. |
| `table9_marker_distribution.csv` | Table 9, marker distribution | Cleaned from the saved test-split marker-count summary and checked against the manuscript table. |

Run-level provenance and known aggregation limits are documented in [`docs/EXPERIMENT_MANIFEST.md`](../../docs/EXPERIMENT_MANIFEST.md). Full regeneration requires the external datasets and trained checkpoints described in the main README.
