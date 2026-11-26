# Experiment Manifest

This manifest links public configurations and table summaries to the saved run identifiers used during the experiments. Run identifiers are retained without machine-specific filesystem paths.

## Configurations

| Public config | Seed | Purpose | Manuscript connection |
| --- | ---: | --- | --- |
| `configs/manuscript/step1_kanana_8b_instruct_2505.yaml` | 42 | Step1 Kanana response-only SFT | Appendix Table A1; Kanana-1.5-8B-instruct-2505 row in Table 5 |
| `configs/manuscript/step1_exaone_1.2b_auxiliary.yaml` | 42 | Step1 EXAONE comparison | EXAONE-4.0-1.2B row in Table 5 |
| `configs/manuscript/step2_full.yaml` | 43 | Full marker-sensitive Step2 | Main KoDiMARC setting; source run `20260406_183903__kakaocorp__kanana-1.5-8b-instruct-2505` |
| `configs/manuscript/step2_ablation_no_mrel.yaml` | 42 | MREL disabled | Table 7; source run `20260413_234912__kakaocorp__kanana-1.5-8b-instruct-2505` |
| `configs/manuscript/step2_ablation_no_supcon.yaml` | 42 | supervised contrastive learning disabled | Table 7; source run `20260413_235041__kakaocorp__kanana-1.5-8b-instruct-2505` |
| `configs/manuscript/step2_ablation_no_marker_corruption.yaml` | 42 | marker corruption disabled | Table 7; source run `20260316_162026__kakaocorp__kanana-1.5-8b-instruct-2505` |
| `configs/manuscript/step2_ablation_no_marker_dropout.yaml` | 42 | marker dropout disabled | Table 7; source run `20260316_162133__kakaocorp__kanana-1.5-8b-instruct-2505` |
| `configs/manuscript/step2_legacy_full.yaml` | 42 | earlier full Step2 implementation | Source run `20260319_182916__kakaocorp__kanana-1.5-8b-instruct-2505` |

The no-corruption and no-dropout public YAML files use the current script schema, the ablation switches confirmed by their saved configs, and the shared settings reported in Appendix Table A2. The saved test metric directories for those variants do not contain a separate current-schema `config_used.yaml`; the public files therefore document the executable ablation definition rather than claiming byte-for-byte identity with those checkpoint directories.

## Result Tables

| Public artifact | Evidence retained | Scope |
| --- | --- | --- |
| `artifacts/results/table5_step1_marker_prediction.csv` | Manuscript Table 5 | All reported SFT and commercial-LLM rows; the commercial-LLM evaluation driver is not included. |
| `artifacts/results/table6_main_results.csv` | Manuscript Table 6 | Main results for AI Malpyeong and KorNLI. |
| `artifacts/results/table7_ablation_results.csv` | Manuscript Table 7 and saved ablation metrics | MREL, SupCon, marker-corruption, and marker-dropout rows. |
| `artifacts/results/table8_no_with_wrong_results.csv` | Manuscript Table 8 | Accuracy, macro precision, macro recall, and macro-F1 for NO, WITH, and WRONG. |
| `artifacts/results/table9_marker_distribution.csv` | Saved test-split marker counts and manuscript Table 9 | Top-three marker counts and within-label ratios. |

## Table 7 Run-Level Correspondence

- The no-MREL and no-SupCon run metric files match their Table 7 rows after rounding.
- Standalone test metric directories for no-corruption and no-dropout match their Table 7 rows after rounding.
