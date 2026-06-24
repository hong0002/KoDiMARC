# Manuscript Run Manifest

This manifest records the local experiment artifacts that were available when preparing the KoDiMARC public reproducibility documentation. It is an audit trail for the copied configs and result summaries referenced from `README.md` and `docs/reproducibility.md`.

## Searched Local Artifact Roots
| Root | Purpose |
| --- | --- |
| `/data1/jihong/multi-task_NLI3` | Step2 multi-task training, evaluation, ablation, diagnostics, and marker-distribution artifacts. |
| `/home/jihong/Multi-Task_NLI` | Step1 and marker-prediction artifacts, including a saved auxiliary EXAONE Step1 SFT local config and marker-only evaluation summary. |

## Review Configs
The following configs are provided under `configs/peerj_review/` so that reviewers can inspect manuscript-facing settings without depending on the local artifact tree.

| Repository config | Source artifact | Verified role |
| --- | --- | --- |
| `configs/peerj_review/step1_kanana_8b_instruct_2505_manuscript.yaml` | Manuscript PDF `/data1/jihong/LaTeX_Template_for_PeerJ_Journal_Submissions/main.pdf`, Table 10 | Manuscript Step1 discourse-marker generator config using `kakaocorp/kanana-1.5-8b-instruct-2505`, 8-bit quantization, max sequence length 256, LoRA rank 64/alpha 128/dropout 0.0, AdamW, learning rate 1.0e-5, warmup 200, early stopping patience 3, and seed 42. |
| `configs/peerj_review/step1_sft_local_artifact.yaml` | `/home/jihong/Multi-Task_NLI/backup/configs/01_1_config_sft_only.yaml` | Auxiliary smaller Step1 local artifact config using `LGAI-EXAONE/EXAONE-4.0-1.2B`, 8-bit quantization, LoRA rank 64, seed 42, and KoWiki SFT JSONL paths. This is not the manuscript Step1 generator used for final Step2 results. |
| `configs/peerj_review/step2_full_result_run_20260319.yaml` | `/data1/jihong/multi-task_NLI3/outputs/20260319_182916__kakaocorp__kanana-1.5-8b-instruct-2505/config_used.yaml` | Step2 full result run with `kakaocorp/kanana-1.5-8b-instruct-2505`, seed 42, bf16, 8-bit quantization, LoRA rank 64, and NLI:LOGIC task ratio 3:1. |
| `configs/peerj_review/step2_marker_sensitive_run_20260406_seed43.yaml` | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260406_183903__kakaocorp__kanana-1.5-8b-instruct-2505/config_used.yaml` | Main marker-sensitive Step2 run with base-delta head, marker-aware head, MREL, SupCon, WITH-WRONG margin, seed 43, bf16, and 8-bit quantization. |
| `configs/peerj_review/step2_ablation_no_mrel_run_20260413.yaml` | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_234912__kakaocorp__kanana-1.5-8b-instruct-2505/config_used.yaml` | Marker-sensitive ablation with MREL disabled and seed 42. |
| `configs/peerj_review/step2_ablation_no_supcon_run_20260413.yaml` | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_235041__kakaocorp__kanana-1.5-8b-instruct-2505/config_used.yaml` | Marker-sensitive ablation with SupCon disabled and seed 42. |

## Environment Evidence
| Item | Verified value |
| --- | --- |
| Operating system | Linux `5.15.0-139-generic`, `x86_64 GNU/Linux` |
| CPU | Intel Core i9-10900X @ 3.70 GHz, 10 cores / 20 threads |
| GPU | 4 x NVIDIA GeForce RTX 3090, 24 GiB VRAM each |
| NVIDIA driver | `535.183.01` |
| System memory | 188 GiB RAM |
| Validation Python | Python `3.12.4` in the released validation environment |
| Step1 manuscript generator | `kakaocorp/kanana-1.5-8b-instruct-2505` |
| Step2 backbone in copied configs | `kakaocorp/kanana-1.5-8b-instruct-2505` |
| Step2 precision / quantization in copied configs | `bf16` and `8bit` |
| Step2 LoRA in copied configs | rank `64`, alpha `128`, dropout `0.0` |

Package names are documented through `requirements.txt`, while review configs preserve the experiment settings available for this release. All `configs/peerj_review/*.yaml` files were reformatted as block-style YAML and validated with PyYAML.

## Artifact Provenance
The Step1 Kanana review config records the settings reported in manuscript Table 10. Local artifact review also identified an auxiliary EXAONE Step1 config and the Step2 configs listed above. This manifest keeps those provenance details separate from the main reviewer-facing README.

## Key Result Artifacts
| Artifact | Local path | Notes |
| --- | --- | --- |
| Step1 marker-only LOGIC top-1 summary | `/home/jihong/Multi-Task_NLI/outputs/marker_only_top1_logic/20260525_170242/summary.csv` | Contains compatible-label handling variants for marker prediction over the LOGIC split. |
| Step2 full result summary | `/data1/jihong/multi-task_NLI3/outputs/20260319_182916__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Summary row for the `full` experiment. |
| Step2 full result metrics | `/data1/jihong/multi-task_NLI3/outputs/20260319_182916__kakaocorp__kanana-1.5-8b-instruct-2505/experiments/full/test/metrics.json` | Test metrics for NO, WITH, and WRONG modes. |
| Step2 marker-sensitive summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260406_183903__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Main marker-sensitive result row. |
| Step2 marker-sensitive metrics | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260406_183903__kakaocorp__kanana-1.5-8b-instruct-2505/experiments/marker_sensitive/test/metrics.json` | Test metrics for NO, WITH, and WRONG modes. |
| Step2 no-MREL ablation summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_234912__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Marker-sensitive ablation with MREL disabled. |
| Step2 no-SupCon ablation summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_235041__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Marker-sensitive ablation with SupCon disabled. |
| Step2 no-corruption ablation summary | `/data1/jihong/multi-task_NLI3/outputs/20260316_162026__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Ablation summary with marker corruption disabled. |
| Step2 no-dropout ablation summary | `/data1/jihong/multi-task_NLI3/outputs/20260316_162133__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Ablation summary with marker dropout disabled. |
| Step2 no-KL ablation summary | `/data1/jihong/multi-task_NLI3/outputs/20260316_162237__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Ablation summary with KL disabled. |
| Step2 baseline CE-only summary | `/data1/jihong/multi-task_NLI3/outputs/20260223_172504__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Earlier baseline-only CE run. |
| Marker distribution summary | `/data1/jihong/multi-task_NLI3/outputs/marker_distribution/label_marker_counts_summary.json` | Summarizes label and marker counts for final Step2 splits. |

## Verified Metric Values
Manuscript-facing metric values are indexed in `docs/results/README.md` and follow the rounded values reported in the manuscript PDF. Local artifact paths above provide cross-check files that may store additional decimal places or intermediate diagnostics.

## Data and Checkpoint Scope
This public release provides code, configs, sample schemas, and documentation. Full raw datasets, Hugging Face model weights, and trained checkpoints remain governed by their original providers' terms and must be prepared locally for full reproduction.

## Archive Scope
The code availability statement uses the public GitHub repository URL. A release tag or archival record can be added after the final repository review.
