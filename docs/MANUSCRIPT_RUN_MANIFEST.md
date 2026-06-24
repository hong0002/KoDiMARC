# Manuscript Run Manifest

This manifest records the local experiment artifacts that were available when preparing the KoDiMARC public reproducibility documentation. It is an audit trail for the copied configs and result summaries referenced from `README.md` and `docs/reproducibility.md`.

## Searched Local Artifact Roots
| Root | Purpose |
| --- | --- |
| `/data1/jihong/multi-task_NLI3` | Step2 multi-task training, evaluation, ablation, diagnostics, and marker-distribution artifacts. |
| `/home/jihong/Multi-Task_NLI` | Step1 and marker-prediction artifacts, including a saved Step1 SFT local config and marker-only evaluation summary. |

## Copied Configs
The following configs were copied into this repository under `configs/peerj_review/` so that reviewers can inspect the saved manuscript-facing settings without depending on the local artifact tree.

| Repository config | Source artifact | Verified role |
| --- | --- | --- |
| `configs/peerj_review/step1_sft_local_artifact.yaml` | `/home/jihong/Multi-Task_NLI/backup/configs/01_1_config_sft_only.yaml` | Step1 discourse-marker SFT local artifact config using `LGAI-EXAONE/EXAONE-4.0-1.2B`, 8-bit quantization, LoRA rank 64, seed 42, and KoWiki SFT JSONL paths. |
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
| Step2 backbone in copied configs | `kakaocorp/kanana-1.5-8b-instruct-2505` |
| Step2 precision / quantization in copied configs | `bf16` and `8bit` |
| Step2 LoRA in copied configs | rank `64`, alpha `128`, dropout `0.0` |

The searched saved run artifacts did not include a complete `pip freeze`, conda export, or immutable dependency lock file. Package names are therefore documented through `requirements.txt`, while copied configs preserve the experiment settings available in the saved artifacts.

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
Compact metric values are indexed in `docs/results/README.md`. The values there were transcribed from the local summaries listed above.

## Data and Checkpoint Scope
Full raw datasets, Hugging Face model weights, and trained checkpoints are not stored in this repository. The public release provides code, configs, sample schemas, and documentation. Dataset downloads and checkpoint generation must be performed locally under the terms of the original providers.

## Archive and DOI Scope
No release archive DOI or persistent archive record was found in the searched local artifacts. The code availability statement therefore uses the public GitHub repository URL and omits a DOI.
