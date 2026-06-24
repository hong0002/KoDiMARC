# Result Artifact Index

This directory documents the manuscript-facing result values and the local artifact files used for cross-checking. The numerical summaries in this repository follow the manuscript tables. Some local result files may store additional decimal places or intermediate diagnostic outputs; the manuscript-facing tables are rounded as reported in the PDF.

Manuscript PDF used as the source for reported values:

```text
/data1/jihong/LaTeX_Template_for_PeerJ_Journal_Submissions/main.pdf
```

## Table 5: Step1 Discourse-marker Prediction
| Regime | Model | Marker Acc | Label Acc | Marker P | Label P | Marker R | Label R | Marker F1 | Label F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SFT fine-tuned | Kanana-1.5-8B-base | 0.470 | 0.782 | 0.163 | 0.520 | 0.148 | 0.474 | 0.133 | 0.472 |
| SFT fine-tuned | Kanana-1.5-8B-instruct-2505 | 0.480 | 0.772 | 0.165 | 0.515 | 0.165 | 0.525 | 0.147 | 0.502 |
| SFT fine-tuned | Kanana-1.5-2.1B-base | 0.124 | 0.464 | 0.062 | 0.367 | 0.052 | 0.395 | 0.037 | 0.316 |
| SFT fine-tuned | Kanana-1.5-2.1B-instruct-2505 | 0.398 | 0.702 | 0.107 | 0.502 | 0.112 | 0.580 | 0.097 | 0.477 |
| SFT fine-tuned | EXAONE-4.0-1.2B | 0.228 | 0.440 | 0.119 | 0.379 | 0.065 | 0.245 | 0.058 | 0.229 |
| LLM zero-shot | GPT-5.2 | 0.288 | 0.528 | 0.129 | 0.338 | 0.115 | 0.426 | 0.090 | 0.317 |
| LLM zero-shot | Claude-sonnet-4.5 | 0.310 | 0.665 | 0.201 | 0.430 | 0.124 | 0.437 | 0.113 | 0.371 |
| LLM few-shot | GPT-5.2 | 0.360 | 0.559 | 0.135 | 0.344 | 0.130 | 0.441 | 0.114 | 0.342 |
| LLM few-shot | Claude-sonnet-4.5 | 0.404 | 0.674 | 0.176 | 0.419 | 0.171 | 0.551 | 0.148 | 0.430 |

Kanana-1.5-8B-instruct-2505 is documented as the manuscript Step1 generator because it achieved the highest exact marker accuracy and marker Macro-F1 among the SFT-trained models.

## Table 6: Main End-to-end Results
| Method | AI-M Acc | AI-M P | AI-M R | AI-M F1 | KorNLI Acc | KorNLI P | KorNLI R | KorNLI F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single-task | 0.855 | 0.579 | 0.597 | 0.588 | 0.870 | 0.870 | 0.870 | 0.869 |
| Single-task+M | 0.881 | 0.590 | 0.615 | 0.602 | 0.870 | 0.870 | 0.870 | 0.870 |
| KoDiMARC (NO) | 0.870 | 0.654 | 0.657 | 0.652 | 0.865 | 0.866 | 0.865 | 0.865 |
| KoDiMARC (WITH) | 0.874 | 0.671 | 0.685 | 0.674 | 0.873 | 0.873 | 0.873 | 0.873 |
| GPT-5.2 | 0.297 | 0.428 | 0.233 | 0.219 | 0.730 | 0.613 | 0.547 | 0.550 |
| Claude-sonnet-4.5 | 0.624 | 0.445 | 0.422 | 0.390 | 0.858 | 0.648 | 0.644 | 0.644 |
| GPT-5.2 few-shot | 0.762 | 0.468 | 0.456 | 0.447 | 0.790 | 0.625 | 0.593 | 0.596 |
| Claude-sonnet-4.5 few-shot | 0.799 | 0.475 | 0.474 | 0.465 | 0.873 | 0.660 | 0.655 | 0.655 |
| KoELECTRA (Base) | 0.758 | 0.511 | 0.530 | 0.516 | 0.793 | 0.795 | 0.793 | 0.793 |
| XLM-Roberta (Base) | 0.565 | 0.433 | 0.400 | 0.355 | 0.783 | 0.783 | 0.782 | 0.783 |
| KoBERT | 0.817 | 0.545 | 0.570 | 0.557 | 0.733 | 0.735 | 0.733 | 0.733 |
| KcELECTRA-base | 0.750 | 0.500 | 0.524 | 0.512 | 0.793 | 0.793 | 0.793 | 0.793 |
| KoELECTRA (Small) | 0.661 | 0.442 | 0.461 | 0.450 | 0.726 | 0.730 | 0.726 | 0.726 |
| SimCSE | 0.776 | 0.518 | 0.542 | 0.529 | 0.800 | 0.801 | 0.800 | 0.799 |
| DisSent-kr | 0.892 | 0.595 | 0.622 | 0.608 | 0.796 | 0.797 | 0.796 | 0.796 |
| ConnPrompt-kr | 0.869 | 0.580 | 0.607 | 0.593 | 0.812 | 0.813 | 0.812 | 0.812 |
| TEPrompt-kr | 0.806 | 0.540 | 0.562 | 0.550 | 0.796 | 0.801 | 0.796 | 0.797 |

## Table 7: Ablation Study
| Variant | KorNLI Acc | KorNLI F1 | KorNLI Delta F1 | AI-M Acc | AI-M F1 | AI-M Delta F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Full KoDiMARC (best) | 0.8730 | 0.8730 | - | 0.8736 | 0.6743 | - |
| w/o MREL | 0.8613 | 0.8617 | -0.0113 | 0.8810 | 0.6816 | +0.0073 |
| w/o SupCon | 0.8654 | 0.8654 | -0.0077 | 0.8401 | 0.6585 | -0.0159 |
| w/o marker corruption | 0.8543 | 0.8532 | -0.0198 | 0.8625 | 0.6282 | -0.0461 |
| w/o marker dropout | 0.8734 | 0.8735 | +0.0005 | 0.8699 | 0.6524 | -0.0219 |

## Table 8: NO / WITH / WRONG Diagnostics
| View | AI-M Acc | AI-M P | AI-M R | AI-M F1 | KorNLI Acc | KorNLI P | KorNLI R | KorNLI F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NO | 0.870 | 0.654 | 0.657 | 0.652 | 0.865 | 0.866 | 0.865 | 0.865 |
| WITH | 0.874 | 0.671 | 0.685 | 0.674 | 0.873 | 0.873 | 0.873 | 0.873 |
| WRONG | 0.862 | 0.650 | 0.651 | 0.647 | 0.861 | 0.863 | 0.861 | 0.861 |

## Table 9: Step1 Top-1 Marker Distribution
| Dataset | Gold label | N | Top-1 predicted marker glosses |
| --- | --- | --- | --- |
| AI Malpyeong | Forward | 125 | therefore 72 (57.6%); rather 48 (38.4%); because 2 (1.6%) |
| AI Malpyeong | Compatible | 12 | rather 9 (75.0%); therefore 3 (25.0%) |
| AI Malpyeong | Contrastive | 132 | rather 69 (52.3%); nevertheless 35 (26.5%); however 20 (15.2%) |
| KorNLI | Contradiction | 1652 | if/when 859 (52.0%); however 579 (35.0%); also 106 (6.4%) |
| KorNLI | Entailment | 1651 | if/when 775 (46.9%); however 635 (38.5%); also 130 (7.9%) |
| KorNLI | Neutral | 1651 | if/when 959 (58.1%); however 509 (30.8%); also 94 (5.7%) |

## Local Artifact Cross-checks
| Artifact | Local path | Use |
| --- | --- | --- |
| Step1 marker-only LOGIC top-1 summary | `/home/jihong/Multi-Task_NLI/outputs/marker_only_top1_logic/20260525_170242/summary.csv` | Local marker-prediction cross-check artifact. |
| Step2 full result summary | `/data1/jihong/multi-task_NLI3/outputs/20260319_182916__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Earlier full result cross-check artifact. |
| Step2 marker-sensitive summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260406_183903__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Main marker-sensitive local artifact corresponding to the manuscript-facing KoDiMARC setting. |
| Step2 no-MREL ablation summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_234912__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Ablation cross-check artifact. |
| Step2 no-SupCon ablation summary | `/data1/jihong/multi-task_NLI3/outputs/marker_sensitive/20260413_235041__kakaocorp__kanana-1.5-8b-instruct-2505/summary.csv` | Ablation cross-check artifact. |
| Marker distribution summary | `/data1/jihong/multi-task_NLI3/outputs/marker_distribution/label_marker_counts_summary.json` | Marker-distribution cross-check artifact. |

## Diagnostics
NO / WITH / WRONG diagnostic outputs are stored next to each run's `metrics.json` files in the local artifact tree. Typical files include:
- `test/metrics.json`
- `test/test_nli_transition_summary.json`
- `test/test_logic_transition_summary.json`
- `test/test_nli_marker_category_report.csv`
- `test/test_logic_marker_category_report.csv`

These diagnostics are generated by `scripts/step2/evaluate_step2.py` and the utilities under `src/kodimarc/step2/eval_utils.py` and `src/kodimarc/step2/metrics.py`.
