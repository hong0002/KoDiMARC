# KoDiMARC

## Title
KoDiMARC: Korean Discourse Marker-aware Relation Classifier

## Description
KoDiMARC is a two-stage framework for Korean sentence-pair relation classification. The repository accompanies a manuscript submission and is intended as a reproducibility release for the algorithms and source code used in the study.

The framework first learns to generate Korean discourse markers from weakly supervised Korean Wikipedia sentence pairs. It then uses predicted discourse marker candidates as additional relation-aware signals for multi-task classification over Korean natural language inference and Korean logical-relation data.

This repository provides:
- Source code for data preparation, weak-supervision construction, model training, and evaluation.
- Example configuration files for Step1 and Step2 experiments.
- Small sample JSONL files that document the expected input formats.
- Reproducibility notes and step-by-step workflow documentation.

This repository does not provide the full raw datasets or trained checkpoints because of licensing, redistribution, and storage constraints.

## Dataset Information
The full experiments require local access to the following external resources:
- Korean Wikipedia / KoWiki extracted text for weakly supervised discourse marker data construction.
- KorNLI data for Korean NLI labels: `entailment`, `neutral`, and `contradiction`.
- AI Malpyeong logical-relation data for Korean logic labels such as `순접`, `역접`, `양립`, `인과`, `양보`, `조건`, `설명`, and `예시`.
- A compatible Korean instruction-tuned base language model. The public example configs use `kakaocorp/kanana-1.5-8b-instruct-2505`.

### External Data Access and Expected Files
The raw datasets and pretrained model weights are external to this repository. They should be obtained from their original providers and placed under the local paths expected by the scripts.

| Resource | Use in KoDiMARC | Expected local path | Access and license notes |
| --- | --- | --- | --- |
| Korean Wikipedia / KoWiki | Step1 weak supervision from adjacent sentence pairs | `data/raw/kowiki/extracted/` | Obtain a Korean Wikipedia dump from [Wikimedia Dumps](https://dumps.wikimedia.org/kowiki/), then extract plain text before running Step1. Record the exact dump date used for reproduction. |
| KorNLI | Step2 NLI supervision | `data/raw/kornli/` | KorNLI is distributed through the [KorNLU dataset repository](https://github.com/kakaobrain/KorNLUDatasets). Keep the original split names and follow the dataset license/citation terms. |
| AI Malpyeong logical-relation data | Step2 logical-relation supervision | `data/raw/ai_malpyeong/` | Obtain the Korean sentence-pair logical-relation task data through the National Institute of Korean Language [AI Malpyeong platform](https://kli.korean.go.kr/benchmark/home.do). Follow the platform access and redistribution terms. |
| `kakaocorp/kanana-1.5-8b-instruct-2505` or compatible model | Step1 generation and Step2 backbone | local Hugging Face cache or model path | The example configs use the [Hugging Face model identifier](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505). Follow the model card license and usage terms. |

For a submitted reproduction package, record the exact external dataset versions, download dates, and access URLs in the manuscript data availability statement or in a release note accompanying this repository.

The repository includes only toy examples under `data/sample/`:
- `data/sample/step1_sample.jsonl`: example Step1 weak-supervision rows with sentence pairs, detected markers, marker-removed hypotheses, and marker categories.
- `data/sample/step2_sample.jsonl`: example Step2 rows with premise/hypothesis text, task labels, and Step1 top-k marker predictions.

The sample files are for schema inspection only. They are not sufficient to reproduce the manuscript results.

Recommended local data layout:
```text
data/
├── raw/
│   ├── ai_malpyeong/
│   ├── kornli/
│   └── kowiki/extracted/
└── processed/
    ├── kowiki/
    └── multitask_nli/final/
```

Create this layout with:
```bash
python scripts/setup_data_layout.py --root .
```

## Code Information
The implementation is organized as follows:
```text
KoDiMARC/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── step1/
│   └── step2/
├── data/
│   └── sample/
├── docs/
├── scripts/
│   ├── step1/
│   └── step2/
└── src/kodimarc/
    ├── common/
    ├── step1/
    └── step2/
```

Important code paths:
- `src/kodimarc/common/markers.py`: discourse marker lexicon, marker categories, and relation/category mappings.
- `scripts/step1/`: KoWiki sentence-pair construction, marker detection/removal, Step1 SFT data generation, Step1 training, and top-k marker scoring.
- `src/kodimarc/step1/`: Step1 model, dataset, and marker-detection support code.
- `scripts/step2/`: KorNLI/AI Malpyeong preparation, final Step2 JSONL construction, Step2 training, and evaluation.
- `src/kodimarc/step2/`: Step2 dataset loading, model, training loop, objectives, memory bank, metrics, and evaluation utilities.
- `configs/`: editable YAML configuration files used by the public workflows.
- `docs/step1.md`, `docs/step2.md`, and `docs/reproducibility.md`: expanded workflow notes.

## Requirements
Install dependencies in a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The required Python packages are listed in `requirements.txt`:
```text
torch
transformers
peft
bitsandbytes
PyYAML
tqdm
ujson
kss
sentencepiece
unsloth
```

Full-scale training uses large language models and is expected to require a CUDA-capable GPU environment. Users should adjust batch size, gradient accumulation, quantization, and model paths in the YAML configs according to their local hardware.

### Reference Environment
The lightweight README smoke test was checked in the following local environment:
- Operating system: Linux `5.15.0-139-generic` on `x86_64`.
- Python: `3.12.4`.
- GPU hardware available during validation: 4 x NVIDIA GeForce RTX 3090, 24 GiB each.
- NVIDIA driver: `535.183.01`.

Package versions observed in the lightweight validation environment:
- `torch`: `2.6.0`
- `transformers`: `4.49.0.dev0`
- `PyYAML`: `6.0.3`
- `tqdm`: `4.66.4`
- `ujson`: `5.10.0`
- `kss`: `6.0.6`

Full Step1/Step2 model training additionally requires the complete package set in `requirements.txt`, including LoRA/quantization-related packages such as `peft`, `bitsandbytes`, `sentencepiece`, and `unsloth`. For exact manuscript reproduction, report the final training environment, CUDA version, GPU model, package versions, random seed, and YAML config files used for the archived run.

## Methodology
KoDiMARC contains two connected algorithmic stages.

### Step1: Discourse Marker Generator
Step1 builds weakly supervised marker-generation data from KoWiki:
1. Convert extracted KoWiki documents into adjacent sentence pairs.
2. Detect explicit Korean discourse markers with a rule-based lexicon.
3. Remove detected marker spans from the second sentence to create marker-removed hypotheses.
4. Convert the labeled pairs into response-only supervised fine-tuning examples.
5. Fine-tune a Korean instruction-tuned language model to generate the missing discourse marker.
6. Use the trained Step1 model to attach top-k marker candidates and scores to downstream sentence-pair data.

The discourse marker categories are:
- `ADD`
- `CONTRAST`
- `CAUSAL`
- `EXPLAN`
- `CONCESS`
- `COND`
- `EXAMPLE`

### Step2: Marker-aware Multi-task Relation Classifier
Step2 trains a multi-task Korean sentence-pair classifier:
1. Prepare KorNLI and AI Malpyeong examples in a common JSONL format.
2. Attach Step1 top-k discourse marker candidates to each sentence pair.
3. Train a marker-aware classifier with no-marker and predicted-marker views.
4. Optionally sample wrong-marker views for corruption-based robustness objectives.
5. Evaluate saved runs in `NO`, `WITH`, and optional `WRONG` modes.

The classifier supports marker dropout, marker corruption, marker-relation compatibility supervision, supervised contrastive learning, and a WITH-WRONG margin objective. These settings are controlled by the Step2 YAML config files.

## Usage Instructions
The recommended reproducibility workflow is to run each stage explicitly. Commands below assume that raw datasets have already been placed under `data/raw/` and that paths in the YAML configs have been checked for the local machine.

### 1. Prepare the repository layout
```bash
python scripts/setup_data_layout.py --root .
```

### 2. Build Step1 weak-supervision data
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

### 3. Train the Step1 generator
```bash
python scripts/step1/train_step1_generator.py \
  --config configs/step1/step1_sft_example.yaml
```

The trained checkpoint is written to the output directory configured in `configs/step1/step1_sft_example.yaml`.

### 4. Build final Step2 JSONL files
```bash
python scripts/step2/build_final_step2_data.py \
  --kornli-dir data/raw/kornli \
  --ai-malpyeong-dir data/raw/ai_malpyeong \
  --step1-model-name-or-path /path/to/step1_checkpoint \
  --base-model-name-or-path kakaocorp/kanana-1.5-8b-instruct-2505 \
  --output-dir data/processed/multitask_nli/final \
  --top-k 5 \
  --fallback
```

Expected output files:
- `data/processed/multitask_nli/final/train.jsonl`
- `data/processed/multitask_nli/final/dev_nli.jsonl`
- `data/processed/multitask_nli/final/dev_logic.jsonl`
- `data/processed/multitask_nli/final/test_nli.jsonl`
- `data/processed/multitask_nli/final/test_logic.jsonl`
- `data/processed/multitask_nli/final/build_manifest.json`

### 5. Train Step2
```bash
python scripts/step2/train_step2.py \
  --config configs/step2/step2_full_example.yaml
```

The default public Step2 example reflects the marker-sensitive setting used for paper-facing runs, with KL consistency disabled by default.

### 6. Evaluate Step2
```bash
python scripts/step2/evaluate_step2.py \
  --config /path/to/run_dir/config_used.yaml \
  --exp marker_sensitive
```

Evaluation modes:
- `NO`: evaluate the no-marker view.
- `WITH`: evaluate the predicted-marker view using Step1 candidates.
- `WRONG`: evaluate a corrupted or forbidden-category wrong-marker view when enabled.

### Optional convenience wrapper
An optional helper script is provided for users who already understand the full workflow:
```bash
bash scripts/run_example_pipeline.sh
```

For first-time reproduction, the explicit commands above and the detailed notes in `docs/step1.md`, `docs/step2.md`, and `docs/reproducibility.md` are recommended.

## Smoke Test
The repository includes small sample files that allow a quick no-GPU sanity check of the Step1 data-conversion path. This smoke test verifies that the JSONL schema can be read and converted into response-only SFT examples.

```bash
python3 scripts/step1/build_sft_data.py \
  --input-jsonl data/sample/step1_sample.jsonl \
  --train-output /tmp/kodimarc_step1_train.jsonl \
  --valid-output /tmp/kodimarc_step1_valid.jsonl \
  --test-output /tmp/kodimarc_step1_test.jsonl
```

Expected output for the included toy file:
```text
[done] train=2, valid=0, test=0
```

This smoke test is only a format and code-path check. It does not train a model and does not reproduce the manuscript results.

## Reproducibility Notes
Exact numerical reproduction depends on:
- The specific versions and preprocessing state of KoWiki, KorNLI, and AI Malpyeong.
- Local access permissions and licenses for those datasets.
- The base model and tokenizer version.
- The trained Step1 checkpoint used to generate top-k markers.
- GPU hardware, numerical precision, quantization, random seed, and stochastic training behavior.

The public release is designed to reconstruct the raw-data-to-results code path, including Step1 weak-supervision data, Step1 SFT data, final Step2 JSONL files, Step2 checkpoints, and evaluation outputs, when the required external datasets and model weights are available locally.

## Code Availability
The source code is maintained at:
- Repository: `https://github.com/hong0002/KoDiMARC`
- License: MIT License, provided in `LICENSE`

For PeerJ review or publication, the reviewed code version should be archived as an immutable release, for example through Zenodo, Figshare, PeerJ supplementary files, or another repository that provides a persistent identifier.

Before submission, record the final archived version here:
- Archived DOI or persistent URL: to be added after creating the submitted release.
- Release tag or commit hash: to be added after creating the submitted release.
- Archive date: to be added after creating the submitted release.

The archived release should correspond to a clean repository state and should include this README, `LICENSE`, `requirements.txt`, `configs/`, `scripts/`, `src/`, `docs/`, and the sample files under `data/sample/`.

## References
If this repository is used in research, please cite the accompanying KoDiMARC manuscript once citation information is available.

Users should also follow the citation and license requirements of the external resources used in their reproduction:
- Korean Wikipedia / KoWiki dump or extracted text source.
- KorNLI.
- AI Malpyeong logical-relation data.
- The selected Korean base language model, such as `kakaocorp/kanana-1.5-8b-instruct-2505`.

Citation information for the KoDiMARC manuscript will be added after publication.

## License
This repository is released under the MIT License. See `LICENSE` for details.

The external datasets and pretrained model weights are not covered by this repository's MIT License. Users are responsible for complying with the licenses and terms of use of each external resource.

## Contribution Guidelines
This repository is maintained primarily as a reproducibility release for the accompanying manuscript. Contributions that improve documentation, fix reproducibility bugs, or clarify setup instructions are welcome.

For proposed changes, please include:
- A clear description of the issue or improvement.
- The relevant command, config file, or dataset split.
- Expected and observed behavior when reporting bugs.
- Environment details for reproducibility issues, including Python version, package versions, GPU type, and CUDA version when applicable.
