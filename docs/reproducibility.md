# Reproducibility Notes

## Data Availability
The repository does not include KoWiki, KorNLI, or AI Malpyeong data because of redistribution constraints. Users must prepare these datasets independently.

## Sample Data
The `data/sample/` directory provides tiny JSONL files only for schema illustration. They are not sufficient to reproduce the paper results.

## Recommended Local Layout
For a clean local setup, initialize the repository layout with:
```bash
python scripts/setup_data_layout.py --root .
```

This prepares a layout such as:
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

## Example Configurations
The configs in `configs/` are example files. They illustrate expected fields and recommended settings, but users should adjust local paths, model names, and hardware-dependent parameters.

## Checkpoints
No model checkpoints are included in the public release. Step1 and Step2 checkpoints must be trained locally.

## High-level Execution Order
1. Create the local directory layout.
2. Build KoWiki sentence pairs.
3. Detect and remove discourse markers.
4. Build Step1 response-only SFT data.
5. Train the Step1 discourse marker generator.
6. Build `data/processed/multitask_nli/final/*.jsonl` from raw KorNLI and AI Malpyeong data, attaching Step1 top-k markers.
7. Train Step2 with the desired config.
8. Evaluate NO / WITH / WRONG views from `config_used.yaml`.

## What The Public Release Can Reconstruct
With local access to:
- KoWiki extracted text,
- KorNLI TSV files,
- AI Malpyeong JSON files,
- and a trained Step1 checkpoint,

the public release is designed to reconstruct:
- Step1 weak supervision data,
- Step1 SFT training data,
- final Step2 split files under `data/processed/multitask_nli/final/`,
- Step2 training runs,
- and saved evaluation outputs.

## Practical Caveat
The public repository provides the code path needed for raw-data-to-results reproduction, but exact numerical reproduction still depends on the original dataset versions, Step1 checkpoint quality, local environment, hardware, and stochastic training behavior.
