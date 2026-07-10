# Contributing

Contributions should focus on reproducibility bugs, documentation errors, and command/config mismatches.

Before opening an issue or pull request, run:

```bash
python scripts/validate_reproducibility.py
python -m compileall scripts src
```

Reports are most useful when they include:

- the complete command and config path;
- the dataset split and preprocessing state;
- expected and observed behavior;
- Python, PyTorch, Transformers, and PEFT versions; and
- GPU model, driver, and CUDA information for training or evaluation failures.

Do not attach third-party dataset records, model weights, adapters, checkpoints, credentials, or private filesystem paths. Small synthetic examples that reproduce a parsing or schema problem are welcome.
