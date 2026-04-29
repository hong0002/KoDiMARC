from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DIRS = [
    'data/raw/kowiki/extracted',
    'data/raw/kornli',
    'data/raw/ai_malpyeong',
    'data/processed/kowiki',
    'data/processed/multitask_nli/final',
    'outputs/step1_sft_example',
    'outputs/marker_sensitive',
    'outputs/marker_sensitive_ablation',
    'logs',
]


def main():
    parser = argparse.ArgumentParser(description='Create the recommended local directory layout for KoDiMARC reproduction.')
    parser.add_argument('--root', type=Path, default=Path('.'), help='Repository root where the directory layout will be created.')
    args = parser.parse_args()

    root = args.root.resolve()
    created = []
    for rel in DEFAULT_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    print('[done] ensured the following directories:')
    for path in created:
        print(f' - {path}')


if __name__ == '__main__':
    main()
