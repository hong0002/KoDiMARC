from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = PROJECT_ROOT / 'scripts' / 'step2' / 'prepare_step2_data.py'
SCORE_SCRIPT = PROJECT_ROOT / 'scripts' / 'step1' / 'score_topk_markers.py'


def run_command(command):
    print('[run]', ' '.join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Build final Step2 train/dev/test JSONL files with Step1 top-k marker fields attached.'
    )
    parser.add_argument('--kornli-dir', type=Path, required=True, help='Directory containing KorNLI TSV files.')
    parser.add_argument('--ai-malpyeong-dir', type=Path, required=True, help='Directory containing AI Malpyeong JSON files.')
    parser.add_argument('--step1-model-name-or-path', type=str, required=True, help='Trained Step1 checkpoint or adapter path.')
    parser.add_argument('--base-model-name-or-path', type=str, default=None, help='Base model path when the Step1 checkpoint is a LoRA adapter.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/multitask_nli/final'))
    parser.add_argument('--device-map', type=str, default='auto')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--merge-lora', action='store_true')
    parser.add_argument('--fallback', action='store_true')
    parser.add_argument('--keep-intermediate', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate_dir = Path(
        tempfile.mkdtemp(prefix='step2_base_', dir=str(output_dir.parent.resolve()))
    )

    prepare_cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        '--kornli-dir', str(args.kornli_dir),
        '--ai-malpyeong-dir', str(args.ai_malpyeong_dir),
        '--output-dir', str(intermediate_dir),
    ]
    run_command(prepare_cmd)

    split_names = ['train', 'dev_nli', 'dev_logic', 'test_nli', 'test_logic']
    for split_name in split_names:
        input_jsonl = intermediate_dir / f'{split_name}.jsonl'
        output_jsonl = output_dir / f'{split_name}.jsonl'
        score_cmd = [
            sys.executable,
            str(SCORE_SCRIPT),
            '--input-jsonl', str(input_jsonl),
            '--output-jsonl', str(output_jsonl),
            '--model-name-or-path', args.step1_model_name_or_path,
            '--device-map', args.device_map,
            '--dtype', args.dtype,
            '--top-k', str(args.top_k),
            '--max-new-tokens', str(args.max_new_tokens),
            '--attach-to', 'hypothesis',
        ]
        if args.base_model_name_or_path:
            score_cmd.extend(['--base-model-name-or-path', args.base_model_name_or_path])
        if args.trust_remote_code:
            score_cmd.append('--trust-remote-code')
        if args.merge_lora:
            score_cmd.append('--merge-lora')
        if args.fallback:
            score_cmd.append('--fallback')
        run_command(score_cmd)

    manifest = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'output_dir': str(output_dir),
        'intermediate_dir': str(intermediate_dir),
        'splits': [f'{name}.jsonl' for name in split_names],
        'step1_model_name_or_path': args.step1_model_name_or_path,
        'base_model_name_or_path': args.base_model_name_or_path,
        'dtype': args.dtype,
        'top_k': args.top_k,
        'max_new_tokens': args.max_new_tokens,
    }
    with (output_dir / 'build_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.keep_intermediate:
        for path in intermediate_dir.glob('*'):
            if path.is_file():
                path.unlink()
        intermediate_dir.rmdir()

    print(f'[done] final Step2 files saved under: {output_dir}')


if __name__ == '__main__':
    main()
