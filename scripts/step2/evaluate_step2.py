#!/usr/bin/env python

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kodimarc.step2.evaluate import evaluate_step2_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved Step2 run using config_used.yaml.')
    parser.add_argument('--config', type=str, required=True, help='Path to run_dir/config_used.yaml.')
    parser.add_argument('--exp', type=str, default=None, help='Experiment name. If omitted, the best experiment is selected.')
    parser.add_argument(
        '--output-subdir',
        type=str,
        default=None,
        help='Optional output subdirectory under the run directory for evaluation artifacts.',
    )
    args = parser.parse_args()
    output_subdir = args.output_subdir
    if output_subdir is None:
        output_subdir = os.path.join('test_eval', args.exp) if args.exp else 'test_eval'
    print(evaluate_step2_checkpoint(args.config, args.exp, output_subdir=output_subdir))


if __name__ == '__main__':
    main()
