#!/usr/bin/env python

import argparse
import json
import os
import shutil
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kodimarc.common.io import ensure_dir, write_csv, write_json
from kodimarc.common.utils import now_seoul_str, sanitize_name
from kodimarc.step2.evaluate import evaluate_step2_checkpoint
from kodimarc.step2.trainer import run_experiment



def write_summary_kv_csv(path: str, metrics: dict, metadata: dict):
    rows = []
    rows.extend(sorted(metadata.items(), key=lambda item: item[0]))
    rows.extend(sorted(metrics.items(), key=lambda item: item[0]))
    write_csv(path, [[key, value] for key, value in rows], header=['key', 'value'])



def main():
    parser = argparse.ArgumentParser(description='Train Step2 experiments from a YAML configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to a Step2 YAML config file.')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    script_path = os.path.abspath(__file__)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    outputs_root = cfg['paths'].get('outputs_root', 'outputs')
    model_name = cfg['model']['name']
    run_name = f"{now_seoul_str()}__{sanitize_name(model_name)}"
    run_dir = os.path.join(outputs_root, run_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, 'experiments'))
    ensure_dir(os.path.join(run_dir, 'tb'))

    used_yaml_path = os.path.join(run_dir, 'config_used.yaml')
    used_script_path = os.path.join(run_dir, 'train_script_used.py')
    shutil.copyfile(config_path, used_yaml_path)
    shutil.copyfile(script_path, used_script_path)

    write_json(
        os.path.join(run_dir, 'run_meta.json'),
        {
            'run_dir': run_dir,
            'model': model_name,
            'created_at': run_name.split('__')[0],
            'config_used': used_yaml_path,
            'script_used': used_script_path,
        },
    )

    experiments = cfg.get('experiments', [])
    if not experiments:
        raise RuntimeError('The YAML config does not define an experiments list.')

    results = []
    for exp in experiments:
        result = run_experiment(cfg, exp, run_dir, used_script_path, used_yaml_path)
        test_metrics = evaluate_step2_checkpoint(
            used_yaml_path,
            exp['name'],
            output_subdir=os.path.join('experiments', exp['name'], 'test'),
        )
        result.update(test_metrics)
        eval_metrics = {'best_score': result.get('best_score', 0.0)}
        eval_dir = os.path.join(result['exp_dir'], 'eval')
        test_dir = os.path.join(result['exp_dir'], 'test')
        ensure_dir(eval_dir)
        ensure_dir(test_dir)

        eval_metadata = {
            'experiment_name': result['exp_name'],
            'selection_metric': result.get('selection_metric', 'acc_weighted'),
            'selection_base_metric': result.get('selection_base_metric', 'acc'),
            'selection_gap_weight': result.get('selection_gap_weight', 0.0),
        }
        test_metadata = dict(eval_metadata)

        write_json(os.path.join(eval_dir, 'metrics.json'), eval_metrics)
        write_json(os.path.join(eval_dir, 'run_metadata.json'), eval_metadata)
        write_summary_kv_csv(os.path.join(eval_dir, 'summary.csv'), eval_metrics, eval_metadata)

        write_json(os.path.join(test_dir, 'run_metadata.json'), test_metadata)
        write_summary_kv_csv(os.path.join(test_dir, 'summary.csv'), test_metrics, test_metadata)
        results.append(result)

    summary_jsonl = os.path.join(run_dir, 'summary.jsonl')
    with open(summary_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    summary_csv = os.path.join(run_dir, 'summary.csv')
    header = [
        'exp_name', 'best_score', 'enable_kl', 'enable_dropout', 'enable_corruption', 'enable_mrel', 'enable_supcon',
        'alpha_v2_ce', 'v2_ce_only_clean', 'with_wrong_margin_weight', 'with_wrong_margin_value',
        'with_wrong_margin_only_clean', 'logic_class_weight_mode', 'logic_class_weight_smoothing',
        'use_marker_aware_head', 'marker_source', 'prompt_style', 'early_stopped', 'no_improve_count', 'patience',
        'min_delta', 'early_stop_start_optim_step', 'mixed_precision', 'quant_type', 'queue_nli_size',
        'queue_logic_size', 'ratio_nli', 'ratio_logic', 'w_nli', 'w_logic', 'selection_weight_no',
        'selection_weight_with', 'selection_weight_wrong', 'selection_metric', 'selection_base_metric',
        'selection_gap_weight', 'microsteps_per_epoch', 'optim_steps_per_epoch', 'strong_wrong_nli',
        'wrong_nli_exclude_topk', 'wrong_nli_exclude_same_category', 'confidence_gating_enabled',
        'confidence_gating_apply_train', 'confidence_gating_apply_eval', 'confidence_gating_temperature',
        'confidence_gating_min_top1_prob', 'confidence_gating_min_top1_gap', 'test_no_nli_acc',
        'test_no_nli_macro_precision', 'test_no_nli_macro_recall', 'test_no_nli_macro_f1', 'test_with_nli_acc',
        'test_with_nli_macro_precision', 'test_with_nli_macro_recall', 'test_with_nli_macro_f1', 'test_no_logic_acc',
        'test_no_logic_macro_precision', 'test_no_logic_macro_recall', 'test_no_logic_macro_f1', 'test_with_logic_acc',
        'test_with_logic_macro_precision', 'test_with_logic_macro_recall', 'test_with_logic_macro_f1',
        'test_wrong_nli_acc', 'test_wrong_nli_macro_precision', 'test_wrong_nli_macro_recall',
        'test_wrong_nli_macro_f1', 'test_wrong_logic_acc', 'test_wrong_logic_macro_precision',
        'test_wrong_logic_macro_recall', 'test_wrong_logic_macro_f1', 'exp_dir', 'tb_dir',
    ]
    rows = [[result.get(key, '') for key in header] for result in results]
    write_csv(summary_csv, rows, header=header)

    results_sorted = sorted(results, key=lambda item: item['best_score'], reverse=True)
    print('\n=== Step2 Summary (sorted) ===')
    for result in results_sorted:
        status = 'ES' if result.get('early_stopped', False) else 'OK'
        print(f"{result['exp_name']:>20s} : {result['best_score']:.6f} ({status})")

    print(f'\nAll artifacts saved under:\n{run_dir}')


if __name__ == '__main__':
    main()
