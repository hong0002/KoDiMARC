#!/usr/bin/env python3
"""Validate public reproduction files without loading datasets or model weights."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_CONFIG_DIR = ROOT / "configs" / "manuscript"

STEP1_SCHEMA = {
    "model": {
        "base_model",
        "dtype",
        "device",
        "device_map",
        "quant_type",
        "max_length",
        "num_labels",
    },
    "lora": {"enable", "rank", "alpha", "dropout"},
    "data": {"train_path", "valid_path", "max_length", "max_samples"},
    "training": {
        "output_dir",
        "num_epochs",
        "batch_size",
        "gradient_accumulation",
        "lr",
        "warmup_steps",
        "save_steps",
        "logging_steps",
        "seed",
        "eval_steps",
        "dataloader_num_workers",
        "logging_dir",
        "gradient_checkpointing",
        "early_stopping_patience",
        "early_stopping_threshold",
        "load_best_model_at_end",
    },
}

STEP2_SCHEMA = {
    "paths": {
        "train_jsonl",
        "dev_kornli_jsonl",
        "dev_ai_jsonl",
        "test_nli_jsonl",
        "test_logic_jsonl",
        "outputs_root",
    },
    "model": {
        "name",
        "max_seq_length",
        "use_marker_aware_head",
        "use_base_delta_head",
        "quant_type",
        "device_map",
        "dtype",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    },
    "marker": {
        "source",
        "topk_temperature",
        "dropout_nli_start",
        "dropout_nli_end",
        "dropout_logic_start",
        "dropout_logic_end",
        "dropout_unk_boost",
        "logic_forbidden_categories",
    },
    "prompt": {"style"},
    "loss": {
        "lambda_kl",
        "gamma_mrel",
        "beta_supcon",
        "supcon_temperature",
        "supcon_forbidden_neg_weight",
        "task_weight_nli",
        "task_weight_logic",
        "alpha_v2_ce",
        "v2_ce_only_clean",
        "with_wrong_margin",
    },
    "train": {
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "lr",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "mixed_precision",
        "log_every_steps",
        "eval_every_steps",
        "early_stop_patience",
        "early_stop_min_delta",
        "supcon_queue",
        "task_ratio",
    },
    "evaluation": {"enable_wrong_marker_eval"},
    "selection": {"metric", "base_metric", "gap_weight"},
    "selection_weights": {"no", "with", "wrong"},
}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "CONTRIBUTING.md",
    "docs/step1.md",
    "docs/step2.md",
    "docs/reproducibility.md",
    "docs/EXPERIMENT_MANIFEST.md",
    "scripts/step1/train_step1_generator.py",
    "scripts/step2/train_step2.py",
    "scripts/step2/evaluate_step2.py",
    "src/kodimarc/common/markers.py",
    "src/kodimarc/step2/model.py",
    "data/sample/step1_sample.jsonl",
    "data/sample/step2_sample.jsonl",
    "artifacts/results/README.md",
    "artifacts/results/table5_step1_marker_prediction.csv",
    "artifacts/results/table6_main_results.csv",
    "artifacts/results/table7_ablation_results.csv",
    "artifacts/results/table8_no_with_wrong_results.csv",
    "artifacts/results/table9_marker_distribution.csv",
]

LOGIC_LABELS_KO = {"순접", "역접", "양립"}
LOGIC_LABELS_EN = {"forward", "contrastive", "compatible"}
STEP1_LABELS = {"ADD", "CONTRAST", "CAUSAL", "EXPLAN", "CONCESS", "COND", "EXAMPLE"}
NLI_LABELS = {"entailment", "neutral", "contradiction"}
RESULT_ROW_COUNTS = {
    "table5_step1_marker_prediction.csv": 9,
    "table6_main_results.csv": 17,
    "table7_ablation_results.csv": 5,
    "table8_no_with_wrong_results.csv": 3,
    "table9_marker_distribution.csv": 17,
}


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def nested_schema(validation: Validation, path: Path, data: dict[str, Any], schema: dict[str, set[str]]) -> None:
    for section, keys in schema.items():
        value = data.get(section)
        validation.check(isinstance(value, dict), f"{path}: missing mapping '{section}'")
        if not isinstance(value, dict):
            continue
        missing = sorted(keys - set(value))
        validation.check(not missing, f"{path}: {section} missing keys {missing}")


def validate_yaml(validation: Validation) -> None:
    yaml_paths = sorted(ROOT.joinpath("configs").rglob("*.yaml")) + sorted(
        ROOT.joinpath("configs").rglob("*.yml")
    )
    validation.check(bool(yaml_paths), "no YAML files found under configs/")
    for path in yaml_paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            validation.check(False, f"{path}: YAML parse failed: {exc}")
            continue
        validation.check(isinstance(data, dict) and bool(data), f"{path}: YAML root must be a non-empty mapping")
        if not isinstance(data, dict):
            continue

        if path.parent == MANUSCRIPT_CONFIG_DIR and path.name.startswith("step1_"):
            nested_schema(validation, path, data, STEP1_SCHEMA)
        if path.parent == MANUSCRIPT_CONFIG_DIR and path.name.startswith("step2_"):
            validation.check("seed" in data and "device" in data, f"{path}: missing seed or device")
            validation.check(isinstance(data.get("experiments"), list) and data["experiments"], f"{path}: experiments must be non-empty")
            nested_schema(validation, path, data, STEP2_SCHEMA)
            marker = data.get("marker", {})
            has_corruption_schedule = all(
                key in marker
                for key in (
                    "corrupt_prob_nli_start",
                    "corrupt_prob_nli_end",
                    "corrupt_prob_logic_start",
                    "corrupt_prob_logic_end",
                )
            )
            has_constant_corruption = "corrupt_prob_nli" in marker and "corrupt_prob_logic" in marker
            validation.check(
                has_corruption_schedule or has_constant_corruption,
                f"{path}: marker corruption probabilities are incomplete",
            )
            logic_map = marker.get("logic_forbidden_categories", {})
            validation.check(
                isinstance(logic_map, dict) and set(logic_map) == LOGIC_LABELS_KO,
                f"{path}: LOGIC mapping must contain only 순접, 역접, 양립",
            )


def parse_jsonl(validation: Validation, relative_path: str) -> list[dict[str, Any]]:
    path = ROOT / relative_path
    rows: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            validation.check(isinstance(value, dict), f"{path}:{line_no}: row must be a JSON object")
            if isinstance(value, dict):
                rows.append(value)
    except Exception as exc:
        validation.check(False, f"{path}: JSONL parse failed: {exc}")
    validation.check(bool(rows), f"{path}: no JSONL rows found")
    return rows


def validate_samples(validation: Validation) -> None:
    step1_rows = parse_jsonl(validation, "data/sample/step1_sample.jsonl")
    for index, row in enumerate(step1_rows, start=1):
        validation.check(
            {"s1", "s2", "s2_no_marker", "label", "marker"} <= set(row),
            f"step1 sample row {index}: missing required fields",
        )
        validation.check(row.get("label") in STEP1_LABELS, f"step1 sample row {index}: invalid Step1 label")

    step2_rows = parse_jsonl(validation, "data/sample/step2_sample.jsonl")
    for index, row in enumerate(step2_rows, start=1):
        validation.check(
            {
                "id",
                "source",
                "premise",
                "hypothesis",
                "nli_label",
                "logic_label",
                "step1_topk_markers",
                "step1_topk_scores",
            }
            <= set(row),
            f"step2 sample row {index}: missing required fields",
        )
        nli_label = row.get("nli_label")
        logic_label = row.get("logic_label")
        validation.check(
            (nli_label in NLI_LABELS and logic_label is None)
            or (nli_label is None and logic_label in (LOGIC_LABELS_KO | LOGIC_LABELS_EN)),
            f"step2 sample row {index}: expected exactly one valid NLI or LOGIC label",
        )
        validation.check(
            len(row.get("step1_topk_markers", [])) == len(row.get("step1_topk_scores", [])),
            f"step2 sample row {index}: marker and score list lengths differ",
        )


def validate_results(validation: Validation) -> None:
    for path in sorted(ROOT.joinpath("artifacts", "results").glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            validation.check(bool(rows), f"{path}: CSV must contain a header and data")
            expected_count = RESULT_ROW_COUNTS.get(path.name)
            validation.check(
                expected_count is not None and len(rows) == expected_count,
                f"{path}: expected {expected_count} data rows, found {len(rows)}",
            )
            if path.name == "table8_no_with_wrong_results.csv":
                validation.check(
                    {row.get("view") for row in rows} == {"NO", "WITH", "WRONG"},
                    f"{path}: diagnostic views must be NO, WITH, and WRONG",
                )
            if path.name == "table9_marker_distribution.csv":
                logic_labels = {
                    row.get("gold_label") for row in rows if row.get("dataset") == "AI Malpyeong"
                }
                validation.check(
                    logic_labels == LOGIC_LABELS_EN,
                    f"{path}: AI Malpyeong labels must be forward, contrastive, and compatible",
                )
        except Exception as exc:
            validation.check(False, f"{path}: CSV parse failed: {exc}")


def validate_readme_commands(validation: Validation) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"^\s*python(?:3)?\s+([^\s\\]+)", readme, flags=re.MULTILINE):
        command_path = match.group(1)
        if command_path.startswith("-"):
            continue
        validation.check((ROOT / command_path).is_file(), f"README command path does not exist: {command_path}")
    for match in re.finditer(r"--config\s+(configs/[^\s\\]+)", readme):
        config_path = match.group(1).rstrip("`.,")
        validation.check((ROOT / config_path).is_file(), f"README config path does not exist: {config_path}")


def validate_public_paths(validation: Validation) -> None:
    targets = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    targets.extend(sorted(ROOT.joinpath("docs").glob("*.md")))
    targets.extend(sorted(ROOT.joinpath("artifacts").rglob("*.md")))
    targets.extend(sorted(ROOT.joinpath("configs").rglob("*.yaml")))
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        validation.check(
            "/data1/jihong" not in text and "/home/jihong" not in text,
            f"{path}: contains a local absolute path",
        )


def main() -> int:
    validation = Validation()
    for relative_path in REQUIRED_FILES:
        validation.check((ROOT / relative_path).is_file(), f"missing repository file: {relative_path}")

    validate_yaml(validation)
    validate_samples(validation)
    validate_results(validation)
    validate_readme_commands(validation)
    validate_public_paths(validation)

    if validation.errors:
        print(f"Reproducibility validation failed ({len(validation.errors)} errors):")
        for error in validation.errors:
            print(f"- {error}")
        return 1

    print(f"Reproducibility validation passed ({validation.checks} checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
