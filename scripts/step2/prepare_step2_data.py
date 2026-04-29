from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List



def load_ai_malpyeong(path: Path, split_name: str) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for example in data:
        rows.append(
            {
                "id": example.get("id"),
                "source": f"ai_malpyeong-{split_name}",
                "premise": example["input"]["front"].strip(),
                "hypothesis": example["input"]["back"].strip(),
                "logic_label": (example.get("output") or "").strip() or None,
                "nli_label": None,
            }
        )
    return rows



def load_kornli_tsv(path: Path, source_name: str) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            label = (row.get("gold_label") or "").strip()
            if label not in {"entailment", "contradiction", "neutral"}:
                continue
            premise = (row.get("sentence1") or "").strip()
            hypothesis = (row.get("sentence2") or "").strip()
            if not premise or not hypothesis:
                continue
            rows.append(
                {
                    "id": f"{source_name}-{idx}",
                    "source": source_name,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "logic_label": None,
                    "nli_label": label,
                }
            )
    return rows



def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Build public-format Step2 JSONL files from KorNLI and AI Malpyeong data.")
    parser.add_argument("--kornli-dir", type=Path, required=True, help="Directory containing KorNLI TSV files.")
    parser.add_argument("--ai-malpyeong-dir", type=Path, required=True, help="Directory containing AI Malpyeong JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/multitask_nli"))
    args = parser.parse_args()

    train_rows = []
    train_rows.extend(load_ai_malpyeong(args.ai_malpyeong_dir / "train.json", "train"))
    train_rows.extend(load_kornli_tsv(args.kornli_dir / "multinli.train.ko.tsv", "korNLI-multinli-train"))
    train_rows.extend(load_kornli_tsv(args.kornli_dir / "snli_1.0_train.ko.tsv", "korNLI-snli-train"))

    dev_nli = load_kornli_tsv(args.kornli_dir / "xnli.dev.ko.tsv", "korNLI-xnli-dev")
    test_nli = load_kornli_tsv(args.kornli_dir / "xnli.test.ko.tsv", "korNLI-xnli-test")
    dev_logic = load_ai_malpyeong(args.ai_malpyeong_dir / "dev.json", "dev")
    test_logic = load_ai_malpyeong(args.ai_malpyeong_dir / "test.json", "test")

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "dev_nli.jsonl", dev_nli)
    write_jsonl(args.output_dir / "test_nli.jsonl", test_nli)
    write_jsonl(args.output_dir / "dev_logic.jsonl", dev_logic)
    write_jsonl(args.output_dir / "test_logic.jsonl", test_logic)

    print(f"[done] train={len(train_rows):,}, dev_nli={len(dev_nli):,}, test_nli={len(test_nli):,}, dev_logic={len(dev_logic):,}, test_logic={len(test_logic):,}")


if __name__ == "__main__":
    main()
