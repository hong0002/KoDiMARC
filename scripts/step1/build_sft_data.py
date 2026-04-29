from __future__ import annotations

import argparse
import random
from pathlib import Path

import ujson
from tqdm import tqdm

VALID_RATIO = 0.05
TEST_RATIO = 0.05



def make_example(example):
    premise = example["s1"]
    hypothesis_no_marker = example["s2_no_marker"]
    label = example["label"]
    marker = example["marker"]

    instruction = "아래 두 문장을 자연스럽게 이어 주는 한국어 담화표지를 한 단어로 생성하세요."
    input_text = (
        f"문장1: {premise}\n"
        f"문장2(표지 제거): {hypothesis_no_marker}\n"
        "출력 형식: 담화표지 한 단어만 출력하세요."
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": marker,
        "label": label,
        "marker": marker,
    }



def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as fout:
        for item in items:
            fout.write(ujson.dumps(item, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Build response-only Step1 SFT data from explicit marker sentence pairs.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/kowiki/wiki_pairs_labeled.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/kowiki/dp_sft_train.jsonl"))
    parser.add_argument("--valid-output", type=Path, default=Path("data/processed/kowiki/dp_sft_valid.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/processed/kowiki/dp_sft_test.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data = []
    with args.input_jsonl.open(encoding="utf-8") as fin:
        for line in tqdm(fin, desc="load"):
            example = ujson.loads(line)
            if len(example["s1"]) < 5 or len(example["s1"]) > 300:
                continue
            if len(example["s2_no_marker"]) < 5 or len(example["s2_no_marker"]) > 300:
                continue
            data.append(make_example(example))

    random.shuffle(data)
    n_total = len(data)
    n_valid = int(n_total * VALID_RATIO)
    n_test = int(n_total * TEST_RATIO)
    test_data = data[:n_test]
    valid_data = data[n_test : n_test + n_valid]
    train_data = data[n_test + n_valid :]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_output, train_data)
    write_jsonl(args.valid_output, valid_data)
    write_jsonl(args.test_output, test_data)

    print(f"[done] train={len(train_data):,}, valid={len(valid_data):,}, test={len(test_data):,}")


if __name__ == "__main__":
    main()
