from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ujson
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kodimarc.step1.marker_detection import detect_and_strip_marker, label_of_marker, normalize_text



def main():
    parser = argparse.ArgumentParser(description="Detect explicit discourse markers and build marker-removed sentence pairs.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/kowiki/wiki_pairs.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/kowiki/wiki_pairs_labeled.jsonl"))
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_written = 0

    with args.input_jsonl.open(encoding="utf-8") as fin, args.output_jsonl.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="marker detection"):
            example = ujson.loads(line)
            premise = normalize_text(example["s1"])
            hypothesis = normalize_text(example["s2"])

            marker, marker_removed_hypothesis = detect_and_strip_marker(hypothesis)
            if not marker:
                continue

            label = label_of_marker(marker)
            if not label:
                continue

            n_kept += 1
            fout.write(
                ujson.dumps(
                    {
                        "s1": premise,
                        "s2": hypothesis,
                        "s2_no_marker": marker_removed_hypothesis,
                        "label": label,
                        "marker": marker,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"[done] kept={n_kept:,}, wrote={n_written:,}, out={args.output_jsonl}")


if __name__ == "__main__":
    main()
