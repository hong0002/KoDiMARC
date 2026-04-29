from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import kss
import ujson
from tqdm import tqdm


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)



def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



def get_text_from_record(doc: dict) -> str:
    if isinstance(doc, dict):
        if isinstance(doc.get("text"), str):
            return doc["text"]
        sections = doc.get("sections")
        if isinstance(sections, list):
            return "\n".join(section.get("text", "") for section in sections if "text" in section)
    return ""



def iter_sent_pairs(text: str):
    for paragraph in (segment for segment in text.split("\n") if segment.strip()):
        try:
            sentences = [clean(sentence) for sentence in kss.split_sentences(paragraph)]
        except Exception:
            continue
        for premise, hypothesis in zip(sentences, sentences[1:]):
            if 5 <= len(premise) <= 300 and 5 <= len(hypothesis) <= 300:
                yield {"s1": premise, "s2": hypothesis}



def process_file(path: Path):
    pairs = []
    n_docs = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = ujson.loads(line)
            except Exception:
                continue
            text = get_text_from_record(doc)
            if not text:
                continue
            n_docs += 1
            pairs.extend(iter_sent_pairs(text))
    return n_docs, pairs



def main():
    parser = argparse.ArgumentParser(description="Build adjacent Korean sentence pairs from extracted KoWiki JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/kowiki/extracted"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/kowiki/wiki_pairs.jsonl"))
    args = parser.parse_args()

    ensure_parent(args.output_jsonl)
    if not args.output_jsonl.exists():
        args.output_jsonl.touch()

    files = [path for path in args.input_dir.rglob("*") if path.is_file()]
    print(f"[info] scan root: {args.input_dir}")
    print(f"[info] total files: {len(files):,}")

    n_docs = 0
    n_pairs = 0
    buffer = []
    chunk_size = 100_000

    if not files:
        print(f"[warn] no files found under {args.input_dir}")
        return

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, args.output_jsonl.open("w", encoding="utf-8") as out:
        futures = {executor.submit(process_file, path): path for path in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="parallel scan"):
            docs, pairs = future.result()
            n_docs += docs
            n_pairs += len(pairs)
            if pairs:
                buffer.extend(ujson.dumps(pair, ensure_ascii=False) for pair in pairs)
                if len(buffer) >= chunk_size:
                    out.write("\n".join(buffer) + "\n")
                    buffer.clear()
        if buffer:
            out.write("\n".join(buffer) + "\n")

    print(f"[done] docs={n_docs:,}, pairs={n_pairs:,}, out={args.output_jsonl}")


if __name__ == "__main__":
    main()
