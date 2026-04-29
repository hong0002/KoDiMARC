from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kodimarc.common.markers import ALL_LEX_MARKERS, LEXICON, logic_label_to_candidate_categories

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover
    PeftConfig = None
    PeftModel = None

BAD_PATTERNS = [r"</s>", r"<s>", r"\[/?INST\]", r"\[/?SYS\]", r"<\|.*?\|>"]



def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)



def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)



def build_prompt(premise: str, hypothesis: str, logic_label: Optional[str]) -> str:
    logic_text = logic_label if logic_label is not None else "알 수 없음"
    candidate_categories = logic_label_to_candidate_categories(logic_label)
    if candidate_categories:
        candidate_markers: List[str] = []
        for category in candidate_categories:
            candidate_markers.extend(LEXICON.get(category, []))
        candidate_markers = list(dict.fromkeys(candidate_markers))[:12]
        hint = f"가능한 담화표지 후보 예시: {', '.join(candidate_markers)}\n"
    else:
        hint = ""

    return (
        "너는 한국어 문장쌍에 적절한 담화표지(접속어)를 1개만 제안하는 모델이야.\n"
        "규칙:\n"
        "1) 출력은 반드시 한 줄로, 형식은 'MARKER: <담화표지>'\n"
        "2) <담화표지>는 짧은 접속어/담화표지 1개만\n"
        "3) 다른 설명/문장 생성 금지\n"
        f"{hint}\n"
        f"premise: {premise}\n"
        f"hypothesis: {hypothesis}\n"
        f"logic_label: {logic_text}\n"
        "MARKER:"
    )



def normalize_generation_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if "MARKER:" in text:
        text = text.split("MARKER:", 1)[-1].strip()
    for pattern in BAD_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = text.strip().strip("\"\'“”‘’ ")
    text = re.sub(r"\s+", " ", text).strip()
    return text



def extract_marker(raw: str, logic_label: Optional[str] = None) -> str:
    text = normalize_generation_text(raw)
    if not text:
        return ""

    candidate_markers: List[str] = []
    for category in logic_label_to_candidate_categories(logic_label):
        candidate_markers.extend(LEXICON.get(category, []))
    candidate_markers = sorted(set(candidate_markers), key=len, reverse=True)

    def find_best(marker_list: List[str]) -> str:
        best = ""
        best_pos = 10**9
        for marker in marker_list:
            pos = text.find(marker)
            if pos != -1 and pos < best_pos:
                best = marker
                best_pos = pos
        return best

    marker = find_best(candidate_markers) if candidate_markers else ""
    if not marker:
        marker = find_best(sorted(ALL_LEX_MARKERS, key=len, reverse=True))
    return marker if marker in ALL_LEX_MARKERS else ""



def is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    )



def load_model_and_tokenizer(
    model_name_or_path: str,
    base_model_name_or_path: Optional[str],
    device_map: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    merge_lora: bool,
) -> Tuple[torch.nn.Module, Any]:
    model_path = Path(model_name_or_path).resolve()

    if is_peft_adapter_dir(model_path):
        if PeftConfig is None or PeftModel is None:
            raise ImportError("peft is required to load a LoRA adapter directory.")
        peft_cfg = PeftConfig.from_pretrained(str(model_path))
        base_id = base_model_name_or_path or getattr(peft_cfg, "base_model_name_or_path", None)
        if not base_id:
            raise ValueError("A LoRA adapter directory was detected, but no base model could be resolved.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_remote_code)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        if merge_lora:
            model = model.merge_and_unload()
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    if getattr(cfg, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_candidate_markers(model, tokenizer, prompt: str, top_k: int, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

    generate_kwargs = {"max_new_tokens": max_new_tokens, "num_return_sequences": top_k}
    if top_k > 1:
        generate_kwargs.update({"do_sample": True, "temperature": 0.8, "top_p": 0.95})
    else:
        generate_kwargs.update({"do_sample": False})

    output_ids = model.generate(**inputs, **generate_kwargs)
    prompt_len = int(inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)[0].item())
    generations = []
    for sequence in output_ids:
        completion_ids = sequence[prompt_len:]
        generations.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

    markers = [extract_marker(text) for text in generations]
    markers = [marker for marker in markers if marker]
    if not markers:
        return [], [], generations

    counts = Counter(markers)
    ranked = counts.most_common(top_k)
    total = sum(counts.values())
    top_markers = [marker for marker, _ in ranked]
    top_scores = [round(count / total, 6) for _, count in ranked]
    return top_markers, top_scores, generations



def main():
    parser = argparse.ArgumentParser(description="Attach Step1 top-k discourse marker candidates to sentence-pair JSONL data.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--base-model-name-or-path", type=str, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attach-to", type=str, default="hypothesis", choices=["hypothesis", "premise", "none"])
    parser.add_argument("--pair-sep", type=str, default=" [SEP] ")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        base_model_name_or_path=args.base_model_name_or_path,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        merge_lora=args.merge_lora,
    )

    fallback_map = {"순접": "따라서", "역접": "그러나", "양립": "또한"}
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = count_jsonl_lines(input_path)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(read_jsonl(input_path), total=total_lines, desc="Scoring Step1 top-k markers", unit="rows"):
            premise = (row.get("premise") or row.get("s1") or "").strip()
            hypothesis = (row.get("hypothesis") or row.get("s2_no_marker") or row.get("s2") or "").strip()
            logic_label = row.get("logic_label")
            if not premise or not hypothesis:
                skipped += 1
                continue

            prompt = build_prompt(premise, hypothesis, logic_label)
            top_markers, top_scores, raw_generations = generate_candidate_markers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                top_k=max(int(args.top_k), 1),
                max_new_tokens=args.max_new_tokens,
            )

            if (not top_markers) and args.fallback:
                fallback_marker = fallback_map.get(logic_label, "")
                if fallback_marker:
                    top_markers = [fallback_marker]
                    top_scores = [1.0]

            top1_marker = top_markers[0] if top_markers else ""
            top1_score = top_scores[0] if top_scores else 0.0

            if top_markers:
                step1_status = "ok"
            elif raw_generations:
                step1_status = "no_marker"
            else:
                step1_status = "empty_generation"

            row_out = dict(row)
            row_out["step1_topk_markers"] = top_markers
            row_out["step1_topk_scores"] = top_scores
            row_out["step1_top1_marker"] = top1_marker
            row_out["step1_top1_score"] = top1_score
            row_out["marker_raw_generations"] = raw_generations
            row_out["step1_status"] = step1_status
            row_out["step1_s2_top1"] = f"{top1_marker}, {hypothesis}" if top1_marker else hypothesis
            row_out["step1_s2_topk"] = [f"{marker}, {hypothesis}" for marker in top_markers]

            if args.attach_to == "hypothesis":
                hypothesis_marked = f"{top1_marker} {hypothesis}".strip() if top1_marker else hypothesis
                premise_marked = premise
            elif args.attach_to == "premise":
                premise_marked = f"{premise} {top1_marker}".strip() if top1_marker else premise
                hypothesis_marked = hypothesis
            else:
                premise_marked = premise
                hypothesis_marked = hypothesis

            row_out["premise_marked"] = premise_marked
            row_out["hypothesis_marked"] = hypothesis_marked
            row_out["pair_marked"] = f"{premise_marked}{args.pair_sep}{hypothesis_marked}".strip()
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] saved -> {output_path} (written={written:,}, skipped={skipped:,})")


if __name__ == "__main__":
    main()
