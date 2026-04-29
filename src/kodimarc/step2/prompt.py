from typing import Optional

# ============================================================
# 2) Prompt
# ============================================================
def _normalize_task_name(task: Optional[str]) -> str:
    if task is None:
        return "generic"
    task_l = str(task).strip().lower()
    if task_l in ("nli", "natural_language_inference"):
        return "nli"
    if task_l in ("logic", "relation", "discourse"):
        return "logic"
    return "generic"


def build_prompt(
    premise: str,
    hypothesis: str,
    marker: Optional[str] = None,
    task: Optional[str] = None,
    style: str = "legacy",
) -> str:
    task_name = _normalize_task_name(task)
    style_name = str(style).strip().lower()

    if style_name == "candidate_marker_hypothesis":
        task_header = {
            "nli": "[과제] 자연어추론",
            "logic": "[과제] 문장관계판별",
            "generic": "[과제] 문장 관계 분류",
        }[task_name]
        label_space = {
            "nli": "함의 / 중립 / 모순",
            "logic": "순접 / 양립 / 역접",
            "generic": "적절한 관계 라벨",
        }[task_name]
        marker_text = marker if marker is not None else "없음"
        return (
            "### 입력\n"
            f"{task_header}\n"
            f"[P] {premise}\n"
            f"[H] {hypothesis}\n"
            f"[M] 후보 담화표지: {marker_text}\n"
            "### 작업\n"
            "후보 담화표지는 두 문장 관계에 대한 가설이며, 정답이 아닐 수 있다.\n"
            "문장 내용과 후보 담화표지의 적합성을 함께 고려하여 두 문장의 관계를 판단하라.\n"
            "후보 담화표지가 부적절하더라도 문장 내용에 따라 최종 판단하라.\n"
            "### 라벨\n"
            f"{label_space}\n"
        )

    marker_part = "" if marker is None else f"[M] {marker}\n"
    return (
        "### 입력\n"
        f"[P] {premise}\n"
        f"{marker_part}"
        f"[H] {hypothesis}\n"
        "### 작업\n"
        "두 문장 사이의 관계를 분류하라.\n"
    )
