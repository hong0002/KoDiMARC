from __future__ import annotations

import re
from typing import Dict, List, Optional

# Korean discourse marker lexicon used by both Step1 and Step2.
# The public release keeps the marker inventory aligned with the observed
# weak-supervision corpus and excludes zero-frequency markers.
LEXICON: Dict[str, List[str]] = {
    "ADD": ["그리고", "또한", "게다가", "나아가", "더구나", "아울러"],
    "CONTRAST": ["그러나", "하지만", "반면에", "그런데", "그럼에도", "오히려", "한편", "다만"],
    "CAUSAL": ["그래서", "따라서", "그러므로", "때문에"],
    "EXPLAN": ["즉", "말하자면", "요컨대", "곧"],
    "CONCESS": ["비록"],
    "COND": ["만약", "만일", "라면", "면"],
    "EXAMPLE": ["이를테면", "예컨대"],
}

ALL_LEX_MARKERS: List[str] = sorted({m for markers in LEXICON.values() for m in markers})
LEX_MARKER2CAT: Dict[str, str] = {marker: cat for cat, markers in LEXICON.items() for marker in markers}
LEX_CAT2MARKERS: Dict[str, List[str]] = {cat: sorted(markers) for cat, markers in LEXICON.items()}
CANONICAL_MARKER_BY_CAT: Dict[str, str] = {cat: markers[0] for cat, markers in LEXICON.items()}

LOGIC_KO_TO_CAT: Dict[str, str] = {
    "순접": "ADD",
    "역접": "CONTRAST",
    "양립": "ADD",
    "인과": "CAUSAL",
    "설명": "EXPLAN",
    "양보": "CONCESS",
    "조건": "COND",
    "예시": "EXAMPLE",
}

NLI_TO_CAT: Dict[str, str] = {
    "entailment": "CAUSAL",
    "neutral": "ADD",
    "contradiction": "CONTRAST",
}

ANTI_CATEGORY_MAP: Dict[str, List[str]] = {
    "ADD": ["CONTRAST", "CONCESS", "COND"],
    "CONTRAST": ["ADD", "CAUSAL", "EXPLAN"],
    "CAUSAL": ["CONTRAST", "ADD", "EXAMPLE"],
    "EXPLAN": ["CONTRAST", "ADD", "COND"],
    "CONCESS": ["CAUSAL", "ADD", "EXPLAN"],
    "COND": ["ADD", "EXPLAN", "CAUSAL"],
    "EXAMPLE": ["CONTRAST", "CAUSAL", "COND"],
    "UNK": ["CONTRAST", "CAUSAL", "ADD"],
    "NONE": ["CONTRAST", "CAUSAL", "ADD"],
}

NLI_WRONG_CATEGORY_MAP: Dict[str, List[str]] = {
    "entailment": ["CONTRAST", "CONCESS", "COND"],
    "neutral": ["CONTRAST", "CAUSAL", "EXPLAN"],
    "contradiction": ["ADD", "CAUSAL", "EXPLAN"],
}

DEFAULT_REL_SPECIAL_TEMPLATE = "[REL_{cat}]"
CAT_VOCAB = ["NONE"] + sorted(list(LEXICON.keys())) + ["UNK"]
CAT2ID = {cat: idx for idx, cat in enumerate(CAT_VOCAB)}
ID2CAT = {idx: cat for cat, idx in CAT2ID.items()}
NLI_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2NLI = {idx: label for label, idx in NLI_LABEL2ID.items()}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_marker(marker: str) -> str:
    marker = normalize_whitespace(marker)
    return marker.replace("…", "").strip()


def marker_category(marker: str) -> str:
    return LEX_MARKER2CAT.get(str(marker).strip(), "UNK")


def relation_special_token(cat: str, template: str = DEFAULT_REL_SPECIAL_TEMPLATE) -> str:
    return template.format(cat=str(cat).upper())


def relation_special_tokens(
    template: str = DEFAULT_REL_SPECIAL_TEMPLATE,
    include_none: bool = False,
    include_unk: bool = True,
) -> List[str]:
    cats: List[str] = []
    if include_none:
        cats.append("NONE")
    cats.extend(sorted(LEXICON.keys()))
    if include_unk:
        cats.append("UNK")
    return [relation_special_token(cat, template=template) for cat in cats]


def marker_to_relation_special_token(marker: str, template: str = DEFAULT_REL_SPECIAL_TEMPLATE) -> str:
    return relation_special_token(marker_category(marker), template=template)


def logic_label_to_candidate_categories(logic_label: Optional[str]) -> List[str]:
    if logic_label == "순접":
        return ["CAUSAL", "ADD"]
    if logic_label == "역접":
        return ["CONTRAST", "CONCESS"]
    if logic_label == "양립":
        return ["ADD"]
    return []
