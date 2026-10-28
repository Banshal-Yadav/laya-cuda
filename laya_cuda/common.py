"""Laya prompt construction and calibration, adapted from upstream laya-mlx (Apache-2.0)."""

import json
import math
from typing import Dict, List, Optional, Union

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [
            k if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
            for k, v in crit.items()
        ]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "")
                     else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "")
                    else "yes, the statement holds"),
    ]


def build_prefix(tok, q: Dict, *, max_len: int, head_max_len: int) -> List[int]:
    """Build the [CLS] instruction [SEP] options [MASK] prefix token IDs."""
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    mask_id = tok.mask_token_id

    instr_ids = tok(q.get("instructions", ""), add_special_tokens=False)
    option_texts = render_options(q)
    option_ids = [tok(o, add_special_tokens=False) for o in option_texts]

    # Budget: at most head_max_len for the whole head
    head_budget = min(head_max_len, max_len)
    # [CLS] instr [SEP] opt1 [MASK] opt2 [MASK] ... optN [MASK]
    base_overhead = 1 + 1 + len(option_texts)   # CLS + SEP + N MASKs
    option_token_count = sum(len(o) for o in option_ids)
    instr_budget = head_budget - base_overhead - option_token_count
    instr_ids = instr_ids[:max(instr_budget, 0)]

    ids = [cls_id] + instr_ids + [sep_id]
    markers = []
    for opt_ids in option_ids:
        ids += opt_ids
        markers.append(len(ids))
        ids.append(mask_id)

    return ids, markers


def build_sequence(tok, state_ids: List[int], head_ids: List[int],
                   markers_raw: List[int], *, max_len: int) -> Dict:
    """Combine state tokens + head prefix into a single sequence."""
    # state goes after the head, separated by [SEP]
    sep_id = tok.sep_token_id
    budget = max_len - len(head_ids) - 1  # 1 for trailing SEP
    state_ids = state_ids[:max(budget, 0)]
    ids = head_ids + state_ids + [sep_id]
    # Markers are absolute positions inside head_ids - they don't shift
    return {"ids": ids, "markers": markers_raw, "qtype": 0}


def temp_bucket(temperature: float, num_buckets: int) -> int:
    """Map a calibration temperature to a bucket index (0-based)."""
    if num_buckets <= 1:
        return 0
    return min(int(temperature * num_buckets), num_buckets - 1)


def confidence_from_probs(probs: np.ndarray) -> float:
    """Max probability as confidence."""
    return float(probs.max())


def _normalize_question(q_id: str, q: Dict) -> Dict:
    """Normalize a user-facing question dict to internal format."""
    t = q.get("type", "choice")
    internal: Dict = {"t": t}
    instr = q.get("instructions", "")
    internal["instructions"] = instr

    crit = q.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            crit = {c: "" for c in crit}
        internal["crit"] = crit or {}
    elif t == "score":
        if isinstance(crit, dict):
            crit = list(crit.values())
        internal["crit"] = crit or []
    elif t == "noul":
        if isinstance(crit, list):
            crit = {}
        internal["crit"] = crit or {}
    else:
        raise ValueError(f"Unknown question type {t!r} in question {q_id!r}")
    return internal


def prepare_questions(questions: Dict) -> List[Dict]:
    """Normalize all questions to internal dicts, preserving order."""
    out = []
    for q_id, q in questions.items():
        norm = _normalize_question(q_id, q)
        norm["id"] = q_id
        norm["qtype"] = QTYPES[norm["t"]]
        out.append(norm)
    return out
