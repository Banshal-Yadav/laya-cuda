# Derived from Laya (Apache-2.0). Modified for laya-cuda.
"""Route a request to the Laya checkpoint best suited to it."""

from typing import Any, Dict, List, Optional, Union

from .lang import analyse

BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english":        (BUNDLE_REPO, None),
    "multilingual":   (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}
STANDALONE_MODELS = {
    "english":        "convaiinnovations/laya",
    "multilingual":   "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service":          {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing":        {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents":        {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


class RouteDecision(dict):
    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def route(
    state: Union[str, dict, list],
    questions: Optional[Dict] = None,
    *,
    task: Optional[str] = None,
    auto_task_detection: bool = False,
    lang: Optional[str] = None,
) -> RouteDecision:
    """Decide which checkpoint to use for this request.

    Args:
        state: The input text or structured state.
        questions: The question dict (used for typed-decisions detection).
        task: Explicit task name - 'english', 'multilingual', 'typed-decisions' (or aliases).
        auto_task_detection: If True, detect typed-decisions workflows from question keys.
        lang: Override language detection with an ISO-639-1 code.

    Returns:
        RouteDecision with keys: model, subfolder, reason, script, lang.
    """
    # Explicit task override
    if task is not None:
        name = normalise_name(task)
        repo, sub = DEFAULT_MODELS[name]
        return RouteDecision(model=repo, subfolder=sub, reason="explicit_task",
                             script=None, lang=lang)

    # Auto typed-decisions detection
    if auto_task_detection and questions:
        q_keys = set(questions.keys())
        for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
            if q_keys >= sig:
                repo, sub = DEFAULT_MODELS["typed-decisions"]
                return RouteDecision(model=repo, subfolder=sub,
                                     reason=f"auto_typed_decisions:{wf}",
                                     script=None, lang=lang)

    # Language/script routing
    text = state if isinstance(state, str) else str(state)
    if lang is None:
        detected = analyse(text)
        script = detected["script"]
        detected_lang = detected["lang"]
        is_en = detected["is_english"]
    else:
        script = None
        detected_lang = lang
        is_en = lang == "en"

    if script is not None or not is_en:
        repo, sub = DEFAULT_MODELS["multilingual"]
        reason = f"non_latin_script:{script}" if script else f"non_english_lang:{detected_lang}"
        return RouteDecision(model=repo, subfolder=sub, reason=reason,
                             script=script, lang=detected_lang)

    repo, sub = DEFAULT_MODELS["english"]
    return RouteDecision(model=repo, subfolder=sub, reason="english_latin",
                         script=None, lang=detected_lang or "en")
