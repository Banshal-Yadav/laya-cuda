# Derived from Laya (Apache-2.0). Modified for laya-cuda.
"""Dependency-free language/script detection used to route between Laya checkpoints."""

import re
from typing import Dict, List, Optional, Union

_SCRIPT_RANGES = [
    ("greek",      ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic",   ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("hebrew",     ((0x0590, 0x05FF),)),
    ("arabic",     ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali",    ((0x0980, 0x09FF),)),
    ("gurmukhi",   ((0x0A00, 0x0A7F),)),
    ("gujarati",   ((0x0A80, 0x0AFF),)),
    ("oriya",      ((0x0B00, 0x0B7F),)),
    ("tamil",      ((0x0B80, 0x0BFF),)),
    ("telugu",     ((0x0C00, 0x0C7F),)),
    ("kannada",    ((0x0C80, 0x0CFF),)),
    ("malayalam",  ((0x0D00, 0x0D7F),)),
    ("sinhala",    ((0x0D80, 0x0DFF),)),
    ("thai",       ((0x0E00, 0x0E7F),)),
    ("lao",        ((0x0E80, 0x0EFF),)),
    ("tibetan",    ((0x0F00, 0x0FFF),)),
    ("myanmar",    ((0x1000, 0x109F),)),
    ("georgian",   ((0x10A0, 0x10FF),)),
    ("ethiopic",   ((0x1200, 0x137F),)),
    ("khmer",      ((0x1780, 0x17FF),)),
    ("hangul",     ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana",       ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han",        ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

_STOP: Dict[str, set] = {
    "en": {"the","and","is","are","was","were","to","of","in","for","with","that",
           "this","it","you","have","has","not","but","on","at","be","as","from",
           "will","can","would","there","their","what","which","please","we","i"},
    "fr": {"le","la","les","des","une","est","pour","dans","que","qui","avec","sur",
           "pas","plus","nous","vous","être","cette","mais","sont","ont","aux","ce"},
    "de": {"der","die","das","und","ist","ein","eine","nicht","auf","mit","von","zu",
           "sich","auch","es","an","im","dem","für","werden","wird","sind","haben"},
    "es": {"el","la","los","las","de","en","que","un","una","con","no","por","para",
           "su","este","esta","pero","como","ya","más","se","al","del","esto"},
    "pt": {"o","a","os","as","de","em","que","um","uma","com","não","por","para",
           "se","mas","ao","da","do","mais","uma","também","são","pelo","pela"},
    "it": {"il","la","di","e","che","un","una","con","per","in","non","si","ma",
           "sono","del","della","dei","delle","degli","questo","questa"},
    "nl": {"de","het","een","en","van","in","is","op","dat","zijn","voor","met",
           "niet","aan","er","maar","ook","als","om","door","bij","hem"},
}

_DIACRITICS_NONEN = re.compile(
    r"[àâäæçèéêëîïôœùûüÿ"   # French
    r"ãõáéíóú"                # Portuguese/Spanish
    r"ñ"                      # Spanish
    r"ß"                      # German
    r"åäö"                    # Swedish/Finnish
    r"]"
)

_WORDS = re.compile(r"\b[a-z]{2,}\b")


def detect_script(text: str) -> Optional[str]:
    """Return the name of the first non-Latin script found, or None for Latin/ASCII."""
    for ch in text:
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                return name
    return None


def _latin_lang_hint(text: str) -> Optional[str]:
    """Very rough Latin-script language guess based on stopwords."""
    words = _WORDS.findall(text.lower())
    if not words:
        return None
    counts: Dict[str, int] = {}
    for lang, stops in _STOP.items():
        counts[lang] = sum(1 for w in words if w in stops)
    en_count = counts.get("en", 0)
    best_other = max((c for l, c in counts.items() if l != "en"), default=0)
    if en_count == 0 and best_other == 0:
        return None
    # Require a margin of 2 over English before calling non-English
    if best_other >= en_count + 2:
        best_lang = max(((l, c) for l, c in counts.items() if l != "en"), key=lambda x: x[1])[0]
        if _DIACRITICS_NONEN.search(text):
            return best_lang
        # Without diacritics, ambiguity is high - stay agnostic
        return None
    return "en"


def analyse(text: str) -> Dict:
    """Analyse language/script of text.

    Returns a dict with:
        script: non-Latin script name, or None
        lang: best-effort ISO-639-1 language code, or None
        is_english: True if the text is clearly English
    """
    script = detect_script(text)
    if script is not None:
        return {"script": script, "lang": None, "is_english": False}
    lang = _latin_lang_hint(text)
    return {
        "script": None,
        "lang": lang,
        "is_english": lang == "en" or (lang is None and not _DIACRITICS_NONEN.search(text)),
    }


def is_english(text: str) -> bool:
    return analyse(text)["is_english"]
