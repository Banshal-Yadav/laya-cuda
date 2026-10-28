# Derived from Laya (Apache-2.0). Modified for laya-cuda.
"""Email utilities for cleaning and structuring email inputs."""

import re
from typing import Dict, Optional

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
]
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(
        r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$",
        re.I,
    ),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
]
_DISCLAIMER = re.compile(
    r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
    r"if you (have )?received this (e-?mail|message) in error)",
    re.I,
)


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted email history, signatures and disclaimers."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = []
    buf: list = []
    for line in lines:
        if line.strip():
            buf.append(line)
        else:
            if buf:
                p = " ".join(buf)
                if not _DISCLAIMER.search(p):
                    paragraphs.append(p)
            buf = []
    if buf:
        p = " ".join(buf)
        if not _DISCLAIMER.search(p):
            paragraphs.append(p)
    result = "\n\n".join(paragraphs)
    return result[:max_chars]


def email_state(
    subject: str = "",
    body: str = "",
    sender: Optional[str] = None,
    *,
    clean: bool = True,
    max_chars: int = 3000,
) -> Dict:
    """Build a structured state dict from email fields.

    Args:
        subject: Email subject line.
        body: Email body text (raw).
        sender: Sender address or name (optional).
        clean: If True, strip quoted history and signatures from body.
        max_chars: Truncate cleaned body to this many characters.

    Returns:
        Dict with 'subject', 'body' (and optionally 'sender').
    """
    cleaned = clean_email_body(body, max_chars=max_chars) if clean else (body or "")[:max_chars]
    state: Dict = {"subject": (subject or "").strip(), "body": cleaned}
    if sender:
        state["sender"] = sender.strip()
    return state
