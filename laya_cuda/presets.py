# Derived from Laya (Apache-2.0). Modified for laya-cuda.
"""Ready-to-use question presets for common production decision workflows."""

from typing import Dict, List, Optional


def triage_questions() -> Dict:
    """Preset questions for customer support ticket triage."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "billing_question": "a question about an invoice, plan or payment method",
                "information": "general information, pricing or how-to",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits",
            },
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure or a deadline?",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": [
                "calm and neutral",
                "concerned but civil",
                "clearly annoyed",
                "very angry or using strong language",
            ],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?",
        },
    }


def email_questions(categories: Optional[Dict[str, str]] = None) -> Dict:
    """Preset questions for inbound email triage and threat filtering."""
    categories = categories or {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, integrations",
        "sales": "pricing, demos, new purchases",
        "security": "phishing, scams, account compromise",
        "hr": "hiring, leave, payroll",
        "legal": "contracts, compliance, IP",
        "other": "anything else",
    }
    return {
        "category": {
            "type": "choice",
            "instructions": "Which department or topic does this email belong to?",
            "criteria": categories,
        },
        "is_threat": {
            "type": "noul",
            "instructions": "Does this email contain a threat, phishing attempt, or social engineering?",
        },
        "needs_human": {
            "type": "noul",
            "instructions": "Does this email require a human to respond (vs. automated handling)?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this email?",
            "criteria": ["not urgent", "low urgency", "moderate urgency", "high urgency", "critical"],
        },
    }


def guard_questions(policy: Optional[str] = None) -> Dict:
    """Preset questions for content moderation / policy guard."""
    policy_hint = f" under the policy: {policy}" if policy else ""
    return {
        "violates_policy": {
            "type": "noul",
            "instructions": f"Does `content` violate content policy{policy_hint}?",
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is any policy violation in `content`?",
            "criteria": ["no violation", "minor", "moderate", "serious", "critical"],
        },
        "category": {
            "type": "choice",
            "instructions": "What type of policy issue, if any, does `content` contain?",
            "criteria": {
                "none": "no violation",
                "harassment": "targeted abuse or bullying",
                "hate_speech": "discriminatory language",
                "spam": "unsolicited commercial content",
                "misinformation": "false or misleading claims",
                "explicit": "sexual or graphic content",
                "violence": "violent threats or graphic violence",
                "other": "other policy issue",
            },
        },
    }


def moderation_questions() -> Dict:
    """Alias for guard_questions() - content moderation preset."""
    return guard_questions()


def router_questions(options: Optional[List[str]] = None) -> Dict:
    """Preset questions for routing a message to the right handler."""
    options = options or ["billing", "technical", "sales", "general"]
    return {
        "route": {
            "type": "choice",
            "instructions": "Which team or workflow should handle `message`?",
            "criteria": {opt: "" for opt in options},
        },
        "confidence": {
            "type": "score",
            "instructions": "How confident are you in the routing decision?",
            "criteria": ["very uncertain", "somewhat uncertain", "fairly confident", "very confident"],
        },
    }
