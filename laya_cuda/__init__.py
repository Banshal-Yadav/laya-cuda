"""Laya typed decisions on NVIDIA CUDA (RTX + CPU) - PyTorch port of laya-mlx."""

from .agent import Agent, load
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .fast_engine import FastDecisionEngine
from .fast_cpu_engine import LayaCPUEngine, OpenJevCPUEngine, load_cpu_laya, load_cpu_open_jev
from .router import route

__all__ = [
    "Agent",
    "load",
    "FastDecisionEngine",
    "LayaCPUEngine",
    "OpenJevCPUEngine",
    "load_cpu_laya",
    "load_cpu_open_jev",
    "detect_language",
    "detect_script",
    "is_english",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "route",
    "clean_email_body",
    "email_state",
]

