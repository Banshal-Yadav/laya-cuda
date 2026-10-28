"""Public PyTorch inference runtime - prompt and result formats follow upstream Laya."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .common import (
    QTYPES,
    QTYPE_NAMES,
    build_prefix,
    build_sequence,
    prepare_questions,
    render_options,
    serialize_state,
)
from .model import DecisionModel, EncoderConfig
from .tokenizer import Tokenizer

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None) -> Path:
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the repo")
    path = Path(str(model_id_or_path)).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if value.startswith(("/", "./", "../", "~")) or isinstance(model_id_or_path, Path):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        patterns = [
            prefix + n for n in (
                "model.safetensors", "rl_agent_config.json",
                "encoder/config.json", "tokenizer/*", "config.json",
            )
        ]
        path = Path(snapshot_download(value, token=token, revision=revision, allow_patterns=patterns))
    if subfolder:
        path = path / subfolder
    if not (path / "rl_agent_config.json").is_file():
        raise FileNotFoundError(f"Not a Laya checkpoint (missing rl_agent_config.json): {path}")
    return path


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def collate_items(items, pad_id, *, pad_to_multiple=None, max_length=None):
    n = len(items)
    length = max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if max_length is not None:
            length = min(length, max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    batch = {
        "input_ids":      np.full((n, length), pad_id, dtype=np.int32),
        "attention_mask": np.zeros((n, length), dtype=bool),
        "marker_pos":     np.zeros((n, count), dtype=np.int32),
        "marker_mask":    np.zeros((n, count), dtype=bool),
        "qtype":          np.array([item["qtype"] for item in items], dtype=np.int32),
    }
    for i, item in enumerate(items):
        sl, mc = len(item["ids"]), len(item["markers"])
        batch["input_ids"][i, :sl] = item["ids"]
        batch["attention_mask"][i, :sl] = True
        batch["marker_pos"][i, :mc] = item["markers"]
        batch["marker_mask"][i, :mc] = True
    return batch


# ---------------------------------------------------------------------------
# Result decoding
# ---------------------------------------------------------------------------

def _decode_result(q, scores_row, act_row, marker_mask_row, temperatures):
    """Decode one row of model output into the upstream result schema.

    Args:
        q:               normalised question dict (has 't', 'id', 'crit', etc.)
        scores_row:      np.ndarray [M] - raw scalar score at each marker
        act_row:         np.ndarray [M, 3] - action logits at each marker
        marker_mask_row: np.ndarray [M] bool - which markers are real
        temperatures:    np.ndarray [3] - per-type calibration temperatures
    """
    options = render_options(q)
    num_opts = len(options)
    t = q["t"]
    qtype = QTYPES[t]

    # Active marker positions for this question (one per option)
    active = np.where(marker_mask_row)[0][:num_opts]
    raw_scores = scores_row[active]   # [num_opts]

    # Apply per-type temperature calibration
    temp = float(temperatures[qtype])
    temp = max(temp, 1e-8)
    logits = raw_scores / temp
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0).numpy()

    selected_idx = int(probs.argmax())
    confidence = float(probs.max())

    result: Dict[str, Any] = {
        "probabilities": {opt: float(probs[i]) for i, opt in enumerate(options)},
        "confidence": confidence,
    }
    if t == "choice":
        result["answer"] = options[selected_idx]
    elif t == "score":
        result["answer"] = selected_idx
        result["expected_score"] = float((probs * np.arange(num_opts)).sum())
    elif t == "noul":
        result["answer"] = bool(selected_idx == 1)
        result["p_true"] = float(probs[1]) if num_opts > 1 else float(probs[0])
    return result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """PyTorch CUDA inference runtime for Laya typed decision models."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        *,
        dtype: str = "float16",
        revision: Optional[str] = None,
        batch_size: int = 16,
        compile: bool = False,
        pad_to_multiple: Optional[int] = 8,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = DTYPES[dtype]
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple

        # Resolve checkpoint
        path = resolve_model(model_id_or_path, token=token, subfolder=subfolder, revision=revision)

        # Load configs
        self.cfg = json.loads((path / "rl_agent_config.json").read_text(encoding="utf-8"))

        enc_cfg_path = path / "encoder" / "config.json"
        if not enc_cfg_path.is_file():
            # Some checkpoints store encoder config at root as config.json
            enc_cfg_path = path / "config.json"
        enc_cfg_raw = json.loads(enc_cfg_path.read_text(encoding="utf-8"))
        self.enc_cfg = EncoderConfig.from_dict(enc_cfg_raw)

        # Tokenizer
        tok_path = path / "tokenizer"
        if not tok_path.is_dir():
            tok_path = path
        self.tok = Tokenizer(tok_path)

        # Load weights first so we can infer dims
        weights_path = path / "model.safetensors"
        raw = load_safetensors(str(weights_path), device="cpu")

        # Remap scorer sequential indices → named submodule keys
        remap = {
            "scorer.0.weight": "scorer.norm.weight",
            "scorer.0.bias":   "scorer.norm.bias",
            "scorer.1.weight": "scorer.dense.weight",
            "scorer.1.bias":   "scorer.dense.bias",
            "scorer.3.weight": "scorer.out.weight",
            "scorer.3.bias":   "scorer.out.bias",
        }
        mapped = {remap.get(k, k): v for k, v in raw.items()}

        # Build model with exact dims from checkpoint
        self.model = DecisionModel.from_weights(self.enc_cfg, self.cfg, raw)
        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        if missing:
            print(f"[laya-cuda] Warning - missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[laya-cuda] Warning - unexpected keys ({len(unexpected)}): {unexpected[:8]}")

        # Move to device/dtype
        self.model = self.model.to(device=self.device, dtype=self.torch_dtype)
        self.model.eval()

        # Cache temperatures as numpy for decoding
        self._temperatures = self.model.temperature.float().cpu().detach().numpy()

        if compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[laya-cuda] torch.compile enabled")
            except Exception as e:
                print(f"[laya-cuda] torch.compile failed ({e}) - running eager")

        self._warmup()

    def _warmup(self):
        with torch.no_grad():
            B, T, M = 1, 16, 2
            ids  = torch.zeros(B, T, dtype=torch.long, device=self.device)
            mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
            mpos = torch.zeros(B, M, dtype=torch.long, device=self.device)
            mmask= torch.ones(B, M, dtype=torch.bool, device=self.device)
            qt   = torch.zeros(B, dtype=torch.long, device=self.device)
            try:
                self.model(ids, mask, mpos, mmask, qt)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, batch):
        t_ids  = torch.tensor(batch["input_ids"],      dtype=torch.long, device=self.device)
        t_mask = torch.tensor(batch["attention_mask"], dtype=torch.bool, device=self.device)
        t_mpos = torch.tensor(batch["marker_pos"],     dtype=torch.long, device=self.device)
        t_mmask= torch.tensor(batch["marker_mask"],    dtype=torch.bool, device=self.device)
        t_qt   = torch.tensor(batch["qtype"],          dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.torch_dtype):
                    scores, act = self.model(t_ids, t_mask, t_mpos, t_mmask, t_qt)
            else:
                scores, act = self.model(t_ids, t_mask, t_mpos, t_mmask, t_qt)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        return scores.float().cpu().numpy(), act.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, state, questions) -> Dict:
        return self.batch_predict([(state, questions)])[0]

    def batch_predict(self, items) -> List[Dict]:
        out = []
        for i in range(0, len(items), self.batch_size):
            out.extend(self._run_chunk(items[i: i + self.batch_size]))
        return out

    def _run_chunk(self, chunk):
        max_len      = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)

        flat_items, meta = [], []
        for c_idx, (state, questions) in enumerate(chunk):
            norm_qs = prepare_questions(questions)
            state_text = serialize_state(state).replace(self.tok.mask_token, " ")
            state_ids  = self.tok(state_text, add_special_tokens=False)
            for q in norm_qs:
                head_ids, markers = build_prefix(
                    self.tok, q, max_len=max_len, head_max_len=head_max_len)
                seq = build_sequence(self.tok, state_ids, head_ids, markers, max_len=max_len)
                seq["qtype"] = q["qtype"]
                flat_items.append(seq)
                meta.append({"c_idx": c_idx, "q": q})

        if not flat_items:
            return [{} for _ in chunk]

        batch = collate_items(
            flat_items, pad_id=self.tok.pad_token_id,
            pad_to_multiple=self.pad_to_multiple, max_length=max_len,
        )
        scores, act_logits = self._forward(batch)

        chunk_answers  = [{} for _ in chunk]
        chunk_details  = [{} for _ in chunk]
        for row, m in enumerate(meta):
            c = m["c_idx"]
            q = m["q"]
            res = _decode_result(
                q,
                scores[row],         # [M]
                act_logits[row],     # [M, 3]
                batch["marker_mask"][row],
                self._temperatures,
            )
            q_id = q["id"]
            chunk_answers[c][q_id] = res["answer"]
            chunk_details[c][q_id] = res

        return [{"answers": chunk_answers[i], "details": chunk_details[i]} for i in range(len(chunk))]

    def benchmark(self, state, questions, n=50) -> Dict:
        for _ in range(3):
            self.predict(state, questions)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(state, questions)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return {
            "p50_ms":  times[n // 2],
            "p95_ms":  times[int(n * 0.95)],
            "mean_ms": sum(times) / len(times),
            "device":  str(self.device),
            "dtype":   str(self.torch_dtype),
            "n":       n,
        }


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load(model_id_or_path="convaiinnovations/laya", *, device=None, dtype="float16",
         token=None, subfolder=None, revision=None, batch_size=16,
         compile=False, pad_to_multiple=8) -> Agent:
    """Load a Laya agent. Downloads from HuggingFace on first use.

    Example::

        import laya_cuda as laya
        agent = laya.load()
        print(agent.predict("I was billed twice.", {"dept": {...}}))
    """
    return Agent(model_id_or_path, device=device, dtype=dtype, token=token,
                 subfolder=subfolder, revision=revision, batch_size=batch_size,
                 compile=compile, pad_to_multiple=pad_to_multiple)
