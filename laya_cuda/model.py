"""Native PyTorch ModernBERT + Laya decision heads - CUDA/CPU inference only.

Architecture reverse-engineered from the upstream convaiinnovations/laya checkpoint.
Weight keys match the safetensors file exactly - no remapping needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: Optional[list] = None
    rope_parameters: Optional[dict] = None

    @classmethod
    def from_dict(cls, value: dict) -> "EncoderConfig":
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0
                else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        return cfg

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind: str) -> float:
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


# ---------------------------------------------------------------------------
# Rotary embeddings
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len, head_dim, base, device, dtype):
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() * 2 / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


# ---------------------------------------------------------------------------
# Sliding-window additive mask
# ---------------------------------------------------------------------------

def sliding_window_mask(seq_len, window, device, dtype):
    row = torch.arange(seq_len, device=device)
    diff = (row.unsqueeze(1) - row.unsqueeze(0)).abs()
    m = torch.where(diff <= window,
                    torch.zeros(1, device=device, dtype=dtype),
                    torch.full((1,), float("-inf"), device=device, dtype=dtype))
    return m[None, None]   # [1,1,T,T]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


# ---------------------------------------------------------------------------
# Attention (with RoPE + optional sliding window)
# ---------------------------------------------------------------------------

class EncoderAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig, kind: str):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.local_attention if kind == "sliding_attention" else None
        self.rope_base = cfg.rope_base(kind)
        # Weight names match checkpoint exactly
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)
        self._rope_cache: Optional[tuple] = None
        self._rope_len = 0

    def _rope(self, T, device, dtype):
        if T > self._rope_len or self._rope_cache is None or self._rope_cache[0].device != device:
            cos, sin = build_rope_cache(max(T, 512), self.head_dim, self.rope_base, device, dtype)
            self._rope_cache = (cos, sin)
            self._rope_len = cos.shape[0]
        cos, sin = self._rope_cache
        return cos[:T], sin[:T]

    def forward(self, x, pad_mask=None):
        B, T, _ = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def rshp(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = rshp(q), rshp(k), rshp(v)

        cos, sin = self._rope(T, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.window is not None:
            win_mask = sliding_window_mask(T, self.window, x.device, x.dtype)
            mask = (pad_mask + win_mask) if pad_mask is not None else win_mask
        else:
            mask = pad_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# MLP - gated linear unit (GLU): Wi projects to 2*intermediate, split + gate
# intermediate_size in config is the full Wi output width
# ---------------------------------------------------------------------------

class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, wi_out: int, wo_out: int, bias: bool = False):
        """wi_out = full Wi output width (5248 in the english checkpoint).
           The GLU split uses wi_out//2 as input to Wo.
           wo_out = Wo output (should equal hidden_size).
        """
        super().__init__()
        self.Wi = nn.Linear(hidden_size, wi_out, bias=bias)
        self.Wo = nn.Linear(wi_out // 2, wo_out, bias=bias)

    def forward(self, x):
        h = self.Wi(x)                    # [..., wi_out]
        gate, val = h.chunk(2, dim=-1)    # each [..., wi_out//2]
        return self.Wo(F.gelu(gate) * val)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, kind: str, wi_out: int, has_attn_norm: bool = True):
        super().__init__()
        # ModernBERT: layer 0 input is already normalized by embeddings.norm, so no attn_norm
        if has_attn_norm:
            self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        else:
            self.attn_norm = None
        self.attn = EncoderAttention(cfg, kind)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg.hidden_size, wi_out=wi_out, wo_out=cfg.hidden_size,
                              bias=cfg.mlp_bias)

    def forward(self, x, pad_mask=None):
        h = self.attn_norm(x) if self.attn_norm is not None else x
        x = x + self.attn(h, pad_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class ModernBertEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, wi_out: int):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList([
            EncoderLayer(cfg, kind, wi_out=wi_out, has_attn_norm=(i > 0))
            for i, kind in enumerate(cfg.layer_types)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids, attention_mask):
        # Additive padding mask [B,1,1,T]
        dt = self.embeddings.tok_embeddings.weight.dtype
        pad_mask = torch.where(
            attention_mask[:, None, None, :],
            torch.zeros(1, device=input_ids.device, dtype=dt),
            torch.full((1,), float("-inf"), device=input_ids.device, dtype=dt),
        )
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Decision head (2-layer Transformer over the marker tokens)
# Uses standard nn.TransformerEncoderLayer which matches checkpoint key names:
#   head.layers.N.self_attn.{in_proj_weight, in_proj_bias, out_proj.*}
#   head.layers.N.{linear1, linear2, norm1, norm2}
# ---------------------------------------------------------------------------

class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, nhead: int = 16, ffn_dim: int = 4096):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=True,  # pre-norm matches checkpoint layout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Scorer: LayerNorm → Linear(H,H) → GELU → Linear(H,1)
# Checkpoint keys: scorer.0.{weight,bias}, scorer.1.{weight,bias},
#                  scorer.3.{weight,bias}  (index 2 = GELU, no params)
# ---------------------------------------------------------------------------

class Scorer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # scorer.0 = LayerNorm, scorer.1 = Linear(H,H), scorer.3 = Linear(H,1)
        self.norm = nn.LayerNorm(hidden_size)   # scorer.0
        self.dense = nn.Linear(hidden_size, hidden_size)  # scorer.1
        self.out = nn.Linear(hidden_size, 1)               # scorer.3

    def forward(self, x):
        return self.out(F.gelu(self.dense(self.norm(x))))


# ---------------------------------------------------------------------------
# Full decision model
# ---------------------------------------------------------------------------

class DecisionModel(nn.Module):
    """Laya: encoder + type_emb + decision head + scorer + act_head.
    
    All dimensions are inferred from the raw checkpoint weights to avoid hardcoding.
    Call DecisionModel.from_weights(enc_cfg, agent_cfg, raw_weights) instead of
    constructing directly.
    """

    def __init__(self, enc_cfg: EncoderConfig, agent_cfg: dict,
                 wi_out: int, act_in: int, act_hidden: int, num_actions: int,
                 num_qtypes: int = 3, head_layers: int = 2):
        super().__init__()
        H = enc_cfg.hidden_size

        self.encoder = ModernBertEncoder(enc_cfg, wi_out=wi_out)
        self.type_emb = nn.Embedding(num_qtypes, H)
        nhead = enc_cfg.num_attention_heads
        self.head = DecisionHead(H, num_layers=head_layers, nhead=nhead, ffn_dim=4 * H)
        self.scorer = Scorer(H)
        self.act_head = nn.Sequential(
            nn.Linear(act_in, act_hidden),
            nn.GELU(),
            nn.Linear(act_hidden, num_actions),
        )
        self.temperature = nn.Parameter(torch.ones(num_qtypes), requires_grad=False)

    @classmethod
    def from_weights(cls, enc_cfg: EncoderConfig, agent_cfg: dict,
                     raw: dict) -> "DecisionModel":
        """Construct model with exact dims read from checkpoint weight shapes."""
        # MLP dims from first encoder layer
        wi_out = raw["encoder.layers.0.mlp.Wi.weight"].shape[0]   # e.g. 5248
        # Act head dims
        act_w0 = raw["act_head.0.weight"]   # [act_hidden, act_in]
        act_hidden, act_in = act_w0.shape
        act_w2 = raw["act_head.2.weight"]   # [num_actions, act_hidden]
        num_actions = act_w2.shape[0]
        # num_qtypes from type_emb
        num_qtypes = raw["type_emb.weight"].shape[0]
        # temperature size
        head_layers = agent_cfg.get("head_layers", 2)
        return cls(
            enc_cfg, agent_cfg,
            wi_out=wi_out, act_in=act_in, act_hidden=act_hidden,
            num_actions=num_actions, num_qtypes=num_qtypes, head_layers=head_layers,
        )

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtypes):
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T] bool
            marker_pos:     [B, M] int - token positions of [MASK] markers
            marker_mask:    [B, M] bool - which markers are real
            qtypes:         [B] int - question type (0=choice,1=score,2=noul)

        Returns:
            scores:     [B, M] - scalar score at each marker
            act_logits: [B, M, num_actions]
        """
        # Encode full sequence
        hidden = self.encoder(input_ids, attention_mask)   # [B, T, H]

        B, M = marker_pos.shape
        H = hidden.shape[-1]
        num_qtypes = self.type_emb.num_embeddings

        # Gather hidden states at marker positions
        idx = marker_pos.clamp(min=0).unsqueeze(-1).expand(B, M, H)
        marker_h = hidden.gather(1, idx)                   # [B, M, H]

        # Add per-type embedding
        type_e = self.type_emb(qtypes.unsqueeze(1).expand(B, M))  # [B, M, H]
        marker_h = marker_h + type_e

        # Decision Transformer (attend markers to each other)
        kpm = ~marker_mask  # [B, M]  True = ignore
        marker_h = self.head(marker_h, key_padding_mask=kpm)   # [B, M, H]

        # Scorer → [B, M, 1] → [B, M]
        scores = self.scorer(marker_h).squeeze(-1)

        # Action head: concat marker_h with qtype one-hot
        # act_head input size = H + act_onehot_dim (inferred from weight shape)
        act_in_size = self.act_head[0].in_features
        act_onehot_dim = act_in_size - H
        qtype_onehot = F.one_hot(qtypes, num_classes=act_onehot_dim).to(marker_h.dtype)  # [B, act_onehot_dim]
        qtype_onehot = qtype_onehot.unsqueeze(1).expand(B, M, act_onehot_dim)             # [B, M, act_onehot_dim]
        act_input = torch.cat([marker_h, qtype_onehot], dim=-1)                            # [B, M, act_in_size]
        act_logits = self.act_head(act_input)

        return scores, act_logits


# ---------------------------------------------------------------------------
# Direct weight loading - keys already match our module names
# ---------------------------------------------------------------------------

def load_weights(model: DecisionModel, raw: dict) -> tuple[list, list]:
    """Load checkpoint weights with name remapping for our PyTorch module names."""
    remap = {
        # Scorer: sequential indices → named submodules
        "scorer.0.weight": "scorer.norm.weight",
        "scorer.0.bias":   "scorer.norm.bias",
        "scorer.1.weight": "scorer.dense.weight",
        "scorer.1.bias":   "scorer.dense.bias",
        "scorer.3.weight": "scorer.out.weight",
        "scorer.3.bias":   "scorer.out.bias",
        # Act head: sequential indices → named submodules
        "act_head.0.weight": "act_head.0.weight",
        "act_head.0.bias":   "act_head.0.bias",
        "act_head.2.weight": "act_head.2.weight",
        "act_head.2.bias":   "act_head.2.bias",
    }
    mapped = {}
    for k, v in raw.items():
        mapped[remap.get(k, k)] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return missing, unexpected
