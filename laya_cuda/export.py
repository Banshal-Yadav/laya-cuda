"""ONNX export + TensorRT engine builder for maximum inference speed.

Usage (from WSL or Windows with TensorRT installed):

    from laya_cuda.export import export_onnx, build_trt_engine
    import laya_cuda as laya

    agent = laya.load("convaiinnovations/laya")
    export_onnx(agent, "laya.onnx")
    build_trt_engine("laya.onnx", "laya.trt", fp16=True)

Then use TRTAgent for ~1-3ms inference:

    from laya_cuda.trt_agent import TRTAgent
    trt = TRTAgent("laya.trt", agent.tok, agent.cfg, agent.mlx_cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def export_onnx(
    agent,
    output_path: str,
    *,
    opset: int = 17,
    seq_len: int = 128,
    batch_size: int = 1,
) -> Path:
    """Export the agent's model to ONNX with dynamic batch + sequence axes.

    Args:
        agent: A loaded laya_cuda.Agent.
        output_path: Destination .onnx file.
        opset: ONNX opset version (17+ recommended for attention ops).
        seq_len: Representative sequence length for tracing.
        batch_size: Representative batch size for tracing.

    Returns:
        Path to the written ONNX file.
    """
    output_path = Path(output_path)
    model = agent.model
    device = agent.device
    orig_training = model.training
    model.eval()

    # Dummy inputs
    dummy = {
        "input_ids":      torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        "marker_pos":     torch.zeros(batch_size, 4, dtype=torch.long, device=device),
        "marker_mask":    torch.ones(batch_size, 4, dtype=torch.bool, device=device),
        "qtypes":         torch.zeros(batch_size, dtype=torch.long, device=device),
    }

    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "marker_pos":     {0: "batch", 1: "num_markers"},
        "marker_mask":    {0: "batch", 1: "num_markers"},
        "qtypes":         {0: "batch"},
        "scoring_logits": {0: "batch", 1: "num_markers"},
        "action_logits":  {0: "batch", 1: "num_markers"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                dummy["input_ids"],
                dummy["attention_mask"],
                dummy["marker_pos"],
                dummy["marker_mask"],
                dummy["qtypes"],
            ),
            str(output_path),
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtypes"],
            output_names=["scoring_logits", "action_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    model.train(orig_training)
    print(f"[laya-cuda] ONNX model written → {output_path}")
    return output_path


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    *,
    fp16: bool = True,
    max_batch: int = 64,
    max_seq_len: int = 512,
    max_markers: int = 16,
    workspace_gb: float = 4.0,
    verbose: bool = False,
) -> Path:
    """Compile an ONNX model to a TensorRT FP16 engine.

    Requires: tensorrt>=10.0 (pip install tensorrt)

    Args:
        onnx_path: Path to the .onnx file.
        engine_path: Destination .trt engine file.
        fp16: Enable FP16 precision (recommended for RTX).
        max_batch: Maximum batch size to optimise for.
        max_seq_len: Maximum sequence length to optimise for.
        max_markers: Maximum number of decision markers per item.
        workspace_gb: GPU memory budget for TRT builder.

    Returns:
        Path to the serialized engine.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "tensorrt is required for TRT engine building. "
            "Install it: pip install tensorrt\n"
            "Or on WSL: pip install tensorrt-cu12"
        )

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    logger_severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_severity)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"TRT ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[laya-cuda] TRT: FP16 precision enabled")

    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    for name, (b_min, b_opt, b_max), (s_min, s_opt, s_max) in [
        ("input_ids",      (1, min(8, max_batch), max_batch),
                           (16, min(128, max_seq_len), max_seq_len)),
        ("attention_mask", (1, min(8, max_batch), max_batch),
                           (16, min(128, max_seq_len), max_seq_len)),
    ]:
        profile.set_shape(name, (b_min, s_min), (b_opt, s_opt), (b_max, s_max))

    for name in ("marker_pos", "marker_mask"):
        profile.set_shape(
            name,
            (1, 2), (min(8, max_batch), min(8, max_markers)), (max_batch, max_markers)
        )

    config.add_optimization_profile(profile)

    print(f"[laya-cuda] Building TRT engine (this takes 1-5 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(str(engine_path), "wb") as f:
        f.write(serialized)

    print(f"[laya-cuda] TRT engine written → {engine_path}")
    return engine_path
