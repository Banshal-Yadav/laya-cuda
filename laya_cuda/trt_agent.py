"""TensorRT inference agent - maximum speed (~1-3 ms) on RTX.

Usage::

    from laya_cuda.trt_agent import TRTAgent
    trt = TRTAgent.from_agent_and_engine(agent, "laya.trt")
    result = trt.predict("I was billed twice.", {"department": {...}})

The engine must have been built with laya_cuda.export.build_trt_engine().
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .agent import _decode_result, collate_items
from .common import (
    build_prefix,
    build_sequence,
    prepare_questions,
    serialize_state,
)


class TRTAgent:
    """Inference via a pre-built TensorRT engine. No PyTorch model in the loop."""

    def __init__(
        self,
        engine_path: str,
        tok,
        cfg: Dict,
        mlx_cfg: Optional[Dict] = None,
        *,
        batch_size: int = 16,
        pad_to_multiple: int = 8,
    ):
        try:
            import tensorrt as trt
            import cuda  # cuda-python
            from cuda import cudart
        except ImportError:
            raise ImportError(
                "tensorrt and cuda-python are required for TRTAgent.\n"
                "pip install tensorrt cuda-python"
            )

        self.tok = tok
        self.cfg = cfg
        self.mlx_cfg = mlx_cfg or {}
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        print(f"[laya-cuda/trt] Engine loaded from {engine_path}")

    @classmethod
    def from_agent_and_engine(cls, agent, engine_path: str, **kwargs) -> "TRTAgent":
        """Convenience constructor - reuse tok/cfg from an existing Agent."""
        return cls(
            engine_path,
            tok=agent.tok,
            cfg=agent.cfg,
            mlx_cfg=agent.mlx_cfg,
            **kwargs,
        )

    def _run_engine(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        marker_pos: np.ndarray,
        marker_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute TRT engine synchronously; returns numpy output arrays."""
        import tensorrt as trt
        import ctypes
        import cuda
        from cuda import cudart

        context = self.context
        engine = self.engine

        # Set input shapes
        context.set_input_shape("input_ids",      input_ids.shape)
        context.set_input_shape("attention_mask",  attention_mask.shape)
        context.set_input_shape("marker_pos",      marker_pos.shape)
        context.set_input_shape("marker_mask",     marker_mask.shape)

        # Allocate and copy inputs
        inputs_np = {
            "input_ids":      input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
            "marker_pos":     marker_pos.astype(np.int32),
            "marker_mask":    marker_mask.astype(np.int32),
        }

        # Get output shapes
        B, M = marker_pos.shape
        num_labels = engine.get_tensor_shape("scoring_logits")[2] if \
            engine.get_tensor_shape("scoring_logits")[2] > 0 else 64
        num_actions = engine.get_tensor_shape("action_logits")[2] if \
            engine.get_tensor_shape("action_logits")[2] > 0 else 3

        scoring_out = np.zeros((B, M, num_labels), dtype=np.float32)
        action_out = np.zeros((B, M, num_actions), dtype=np.float32)

        # Use CUDA stream for async execution
        err, stream = cudart.cudaStreamCreate()

        # Build binding dict
        bindings = {}
        device_ptrs = {}
        host_ptrs = {}

        for name, arr in inputs_np.items():
            arr = np.ascontiguousarray(arr)
            host_ptrs[name] = arr
            err, d_ptr = cudart.cudaMallocAsync(arr.nbytes, stream)
            cudart.cudaMemcpyAsync(d_ptr, arr.ctypes.data, arr.nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            device_ptrs[name] = d_ptr

        for name, arr in [("scoring_logits", scoring_out), ("action_logits", action_out)]:
            arr = np.ascontiguousarray(arr)
            host_ptrs[name] = arr
            err, d_ptr = cudart.cudaMallocAsync(arr.nbytes, stream)
            device_ptrs[name] = d_ptr

        # Set tensor addresses
        for name, d_ptr in device_ptrs.items():
            context.set_tensor_address(name, d_ptr)

        # Execute
        context.execute_async_v3(stream)

        # Copy outputs back
        for name, arr in [("scoring_logits", scoring_out), ("action_logits", action_out)]:
            cudart.cudaMemcpyAsync(
                arr.ctypes.data, device_ptrs[name], arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
            )

        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)
        for d_ptr in device_ptrs.values():
            cudart.cudaFree(d_ptr)

        return scoring_out, action_out

    # Public API mirrors Agent
    def predict(self, state: Union[str, dict, list], questions: Dict) -> Dict:
        return self.batch_predict([(state, questions)])[0]

    def batch_predict(self, items: List[tuple]) -> List[Dict]:
        all_results = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i: i + self.batch_size]
            all_results.extend(self._run_chunk(chunk))
        return all_results

    def _run_chunk(self, chunk: List[tuple]) -> List[Dict]:
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        calibration = self.mlx_cfg.get("calibration")

        flat_items: List[Dict] = []
        meta: List[Dict] = []

        for c_idx, (state, questions) in enumerate(chunk):
            norm_qs = prepare_questions(questions)
            state_text = serialize_state(state).replace(self.tok.mask_token, " ")
            state_ids = self.tok(state_text, add_special_tokens=False)
            for q in norm_qs:
                head_ids, markers = build_prefix(
                    self.tok, q, max_len=max_len, head_max_len=head_max_len
                )
                seq = build_sequence(self.tok, state_ids, head_ids, markers, max_len=max_len)
                seq["qtype"] = q["qtype"]
                flat_items.append(seq)
                meta.append({"c_idx": c_idx, "q": q})

        if not flat_items:
            return [{} for _ in chunk]

        batch = collate_items(
            flat_items,
            pad_id=self.tok.pad_token_id,
            pad_to_multiple=self.pad_to_multiple,
            max_length=max_len,
        )

        scoring_logits, action_logits = self._run_engine(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
        )

        chunk_details: List[Dict] = [{} for _ in chunk]
        chunk_answers: List[Dict] = [{} for _ in chunk]
        for row_idx, m in enumerate(meta):
            c_idx = m["c_idx"]
            q = m["q"]
            result = _decode_result(
                q,
                scoring_logits[row_idx],
                action_logits[row_idx],
                batch["marker_mask"][row_idx],
                calibration,
            )
            q_id = q["id"]
            chunk_details[c_idx][q_id] = result
            chunk_answers[c_idx][q_id] = result["answer"]

        return [
            {"answers": chunk_answers[i], "details": chunk_details[i]}
            for i in range(len(chunk))
        ]

    def benchmark(self, state: str, questions: Dict, n: int = 50) -> Dict:
        for _ in range(3):
            self.predict(state, questions)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(state, questions)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return {
            "p50_ms": times[n // 2],
            "p95_ms": times[int(n * 0.95)],
            "mean_ms": sum(times) / len(times),
            "backend": "tensorrt",
            "n": n,
        }
