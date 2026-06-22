"""MAX PipelineModel implementation for Apple's OpenELM architecture."""

import math
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weight import Weight
from max.graph.weights import Weights as WeightsLoader
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines import PipelineConfig


@dataclass
class OpenELMInputs(ModelInputs):
    input_ids: np.ndarray


@dataclass
class OpenELMLayerConfig:
    """Per-layer dimensions for OpenELM's layer-wise scaling."""
    num_query_heads: int
    num_kv_heads:    int
    ffn_hidden_dim:  int
    head_dim:        int


def compute_layer_configs(hf_config) -> list[OpenELMLayerConfig]:
    """Compute per-layer dimensions from the HuggingFace config.

    Query heads, KV heads, and FFN multipliers may be stored as a scalar
    (uniform across layers) or a list (layer-wise scaling). Both cases are
    normalised to a list of length num_transformer_layers.

    FFN hidden dim is not stored directly — it is derived as:
        make_divisible(ffn_multipliers[i] * model_dim, ffn_dim_divisor)
    matching the make_divisible() logic in Apple's modeling_openelm.py.
    """
    num_layers = hf_config.num_transformer_layers
    head_dim   = hf_config.head_dim

    def to_list(val, length):
        return val if isinstance(val, list) else [val] * length

    query_heads = to_list(hf_config.num_query_heads, num_layers)
    kv_heads    = to_list(hf_config.num_kv_heads,    num_layers)
    ffn_mults   = to_list(hf_config.ffn_multipliers, num_layers)

    model_dim = hf_config.model_dim
    divisor   = getattr(hf_config, "ffn_dim_divisor", 256)

    def make_divisible(v, d):
        # Round to nearest multiple of d, matching Apple's reference implementation.
        new_v = max(d, int(v + d / 2) // d * d)
        if new_v < 0.9 * v:
            new_v += d
        return new_v

    return [
        OpenELMLayerConfig(
            num_query_heads = query_heads[i],
            num_kv_heads    = kv_heads[i],
            ffn_hidden_dim  = make_divisible(ffn_mults[i] * model_dim, divisor),
            head_dim        = head_dim,
        )
        for i in range(num_layers)
    ]


class OpenELMModel(PipelineModel):
    """MAX PipelineModel for Apple's OpenELM family (270M / 450M / 1.1B / 3B)."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session,
        devices:         list,
        kv_cache_config,
        weights:         WeightsLoader,
        adapter,
        return_logits,
        **kwargs,
    ):
        _session = session

        super().__init__(
            pipeline_config = pipeline_config,
            session         = session,
            devices         = devices,
            kv_cache_config = kv_cache_config,
            weights         = weights,
            adapter         = adapter,
            return_logits   = return_logits,
            **kwargs,
        )

        hf = self.huggingface_config
        self.hf_config     = hf
        self.layer_configs = compute_layer_configs(hf)
        self.num_layers    = hf.num_transformer_layers
        self.model_dim     = hf.model_dim
        self.vocab_size    = hf.vocab_size
        self._head_dim     = hf.head_dim

        rope_base = float(getattr(hf, "rope_freq_constant", 10000.0))
        self._cos_table, self._sin_table = self._build_rope_cache(
            self.max_seq_len, hf.head_dim, rope_base
        )

        from safetensors import safe_open
        from pathlib import Path

        model_path = Path(pipeline_config.model.model_path)

        # Support both single-file (270M, 450M) and sharded (1.1B, 3B) layouts.
        safetensor_files = sorted(
            list(model_path.glob("model.safetensors")) +
            list(model_path.glob("model-*.safetensors"))
        )
        if not safetensor_files:
            raise FileNotFoundError(
                f"No safetensors weight files found in {model_path}."
            )

        state_dict: dict[str, np.ndarray] = {}
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key).float().numpy()

        graph = self._build_graph(state_dict)
        self._compiled_model = _session.load(graph, weights_registry=state_dict)
        # Keep arrays alive: the engine holds raw pointers into these buffers
        # and does not increment Python refcounts (per InferenceSession.load docs).
        self._state_dict = state_dict

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config:    PipelineConfig,
        huggingface_config,
    ) -> int:
        user_max  = getattr(pipeline_config.model, "max_length", None)
        model_max = getattr(huggingface_config, "max_context_length", 2048)
        return model_max if user_max is None else min(user_max, model_max)

    @staticmethod
    def _build_rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0):
        inv_freqs = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles    = np.outer(positions, inv_freqs)
        angles    = np.concatenate([angles, angles], axis=-1)
        return np.cos(angles), np.sin(angles)

    def _build_graph(self, state_dict: dict) -> Graph:
        hf     = self.hf_config
        device = DeviceRef.CPU()
        eps    = getattr(hf, "rms_norm_eps", 1e-6)
        H      = self._head_dim

        def w(name: str) -> Weight:
            arr = state_dict[name]
            return Weight(
                name   = name,
                dtype  = DType.float32,
                shape  = list(arr.shape),
                device = device,
            )

        with Graph(
            "openelm",
            input_types=[
                TensorType(DType.int64,   shape=["batch", "seq_len"],         device=device),
                TensorType(DType.float32, shape=[1, 1, "seq_len", H],         device=device),
                TensorType(DType.float32, shape=[1, 1, "seq_len", H],         device=device),
                TensorType(DType.float32, shape=[1, 1, "seq_len", "seq_len"], device=device),
            ],
        ) as graph:

            tokens, cos, sin, mask = graph.inputs
            B = tokens.shape[0]
            S = tokens.shape[1]

            emb_w = w("transformer.token_embeddings.weight")
            x     = ops.gather(emb_w, tokens, axis=0)

            for i, layer_cfg in enumerate(self.layer_configs):
                prefix   = f"transformer.layers.{i}"
                q_heads  = layer_cfg.num_query_heads
                kv_heads = layer_cfg.num_kv_heads
                h_dim    = layer_cfg.head_dim
                q_size   = q_heads  * h_dim
                k_size   = kv_heads * h_dim
                v_size   = kv_heads * h_dim

                normed = ops.rms_norm(x, w(f"{prefix}.attn_norm.weight"), eps)

                qkv = ops.matmul(normed, ops.transpose(w(f"{prefix}.attn.qkv_proj.weight"), 0, 1))
                q, k, v = ops.split(qkv, [q_size, k_size, v_size], axis=2)

                q = ops.transpose(ops.reshape(q, [B, S, q_heads,  h_dim]), 1, 2)
                k = ops.transpose(ops.reshape(k, [B, S, kv_heads, h_dim]), 1, 2)
                v = ops.transpose(ops.reshape(v, [B, S, kv_heads, h_dim]), 1, 2)

                # OpenELM applies per-head RMSNorm to Q and K before RoPE.
                if getattr(hf, "normalize_qk_projections", False):
                    q = ops.rms_norm(q, w(f"{prefix}.attn.q_norm.weight"), eps)
                    k = ops.rms_norm(k, w(f"{prefix}.attn.k_norm.weight"), eps)

                half = h_dim // 2

                def apply_rope(x_in):
                    x1 = ops.slice_tensor(x_in, (slice(None), slice(None), slice(None), slice(None, half)))
                    x2 = ops.slice_tensor(x_in, (slice(None), slice(None), slice(None), slice(half, None)))
                    return x_in * cos + ops.concat([ops.negate(x2), x1], axis=-1) * sin

                q = apply_rope(q)
                k = apply_rope(k)

                num_groups = q_heads // kv_heads
                if num_groups > 1:
                    k = ops.repeat_interleave(k, num_groups, axis=1, out_dim=q_heads)
                    v = ops.repeat_interleave(v, num_groups, axis=1, out_dim=q_heads)

                scale    = 1.0 / math.sqrt(h_dim)
                scores   = ops.matmul(q, ops.transpose(k, 2, 3)) * scale
                attn_w   = ops.softmax(scores + mask, axis=-1)
                attn_out = ops.matmul(attn_w, v)

                attn_out = ops.reshape(ops.transpose(attn_out, 1, 2), [B, S, q_size])
                attn_out = ops.matmul(attn_out, ops.transpose(w(f"{prefix}.attn.out_proj.weight"), 0, 1))
                x        = x + attn_out

                normed  = ops.rms_norm(x, w(f"{prefix}.ffn_norm.weight"), eps)
                ffn_h   = layer_cfg.ffn_hidden_dim
                y12     = ops.matmul(normed, ops.transpose(w(f"{prefix}.ffn.proj_1.weight"), 0, 1))
                y1, y2  = ops.split(y12, [ffn_h, ffn_h], axis=2)
                ffn_out = ops.matmul(
                    ops.silu(y1) * y2,
                    ops.transpose(w(f"{prefix}.ffn.proj_2.weight"), 0, 1),
                )
                x = x + ffn_out

            x      = ops.rms_norm(x, w("transformer.norm.weight"), eps)
            logits = ops.matmul(x, ops.transpose(emb_w, 0, 1))
            graph.output(logits)

        return graph

    def execute(self, model_inputs: "OpenELMInputs") -> ModelOutputs:
        input_ids = np.array(model_inputs.input_ids, dtype=np.int64)
        seq_len   = input_ids.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Reduce prompt length or MAX_NEW_TOKENS so total tokens ≤ {self.max_seq_len}."
            )

        cos_np  = self._cos_table[:seq_len].reshape(1, 1, seq_len, self._head_dim)
        sin_np  = self._sin_table[:seq_len].reshape(1, 1, seq_len, self._head_dim)
        mask_np = np.triu(np.full((seq_len, seq_len), float("-inf"), dtype=np.float32), k=1)
        mask_np = mask_np.reshape(1, 1, seq_len, seq_len)

        result = self._compiled_model.execute(
            np.ascontiguousarray(input_ids),
            np.ascontiguousarray(cos_np),
            np.ascontiguousarray(sin_np),
            np.ascontiguousarray(mask_np),
        )

        return ModelOutputs(logits=result[0])

    def prepare_initial_token_inputs(
        self,
        context_batch,
        kv_cache_inputs = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        input_ids = np.array(
            [ctx.tokens for ctx in context_batch],
            dtype=np.int64,
        )
        return OpenELMInputs(input_ids=input_ids)

    def prepare_next_token_inputs(
        self,
        next_tokens,
        prev_model_inputs: "OpenELMInputs",
    ) -> "OpenELMInputs":
        # No KV cache: carry the full token history so the model has context.
        # This is O(n²) in computation but keeps Phase 1 simple and correct.
        prev_ids  = prev_model_inputs.input_ids
        next_tok  = np.array(next_tokens, dtype=np.int64).reshape(-1, 1)
        return OpenELMInputs(input_ids=np.concatenate([prev_ids, next_tok], axis=1))
