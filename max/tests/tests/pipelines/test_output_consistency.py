"""Logit consistency check: MAX pipeline vs. HuggingFace reference for OpenELM.

Both pipelines receive the same token IDs. We compare logits at the final token
position. Requires weights in models/openelm-270m-instruct/; skipped if absent.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT       = Path(__file__).parent.parent
MODEL_PATH      = REPO_ROOT / "models" / "openelm-270m-instruct"
TOKENIZER_PATH  = REPO_ROOT / "models" / "llama2-tokenizer"
model_available = MODEL_PATH.exists() and any(MODEL_PATH.iterdir())

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MODEL_PATH))


@pytest.mark.skipif(not model_available, reason="Model weights not downloaded. Run step_02 first.")
class TestOutputConsistency:
    """End-to-end logit comparison: HuggingFace reference vs. MAX pipeline."""

    PROMPT = "The capital of France is"

    def _get_hf_logits(self) -> np.ndarray:
        """Return last-position logits from the HuggingFace reference model.

        modeling_openelm.py uses relative imports which fail when loaded directly
        via sys.path. Using get_class_from_dynamic_module lets HuggingFace register
        the file under its own package context so the imports resolve correctly.
        """
        from transformers import AutoConfig, LlamaTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from safetensors.torch import load_file

        tokenizer  = LlamaTokenizer.from_pretrained(str(TOKENIZER_PATH))
        hf_config  = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        openelm_cls = get_class_from_dynamic_module(
            "modeling_openelm.OpenELMForCausalLM",
            str(MODEL_PATH),
        )

        model = openelm_cls(hf_config)
        model.load_state_dict(load_file(str(MODEL_PATH / "model.safetensors")))
        model.eval()

        input_ids = tokenizer(self.PROMPT, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        return outputs.logits[0, -1, :].float().numpy()

    def _get_max_logits(self) -> np.ndarray:
        """Return last-position logits from the MAX pipeline."""
        from transformers import LlamaTokenizer
        from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
        from max.engine import InferenceSession
        from max.driver import CPU
        from max.pipelines.lib.config.kv_cache_config import KVCacheConfig
        from max.pipelines.lib.interfaces.pipeline_model import ReturnLogits

        from openelm_pipeline import ARCHITECTURES
        from openelm_pipeline.model import OpenELMModel

        # PipelineConfig validates the registry at construction time.
        for arch in ARCHITECTURES:
            PIPELINE_REGISTRY.register(arch, allow_override=True)

        tokenizer = LlamaTokenizer.from_pretrained(str(TOKENIZER_PATH))
        token_ids = tokenizer.encode(self.PROMPT)

        config  = PipelineConfig(
            model_path            = str(MODEL_PATH),
            max_length            = 512,
            trust_remote_code     = True,
            quantization_encoding = "float32",
        )
        devices = [CPU()]
        session = InferenceSession(devices=devices)
        kv_cfg  = KVCacheConfig()

        model = OpenELMModel(
            pipeline_config = config,
            session         = session,
            devices         = devices,
            kv_cache_config = kv_cfg,
            weights         = None,
            adapter         = None,
            return_logits   = ReturnLogits.ALL,
        )

        context      = SimpleNamespace(tokens=token_ids)
        model_inputs = model.prepare_initial_token_inputs(context_batch=[context])
        model_outputs = model.execute(model_inputs)

        logits_np = model_outputs.logits.to_numpy()
        return logits_np[0, -1, :].astype(np.float32)

    def test_top1_token_matches(self):
        """Greedy argmax token must be identical between HF and MAX."""
        hf_logits  = self._get_hf_logits()
        max_logits = self._get_max_logits()

        hf_top1  = int(np.argmax(hf_logits))
        max_top1 = int(np.argmax(max_logits))

        assert hf_top1 == max_top1, (
            f"Top-1 token differs: HF predicts {hf_top1}, MAX predicts {max_top1}"
        )

    def test_top5_tokens_match(self):
        """At least 3 of the top-5 predicted tokens must overlap.

        Requires ≥3/5 rather than exact agreement to tolerate float32 vs.
        bfloat16 rounding while still catching a structurally broken forward pass.
        """
        hf_logits  = self._get_hf_logits()
        max_logits = self._get_max_logits()

        hf_top5  = set(np.argsort(hf_logits)[-5:])
        max_top5 = set(np.argsort(max_logits)[-5:])
        overlap  = len(hf_top5 & max_top5)

        assert overlap >= 3, (
            f"Only {overlap}/5 top tokens overlap between HF and MAX.\n"
            f"HF top-5:  {sorted(hf_top5)}\n"
            f"MAX top-5: {sorted(max_top5)}"
        )

    def test_logit_magnitude_is_similar(self):
        """Max logit value must be within 10 units between HF and MAX.

        A large divergence here indicates a structural issue in the forward pass
        (wrong layer order, missing normalization, incorrect RoPE, etc.).
        """
        hf_logits  = self._get_hf_logits()
        max_logits = self._get_max_logits()

        hf_max  = float(np.max(hf_logits))
        max_max = float(np.max(max_logits))
        diff    = abs(hf_max - max_max)

        assert diff < 10.0, (
            f"Max logit diverges: HF={hf_max:.3f}, MAX={max_max:.3f}, diff={diff:.3f}"
        )
