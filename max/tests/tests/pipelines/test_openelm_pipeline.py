"""Structural tests for the OpenELM MAX pipeline."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT       = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH      = REPO_ROOT / "models" / "openelm-270m-instruct"
model_available = MODEL_PATH.exists() and any(MODEL_PATH.iterdir())


class TestArchitectureRegistration:
    """Verify ARCHITECTURES list content and SupportedArchitecture fields."""

    def test_architectures_list_exists(self):
        from openelm_pipeline import ARCHITECTURES

        assert isinstance(ARCHITECTURES, list), (
            f"ARCHITECTURES must be a list, not a {type(ARCHITECTURES).__name__}"
        )
        assert len(ARCHITECTURES) > 0, "ARCHITECTURES list is empty"

    def test_architecture_name_matches_hf_config(self):
        from openelm_pipeline import ARCHITECTURES

        arch = ARCHITECTURES[0]
        assert arch.name == "OpenELMForCausalLM", (
            f"Architecture name is '{arch.name}'. "
            "Must be 'OpenELMForCausalLM' to match OpenELM's config.json."
        )

    def test_architecture_has_pipeline_model(self):
        from openelm_pipeline import ARCHITECTURES
        from max.pipelines.lib import PipelineModel

        arch = ARCHITECTURES[0]
        assert arch.pipeline_model is not None
        assert issubclass(arch.pipeline_model, PipelineModel), (
            f"{arch.pipeline_model.__name__} must inherit from PipelineModel"
        )

    def test_architecture_has_tokenizer(self):
        from openelm_pipeline import ARCHITECTURES

        arch = ARCHITECTURES[0]
        assert arch.tokenizer is not None

    def test_architecture_has_example_repos(self):
        from openelm_pipeline import ARCHITECTURES

        arch = ARCHITECTURES[0]
        assert len(arch.example_repo_ids) > 0
        for repo in arch.example_repo_ids:
            assert "openelm" in repo.lower(), (
                f"Repo '{repo}' does not look like an OpenELM repo"
            )

    def test_architecture_supports_float32(self):
        from openelm_pipeline import ARCHITECTURES

        arch = ARCHITECTURES[0]
        assert "float32" in arch.supported_encodings

    def test_architecture_supports_bfloat16(self):
        from openelm_pipeline import ARCHITECTURES

        arch = ARCHITECTURES[0]
        assert "bfloat16" in arch.supported_encodings

    def test_default_weights_format_is_safetensors(self):
        from openelm_pipeline import ARCHITECTURES
        from max.graph.weights import WeightsFormat

        arch = ARCHITECTURES[0]
        assert arch.default_weights_format == WeightsFormat.safetensors, (
            f"default_weights_format is {arch.default_weights_format}, "
            "expected WeightsFormat.safetensors"
        )


class TestLayerConfigs:
    """Verify compute_layer_configs() for both uniform and layer-wise configs."""

    def _make_uniform_config(self, num_layers=4):
        return SimpleNamespace(
            num_transformer_layers = num_layers,
            head_dim               = 64,
            model_dim              = 1024,
            num_query_heads        = 4,
            num_kv_heads           = 2,
            ffn_multipliers        = 4.0,
            ffn_dim_divisor        = 256,
            max_context_length     = 2048,
            rms_norm_eps           = 1e-6,
            rope_freq_constant     = 10000,
            vocab_size             = 32000,
        )

    def _make_layerwise_config(self, num_layers=8):
        return SimpleNamespace(
            num_transformer_layers = num_layers,
            head_dim               = 64,
            model_dim              = 1024,
            num_query_heads        = list(range(4, 4 + num_layers)),
            num_kv_heads           = [max(1, h // 2) for h in range(4, 4 + num_layers)],
            ffn_multipliers        = [4.0 + i * 0.25 for i in range(num_layers)],
            ffn_dim_divisor        = 256,
            max_context_length     = 2048,
            rms_norm_eps           = 1e-6,
            rope_freq_constant     = 10000,
            vocab_size             = 32000,
        )

    def test_correct_number_of_layer_configs(self):
        from openelm_pipeline.model import compute_layer_configs

        for num_layers in [4, 16, 20, 28, 36]:
            cfg     = self._make_uniform_config(num_layers)
            configs = compute_layer_configs(cfg)
            assert len(configs) == num_layers, (
                f"Expected {num_layers} configs, got {len(configs)}"
            )

    def test_uniform_config_produces_identical_layers(self):
        from openelm_pipeline.model import compute_layer_configs

        cfg     = self._make_uniform_config(num_layers=4)
        configs = compute_layer_configs(cfg)
        first   = configs[0]

        for i, c in enumerate(configs[1:], start=1):
            assert c.num_query_heads == first.num_query_heads, (
                f"Layer {i} query heads differ in a uniform config"
            )
            assert c.ffn_hidden_dim == first.ffn_hidden_dim, (
                f"Layer {i} FFN dim differs in a uniform config"
            )

    def test_layerwise_config_produces_increasing_heads(self):
        from openelm_pipeline.model import compute_layer_configs

        cfg     = self._make_layerwise_config(num_layers=8)
        configs = compute_layer_configs(cfg)
        heads   = [c.num_query_heads for c in configs]

        assert heads == sorted(heads), (
            f"Expected query heads to increase monotonically, got: {heads}"
        )

    def test_ffn_hidden_dim_is_multiple_of_256(self):
        from openelm_pipeline.model import compute_layer_configs

        for cfg in [self._make_uniform_config(), self._make_layerwise_config()]:
            configs = compute_layer_configs(cfg)
            for i, c in enumerate(configs):
                assert c.ffn_hidden_dim % 256 == 0, (
                    f"Layer {i} FFN hidden dim {c.ffn_hidden_dim} is not a multiple of 256"
                )

    def test_ffn_hidden_dim_is_positive(self):
        from openelm_pipeline.model import compute_layer_configs

        cfg     = self._make_uniform_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.ffn_hidden_dim > 0, (
                f"Layer {i} FFN hidden dim is {c.ffn_hidden_dim}"
            )

    def test_head_dim_is_preserved(self):
        from openelm_pipeline.model import compute_layer_configs

        cfg     = self._make_uniform_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.head_dim == cfg.head_dim, (
                f"Layer {i} head_dim is {c.head_dim}, expected {cfg.head_dim}"
            )

    def test_kv_heads_never_exceed_query_heads(self):
        from openelm_pipeline.model import compute_layer_configs

        cfg     = self._make_layerwise_config()
        configs = compute_layer_configs(cfg)

        for i, c in enumerate(configs):
            assert c.num_kv_heads <= c.num_query_heads, (
                f"Layer {i}: kv_heads ({c.num_kv_heads}) > query_heads ({c.num_query_heads})"
            )


class TestWeightAdapter:
    """Verify that the weight adapter is a correct pass-through."""

    def test_adapter_returns_all_weights(self):
        import numpy as np
        from openelm_pipeline.weight_adapters import openelm_weight_adapter

        mock_weights = {
            "transformer.token_embeddings.weight":    np.zeros((32000, 1024)),
            "transformer.layers.0.attn_norm.weight":  np.ones((1024,)),
            "transformer.layers.0.attn.qkv_proj.weight": np.zeros((512, 1024)),
            "transformer.norm.weight":                np.ones((1024,)),
        }

        result = openelm_weight_adapter(mock_weights)
        assert len(result) == len(mock_weights), (
            f"Expected {len(mock_weights)} weights, got {len(result)}"
        )

    def test_adapter_preserves_names(self):
        import numpy as np
        from openelm_pipeline.weight_adapters import openelm_weight_adapter

        mock_weights = {
            "transformer.token_embeddings.weight":    np.zeros((32000, 1024)),
            "transformer.layers.0.ffn.proj_1.weight": np.zeros((512, 1024)),
        }

        result = openelm_weight_adapter(mock_weights)
        for name in mock_weights:
            assert name in result, f"Weight '{name}' missing from adapter output"

    def test_adapter_preserves_tensor_values(self):
        import numpy as np
        from openelm_pipeline.weight_adapters import openelm_weight_adapter

        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result   = openelm_weight_adapter({"transformer.norm.weight": original})

        adapted    = result["transformer.norm.weight"]
        underlying = adapted.data if hasattr(adapted, "data") else adapted

        assert np.array_equal(original, underlying), (
            "Tensor values were modified by the adapter"
        )

    def test_adapter_handles_empty_input(self):
        from openelm_pipeline.weight_adapters import openelm_weight_adapter

        result = openelm_weight_adapter({})
        assert result == {}


@pytest.mark.skipif(
    not model_available,
    reason="Model weights not downloaded. Run step_02 first."
)
class TestModelWeightFiles:
    """Verify downloaded model weight keys match model.py expectations."""

    def test_weight_file_exists(self):
        weight_file = MODEL_PATH / "model.safetensors"
        assert weight_file.exists(), (
            f"model.safetensors not found at {weight_file}"
        )

    def test_embedding_weight_exists(self):
        from openelm_pipeline.weight_adapters import list_weight_names

        names = set(list_weight_names(str(MODEL_PATH)))
        assert "transformer.token_embeddings.weight" in names

    def test_final_norm_weight_exists(self):
        from openelm_pipeline.weight_adapters import list_weight_names

        names = set(list_weight_names(str(MODEL_PATH)))
        assert "transformer.norm.weight" in names

    def test_layer_zero_weights_exist(self):
        from openelm_pipeline.weight_adapters import list_weight_names

        names = set(list_weight_names(str(MODEL_PATH)))

        required = [
            "transformer.layers.0.attn_norm.weight",
            "transformer.layers.0.attn.qkv_proj.weight",
            "transformer.layers.0.attn.out_proj.weight",
            "transformer.layers.0.ffn_norm.weight",
            "transformer.layers.0.ffn.proj_1.weight",
            "transformer.layers.0.ffn.proj_2.weight",
        ]

        for key in required:
            assert key in names, (
                f"Required weight '{key}' not found in model.safetensors"
            )

    def test_config_json_exists_and_readable(self):
        import json

        config_file = MODEL_PATH / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        with open(config_file) as f:
            config = json.load(f)

        assert "architectures" in config
        assert "OpenELMForCausalLM" in config["architectures"], (
            f"Expected 'OpenELMForCausalLM' in architectures, "
            f"got: {config['architectures']}"
        )
