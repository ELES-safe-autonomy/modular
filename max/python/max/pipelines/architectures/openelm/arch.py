"""SupportedArchitecture registration for OpenELM."""

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.lib.config import PipelineConfig, MAXModelConfig

from openelm_pipeline.model import OpenELMModel
from openelm_pipeline.weight_adapters import openelm_weight_adapter


class OpenELMArchConfig:
    """ArchConfig implementation for the OpenELM family."""

    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len

    @classmethod
    def initialize(
        cls,
        pipeline_config: "PipelineConfig",
        model_config: "MAXModelConfig | None" = None,
    ) -> "OpenELMArchConfig":
        mc = model_config or pipeline_config.model
        hf_config = mc.huggingface_config
        model_max = getattr(hf_config, "max_context_length", 2048)
        user_max = getattr(mc, "max_length", None) or model_max
        return cls(max_seq_len=min(user_max, model_max))

    def get_max_seq_len(self) -> int:
        return self._max_seq_len


openelm_arch = SupportedArchitecture(
    name="OpenELMForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "apple/OpenELM-270M-Instruct",
        "apple/OpenELM-450M-Instruct",
        "apple/OpenELM-1_1B-Instruct",
        "apple/OpenELM-3B-Instruct",
    ],
    pipeline_model=OpenELMModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    config=OpenELMArchConfig,
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="float32",
    # bfloat16 listed for forward compatibility; Phase 1 runs float32 on CPU.
    supported_encodings={"float32", "bfloat16"},
    rope_type="normal",
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: openelm_weight_adapter,
    },
)
