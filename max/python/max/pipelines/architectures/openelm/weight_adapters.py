"""Weight adapter and inspection utilities for OpenELM safetensors weights."""


def openelm_weight_adapter(
    state_dict,
    huggingface_config=None,
    pipeline_config=None,
    **unused_kwargs,
):
    """
    Pass-through weight adapter for OpenELM.

    OpenELM's safetensors key names already match the names used in model.py,
    so no renaming is needed. Each tensor is wrapped in WeightData to satisfy
    the SupportedArchitecture API contract.
    """
    from max.graph.weights import WeightData
    from max.dtype import DType

    if not isinstance(state_dict, dict):
        return state_dict

    return {
        name: (
            tensor
            if isinstance(tensor, WeightData)
            else WeightData(
                data=tensor,
                name=name,
                dtype=DType.from_numpy(tensor.dtype),
                shape=list(tensor.shape),
            )
        )
        for name, tensor in state_dict.items()
    }


def list_weight_names(model_dir: str) -> list[str]:
    """Return all weight key names found in a safetensors model directory.

    Handles both single-file and sharded layouts. Useful for debugging key
    mismatches between the safetensors file and model.py weight references.
    """
    from pathlib import Path
    from safetensors import safe_open

    model_path = Path(model_dir)
    names = []

    safetensor_files = (
        list(model_path.glob("model.safetensors")) +
        list(model_path.glob("model-*.safetensors"))
    )

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt") as f:
            names.extend(f.keys())

    return sorted(names)
