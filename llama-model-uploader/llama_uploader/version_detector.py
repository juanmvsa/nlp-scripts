"""
llama version detection and configuration utilities.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Tuple
from enum import Enum


class LlamaVersion(Enum):
    """enumeration of supported llama versions."""

    LLAMA_3_2 = "3.2"
    LLAMA_3_3 = "3.3"
    LLAMA_4 = "4.0"
    UNKNOWN = "unknown"


class ModelSize(Enum):
    """enumeration of common model sizes."""

    SIZE_1B = "1B"
    SIZE_3B = "3B"
    SIZE_8B = "8B"
    SIZE_11B = "11B"
    SIZE_70B = "70B"
    SIZE_405B = "405B"
    UNKNOWN = "unknown"


def detect_llama_version(model_path: Union[str, Path]) -> Tuple[LlamaVersion, ModelSize]:
    """detect llama version and size from model files."""
    model_path = Path(model_path)

    # try to read params.json for version detection.
    params_path = model_path / "params.json"
    if params_path.exists():
        try:
            with open(params_path, "r") as f:
                params = json.load(f)

            # detect size based on model dimensions.
            hidden_size = params.get("dim", 0)
            n_layers = params.get("n_layers", 0)
            vocab_size = params.get("vocab_size", 0)

            # detect model size.
            model_size = detect_model_size(hidden_size, n_layers, vocab_size)

            # detect version based on various indicators.
            version = detect_version_from_params(params, model_path)

            return version, model_size

        except Exception as e:
            print(f"⚠️ could not parse params.json: {e}")

    # fallback: try to detect from folder name or checkpoint files.
    version = detect_version_from_path(model_path)
    model_size = detect_size_from_path(model_path)

    return version, model_size


def detect_model_size(hidden_size: int, n_layers: int, vocab_size: int) -> ModelSize:
    """detect model size based on architecture parameters."""

    # llama-3.2 and 3.3 size detection.
    if hidden_size == 2048 and n_layers in [16, 20]:
        return ModelSize.SIZE_1B
    elif hidden_size == 3072 and n_layers == 28:
        return ModelSize.SIZE_3B
    elif hidden_size == 4096 and n_layers == 32:
        return ModelSize.SIZE_8B
    elif hidden_size == 4096 and n_layers == 48:
        return ModelSize.SIZE_11B
    elif hidden_size == 8192 and n_layers == 80:
        return ModelSize.SIZE_70B
    elif hidden_size == 16384 and n_layers == 126:
        return ModelSize.SIZE_405B

    # fallback size detection for llama-4 (parameters may differ).
    total_params = estimate_parameters(hidden_size, n_layers, vocab_size)

    if total_params < 2e9:
        return ModelSize.SIZE_1B
    elif total_params < 5e9:
        return ModelSize.SIZE_3B
    elif total_params < 12e9:
        return ModelSize.SIZE_8B
    elif total_params < 15e9:
        return ModelSize.SIZE_11B
    elif total_params < 100e9:
        return ModelSize.SIZE_70B
    elif total_params >= 100e9:
        return ModelSize.SIZE_405B

    return ModelSize.UNKNOWN


def estimate_parameters(hidden_size: int, n_layers: int, vocab_size: int) -> float:
    """estimate total parameters based on architecture."""
    if hidden_size == 0 or n_layers == 0:
        return 0

    # rough parameter estimation for transformer models.
    # this is approximate and may not be exact for all architectures.
    embedding_params = vocab_size * hidden_size
    layer_params = n_layers * (
        4 * hidden_size * hidden_size  # attention weights
        + 8 * hidden_size * hidden_size  # ffn weights
        + 2 * hidden_size  # layer norms
    )

    return embedding_params + layer_params


def detect_version_from_params(params: Dict[str, Any], model_path: Path) -> LlamaVersion:
    """detect llama version from params.json content."""

    # check for version indicators in params.
    vocab_size = params.get("vocab_size", 0)
    rope_theta = params.get("rope_theta", 0)

    # check for llama-4 indicators (hypothetical).
    if "version" in params:
        version_str = str(params["version"])
        if "4." in version_str or version_str.startswith("4"):
            return LlamaVersion.LLAMA_4
        elif "3.3" in version_str:
            return LlamaVersion.LLAMA_3_3
        elif "3.2" in version_str:
            return LlamaVersion.LLAMA_3_2

    # detect based on vocabulary size and other parameters.
    if vocab_size == 128256:  # llama 3.x vocabulary.
        # check rope scaling or other 3.3/4.0 specific features.
        use_scaled_rope = params.get("use_scaled_rope", False)

        # check for newer features that might indicate 3.3 or 4.0.
        if "attention_variant" in params or "moe_config" in params:
            return LlamaVersion.LLAMA_4  # hypothetical indicators.
        elif use_scaled_rope and rope_theta > 500000:
            return LlamaVersion.LLAMA_3_3  # educated guess.
        else:
            return LlamaVersion.LLAMA_3_2

    # check for potential llama-4 vocabulary size (hypothetical).
    elif vocab_size > 130000:
        return LlamaVersion.LLAMA_4

    return LlamaVersion.UNKNOWN


def detect_version_from_path(model_path: Path) -> LlamaVersion:
    """detect version from model path or folder name."""
    path_str = str(model_path).lower()

    if "llama-4" in path_str or "llama4" in path_str:
        return LlamaVersion.LLAMA_4
    elif "llama-3.3" in path_str or "llama3.3" in path_str:
        return LlamaVersion.LLAMA_3_3
    elif "llama-3.2" in path_str or "llama3.2" in path_str:
        return LlamaVersion.LLAMA_3_2
    elif "llama-3" in path_str or "llama3" in path_str:
        return LlamaVersion.LLAMA_3_2  # default to 3.2 for generic llama-3.

    return LlamaVersion.UNKNOWN


def detect_size_from_path(model_path: Path) -> ModelSize:
    """detect model size from path name."""
    path_str = str(model_path).lower()

    if "1b" in path_str:
        return ModelSize.SIZE_1B
    elif "3b" in path_str:
        return ModelSize.SIZE_3B
    elif "8b" in path_str:
        return ModelSize.SIZE_8B
    elif "11b" in path_str:
        return ModelSize.SIZE_11B
    elif "70b" in path_str:
        return ModelSize.SIZE_70B
    elif "405b" in path_str:
        return ModelSize.SIZE_405B

    return ModelSize.UNKNOWN


def get_base_model_name(version: LlamaVersion, size: ModelSize) -> str:
    """get the appropriate base model name for huggingface."""

    if version == LlamaVersion.LLAMA_3_2:
        if size == ModelSize.SIZE_1B:
            return "meta-llama/Llama-3.2-1B"
        elif size == ModelSize.SIZE_3B:
            return "meta-llama/Llama-3.2-3B"
        elif size == ModelSize.SIZE_11B:
            return "meta-llama/Llama-3.2-11B-Vision"  # vision model.
        elif size == ModelSize.SIZE_70B:
            return "meta-llama/Llama-3.2-70B"
        elif size == ModelSize.SIZE_405B:
            return "meta-llama/Llama-3.2-405B"

    elif version == LlamaVersion.LLAMA_3_3:
        if size == ModelSize.SIZE_70B:
            return "meta-llama/Llama-3.3-70B-Instruct"
        # add other 3.3 sizes when available.

    elif version == LlamaVersion.LLAMA_4:
        # hypothetical llama-4 model names.
        if size == ModelSize.SIZE_8B:
            return "meta-llama/Llama-4-8B"
        elif size == ModelSize.SIZE_70B:
            return "meta-llama/Llama-4-70B"
        # add other sizes when available.

    # fallback to generic names.
    return f"meta-llama/Llama-{version.value}-{size.value}"


def get_version_info(version: LlamaVersion, size: ModelSize) -> Dict[str, Any]:
    """get version-specific configuration information."""

    info = {
        "version": version.value,
        "size": size.value,
        "base_model": get_base_model_name(version, size),
        "tags": [f"llama-{version.value}", "fine-tuned", "causal-lm"],
        "supported": True,
    }

    if version == LlamaVersion.LLAMA_3_2:
        info.update(
            {
                "bos_token_id": 128000,
                "eos_token_id": 128009,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
                "rope_theta": 500000.0,
                "transformers_version": "4.43.0",
            }
        )

    elif version == LlamaVersion.LLAMA_3_3:
        info.update(
            {
                "bos_token_id": 128000,
                "eos_token_id": 128009,  # may change.
                "vocab_size": 128256,  # may change.
                "max_position_embeddings": 131072,  # may change.
                "rope_theta": 500000.0,
                "transformers_version": "4.46.0",
                "supported": True,  # likely compatible.
            }
        )

    elif version == LlamaVersion.LLAMA_4:
        info.update(
            {
                "bos_token_id": 128000,  # hypothetical.
                "eos_token_id": 128009,  # hypothetical.
                "vocab_size": 128256,  # may be different.
                "max_position_embeddings": 131072,  # may be different.
                "rope_theta": 500000.0,
                "transformers_version": "4.50.0",  # hypothetical.
                "supported": False,  # experimental support.
                "warning": "llama-4 support is experimental and may require manual configuration.",
            }
        )

    else:
        info.update(
            {
                "supported": False,
                "warning": "unknown llama version detected. manual configuration may be required.",
            }
        )

    return info
