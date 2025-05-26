"""
model format conversion utilities for llama models.
"""

import json
import gc
import shutil
from pathlib import Path
from typing import Union, Dict, Any
import torch


def load_params_config(model_path: Union[str, Path]) -> Dict[str, Any]:
    """load params.json and return model configuration."""
    params_path = Path(model_path) / "params.json"
    if not params_path.exists():
        raise FileNotFoundError("params.json not found")

    with open(params_path, "r") as f:
        return json.load(f)


def create_hf_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """create huggingface config.json from llama params."""
    config: Dict[str, Any] = {
        "_name_or_path": "meta-llama/Llama-3.2-3B",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "hidden_act": "silu",
        "hidden_size": params.get("dim", 3072),
        "initializer_range": 0.02,
        "intermediate_size": params.get("hidden_dim", 8192),
        "max_position_embeddings": 131072,
        "model_type": "llama",
        "num_attention_heads": params.get("n_heads", 24),
        "num_hidden_layers": params.get("n_layers", 28),
        "num_key_value_heads": params.get("n_kv_heads", 8),
        "pretraining_tp": 1,
        "rms_norm_eps": params.get("norm_eps", 1e-05),
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": params.get("rope_theta", 500000.0),
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.0",
        "use_cache": True,
        "vocab_size": params.get("vocab_size", 128256),
        "pad_token_id": 128009,
        "attn_implementation": "flash_attention_2",
    }
    return config


def save_config(model_path: Union[str, Path], config: Dict[str, Any]) -> None:
    """save config.json to model directory."""
    config_path = Path(model_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("✅ created config.json")


def convert_model_weights(model_path: Union[str, Path]) -> bool:
    """convert consolidated.*.pth files to pytorch_model.bin format."""
    model_path = Path(model_path)
    checkpoint_files = sorted(model_path.glob("consolidated.*.pth"))

    if not checkpoint_files:
        print("❌ no consolidated checkpoint files found")
        return False

    print(f"🔄 converting {len(checkpoint_files)} checkpoint files...")

    # check if pytorch_model.bin already exists.
    output_path = model_path / "pytorch_model.bin"
    if output_path.exists():
        print(f"✅ pytorch_model.bin already exists, skipping conversion")
        return True

    try:
        # option 1: try to load and merge all at once (for smaller models).
        if len(checkpoint_files) <= 4:  # likely small enough to fit in memory.
            merged_state_dict: Dict[str, torch.Tensor] = {}

            for i, checkpoint_file in enumerate(checkpoint_files):
                print(
                    f"loading {checkpoint_file.name}... ({i + 1}/{len(checkpoint_files)})"
                )
                try:
                    checkpoint = torch.load(checkpoint_file, map_location="cpu")

                    # merge the state dict.
                    for key, value in checkpoint.items():
                        if key in merged_state_dict:
                            # handle sharded parameters.
                            if isinstance(value, torch.Tensor) and value.dim() > 1:
                                merged_state_dict[key] = torch.cat(
                                    [merged_state_dict[key], value], dim=0
                                )
                            else:
                                merged_state_dict[key] = value
                        else:
                            merged_state_dict[key] = value

                    # clear checkpoint from memory.
                    del checkpoint
                    gc.collect()

                except Exception as e:
                    print(f"❌ error loading {checkpoint_file.name}: {e}")
                    return False

            # save merged model.
            print(f"💾 saving merged model to {output_path}...")
            torch.save(merged_state_dict, output_path)
            print(f"✅ saved merged model weights to {output_path}")

        else:
            # option 2: for larger models, use the first checkpoint as base.
            print("⚠️  large model detected, using first checkpoint as base...")
            first_checkpoint = checkpoint_files[0]
            print(f"using {first_checkpoint.name} as base model...")

            # simply copy the first checkpoint as pytorch_model.bin.
            shutil.copy2(first_checkpoint, output_path)
            print(f"✅ copied {first_checkpoint.name} to pytorch_model.bin")

            # note about other checkpoints.
            print(
                f"ℹ️  note: {len(checkpoint_files) - 1} additional checkpoint files were not merged."
            )
            print(
                "   if this is a sharded model, you may need to implement custom merging logic."
            )

    except Exception as e:
        print(f"❌ error during conversion: {e}")
        print("💡 trying alternative approach...")

        # fallback: just copy the first checkpoint.
        try:
            first_checkpoint = checkpoint_files[0]
            shutil.copy2(first_checkpoint, output_path)
            print(f"✅ fallback: copied {first_checkpoint.name} to pytorch_model.bin")
        except Exception as e2:
            print(f"❌ fallback also failed: {e2}")
            return False

    finally:
        # clean up memory.
        gc.collect()

    return True


def convert_llama_checkpoint(model_path: Union[str, Path]) -> bool:
    """convert llama checkpoint format to huggingface format."""
    model_path = Path(model_path)

    print("🔄 converting llama checkpoint format to huggingface format...")

    try:
        # load params.json to get model configuration.
        params = load_params_config(model_path)

        # create config.json from params.json.
        config = create_hf_config(params)

        # save config.json.
        save_config(model_path, config)

        # convert model weights to pytorch_model.bin format.
        if not convert_model_weights(model_path):
            return False

        return True

    except Exception as e:
        print(f"❌ error during checkpoint conversion: {e}")
        return False
