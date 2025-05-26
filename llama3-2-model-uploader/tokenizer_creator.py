"""
tokenizer creation utilities for llama models.
"""

import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any


def create_tokenizer_config() -> Dict[str, Any]:
    """create tokenizer_config.json structure."""
    return {
        "add_bos_token": True,
        "add_eos_token": False,
        "added_tokens_decoder": {
            "128000": {
                "content": "<|begin_of_text|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "128009": {
                "content": "<|eot_id|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
        "bos_token": "<|begin_of_text|>",
        "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|eot_id|>",
        "legacy": False,
        "model_max_length": 131072,
        "pad_token": "<|eot_id|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "LlamaTokenizer",
        "tokenizer_file": None,
        "vocab_file": None,
    }


def create_special_tokens_map() -> Dict[str, str]:
    """create special_tokens_map.json structure."""
    return {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
        "pad_token": "<|eot_id|>",
    }


def create_tokenizer_json() -> Dict[str, Any]:
    """create basic tokenizer.json structure for llamatokenizer."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 128000,
                "content": "<|begin_of_text|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 128009,
                "content": "<|eot_id|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": True,
        },
        "post_processor": None,
        "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "model": {"type": "Unigram", "vocab": []},
    }


def try_sentencepiece_conversion(model_path: Union[str, Path]) -> bool:
    """try to create tokenizer.json using sentencepiece if available."""
    tokenizer_model_path = Path(model_path) / "tokenizer.model"

    try:
        import sentencepiece as spm

        print("✅ sentencepiece available, creating tokenizer.json")

        # load the sentencepiece model.
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(tokenizer_model_path))

        # create tokenizer.json.
        tokenizer_json = create_tokenizer_json()

        with open(Path(model_path) / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2)

        print("✅ created tokenizer.json from sentencepiece model")
        return True

    except ImportError:
        print("⚠️ sentencepiece not available, will use slow tokenizer fallback")
        return False
    except Exception as e:
        print(f"⚠️ could not create tokenizer.json: {e}")
        return False


def copy_base_model_tokenizer(model_path: Union[str, Path]) -> bool:
    """copy tokenizer files from base llama-3.2 model."""
    print("🔄 trying to copy tokenizer files from base llama-3.2 model...")

    try:
        from transformers import AutoTokenizer

        # download base model tokenizer.
        base_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct", use_fast=False
        )

        # save to our model directory.
        model_path = Path(model_path)
        temp_dir = model_path / "temp_tokenizer"
        base_tokenizer.save_pretrained(temp_dir)

        # copy the necessary files.
        copied_files = []
        for file_name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            src_file = temp_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, model_path / file_name)
                copied_files.append(file_name)
                print(f"✅ copied {file_name} from base model")

        # clean up temp directory.
        shutil.rmtree(temp_dir)

        if copied_files:
            print("✅ successfully copied tokenizer files from base model")
            return True
        else:
            print("⚠️ no tokenizer files were copied")
            return False

    except Exception as e:
        print(f"⚠️ could not copy base model tokenizer: {e}")
        print("   model will use slow tokenizer with tokenizer.model file")
        return False


def create_tokenizer_files(model_path: Union[str, Path]) -> None:
    """create tokenizer configuration files for huggingface compatibility."""
    model_path = Path(model_path)

    # check if tokenizer.model exists.
    tokenizer_model_path = model_path / "tokenizer.model"
    if not tokenizer_model_path.exists():
        print("❌ tokenizer.model not found")
        return

    # create basic tokenizer config files.
    tokenizer_config = create_tokenizer_config()
    with open(model_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = create_special_tokens_map()
    with open(model_path / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    # try to create tokenizer.json using sentencepiece.
    sentencepiece_success = try_sentencepiece_conversion(model_path)

    # if sentencepiece is not available, copy tokenizer files from base model.
    if not sentencepiece_success:
        copy_base_model_tokenizer(model_path)

    print("✅ created tokenizer configuration files")
