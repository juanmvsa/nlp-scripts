"""
tokenizer creation utilities for llama models using spacy as primary tokenizer.
"""

import json
import shutil
from pathlib import Path
from typing import Union, Dict, Any, List, Optional


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
    return {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>", "pad_token": "<|eot_id|>"}


def create_tokenizer_json_from_spacy(
    vocab: List[str], special_tokens: Dict[str, int]
) -> Dict[str, Any]:
    """create tokenizer.json structure using spacy vocabulary."""
    # create vocabulary entries for tokenizer.json.
    vocab_entries = []
    for i, token in enumerate(vocab):
        vocab_entries.append([token, float(i)])  # token and score.

    # add special tokens to vocab if not present.
    for token, token_id in special_tokens.items():
        if token not in vocab:
            vocab_entries.append([token, float(token_id)])

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": special_tokens.get("<|begin_of_text|>", 128000),
                "content": "<|begin_of_text|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": special_tokens.get("<|eot_id|>", 128009),
                "content": "<|eot_id|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": {"type": "NFKC"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<|begin_of_text|>": {
                    "id": "<|begin_of_text|>",
                    "ids": [special_tokens.get("<|begin_of_text|>", 128000)],
                    "tokens": ["<|begin_of_text|>"],
                },
                "<|eot_id|>": {
                    "id": "<|eot_id|>",
                    "ids": [special_tokens.get("<|eot_id|>", 128009)],
                    "tokens": ["<|eot_id|>"],
                },
            },
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {"type": "WordLevel", "vocab": dict(vocab_entries), "unk_token": "<unk>"},
    }


def try_spacy_tokenizer_conversion(model_path: Union[str, Path]) -> bool:
    """try to create tokenizer.json using spacy tokenizer."""
    try:
        import spacy
        from spacy.lang.en import English

        print("✅ spacy available, creating tokenizer.json with spacy")

        # load or create spacy model.
        try:
            # try to load a transformer model if available.
            nlp = spacy.load("en_core_web_trf")
            print("✅ using spacy transformer model (en_core_web_trf)")
        except OSError:
            try:
                # fallback to medium model.
                nlp = spacy.load("en_core_web_md")
                print("✅ using spacy medium model (en_core_web_md)")
            except OSError:
                try:
                    # fallback to small model.
                    nlp = spacy.load("en_core_web_sm")
                    print("✅ using spacy small model (en_core_web_sm)")
                except OSError:
                    # create blank english model as last resort.
                    nlp = English()
                    print("✅ using blank spacy english model")

        # extract vocabulary from spacy model.
        vocab_items = list(nlp.vocab.strings)

        # add common tokens that might be missing.
        additional_tokens = [
            "<unk>",
            "<pad>",
            "<mask>",
            "<cls>",
            "<sep>",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        ]

        for token in additional_tokens:
            if token not in vocab_items:
                vocab_items.append(token)

        # ensure we have a reasonable vocabulary size.
        if len(vocab_items) < 1000:
            print(f"⚠️ small vocabulary detected ({len(vocab_items)} tokens), expanding...")
            # add more common english words.
            common_words = [
                "a",
                "an",
                "as",
                "are",
                "was",
                "were",
                "been",
                "be",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "must",
                "shall",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "her",
                "its",
                "our",
                "their",
                "mine",
                "yours",
                "ours",
                "theirs",
            ]
            vocab_items.extend([w for w in common_words if w not in vocab_items])

        # special tokens with their ids.
        special_tokens = {"<|begin_of_text|>": 128000, "<|eot_id|>": 128009, "<unk>": 0, "<pad>": 1}

        # create tokenizer.json.
        tokenizer_json = create_tokenizer_json_from_spacy(vocab_items, special_tokens)

        with open(Path(model_path) / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2)

        print(f"✅ created tokenizer.json with spacy ({len(vocab_items)} vocabulary items)")
        return True

    except ImportError:
        print("⚠️ spacy not available, will try other tokenizer options")
        return False
    except Exception as e:
        print(f"⚠️ could not create tokenizer.json with spacy: {e}")
        return False


def try_sentencepiece_conversion(model_path: Union[str, Path]) -> bool:
    """try to create tokenizer.json using sentencepiece if available (fallback)."""
    tokenizer_model_path = Path(model_path) / "tokenizer.model"

    if not tokenizer_model_path.exists():
        print("⚠️ tokenizer.model not found, skipping sentencepiece conversion")
        return False

    try:
        import sentencepiece as spm

        print("✅ sentencepiece available as fallback, creating tokenizer.json")

        # load the sentencepiece model.
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(tokenizer_model_path))

        # extract vocabulary from sentencepiece model.
        vocab_size = sp_model.get_piece_size()
        vocab_items = []

        for i in range(vocab_size):
            piece = sp_model.id_to_piece(i)
            vocab_items.append(piece)

        # special tokens.
        special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|eot_id|>": 128009,
        }

        # create tokenizer.json.
        tokenizer_json = create_tokenizer_json_from_spacy(vocab_items, special_tokens)

        with open(Path(model_path) / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2)

        print(f"✅ created tokenizer.json from sentencepiece model ({vocab_size} vocabulary items)")
        return True

    except ImportError:
        print("⚠️ sentencepiece not available")
        return False
    except Exception as e:
        print(f"⚠️ could not create tokenizer.json from sentencepiece: {e}")
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
        for file_name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
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


def install_spacy_model() -> Optional[str]:
    """try to install a spacy model if none are available."""
    try:
        import spacy
        import subprocess
        import sys

        print("🔄 no spacy models found, attempting to download en_core_web_sm...")

        # try to download the small english model.
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ successfully downloaded en_core_web_sm")
            return "en_core_web_sm"
        else:
            print(f"⚠️ failed to download spacy model: {result.stderr}")
            return None

    except Exception as e:
        print(f"⚠️ could not install spacy model: {e}")
        return None


def create_tokenizer_files(model_path: Union[str, Path]) -> None:
    """create tokenizer configuration files for huggingface compatibility."""
    model_path = Path(model_path)

    # check if tokenizer.model exists (for fallback).
    tokenizer_model_path = model_path / "tokenizer.model"
    if not tokenizer_model_path.exists():
        print("⚠️ tokenizer.model not found, proceeding with spacy-only approach")

    # create basic tokenizer config files.
    tokenizer_config = create_tokenizer_config()
    with open(model_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = create_special_tokens_map()
    with open(model_path / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    # try tokenizer creation in order of preference.
    print("🎯 tokenizer creation priority: spacy -> sentencepiece -> base model copy")

    # 1. try spacy tokenizer (preferred).
    spacy_success = try_spacy_tokenizer_conversion(model_path)

    if not spacy_success:
        # try to install spacy model and retry.
        installed_model = install_spacy_model()
        if installed_model:
            spacy_success = try_spacy_tokenizer_conversion(model_path)

    # 2. if spacy fails, try sentencepiece (fallback).
    if not spacy_success:
        sentencepiece_success = try_sentencepiece_conversion(model_path)

        # 3. if both fail, copy from base model.
        if not sentencepiece_success:
            copy_base_model_tokenizer(model_path)

    print("✅ created tokenizer configuration files")
