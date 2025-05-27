#!/usr/bin/env python3
"""
command line interface for llama-model-uploader.
main script to upload a fine-tuned llama model to hugging face hub.
converts from llama checkpoint format to huggingface format.
supports llama-3.2, llama-3.3, and experimental llama-4.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

# import our custom modules.
from .file_validator import validate_model_files
from .model_converter import convert_llama_checkpoint
from .tokenizer_creator import create_tokenizer_files
from .hf_uploader import upload_model_to_hf
from .model_tester import test_model_loading_full


def main() -> None:
    """main function to handle command line arguments and execute upload process."""
    parser = argparse.ArgumentParser(description="upload llama finetuned model to hugging face")
    parser.add_argument("--model_path", required=True, help="path to your model folder")
    parser.add_argument(
        "--repo_name", required=True, help="huggingface repo name (username/model-name)"
    )
    parser.add_argument("--hf_token", help="huggingface token (or set hf_token env var)")
    parser.add_argument("--private", action="store_true", help="make repository private")
    parser.add_argument("--test", action="store_true", help="test model loading after upload")
    parser.add_argument(
        "--skip_conversion", action="store_true", help="skip model format conversion"
    )

    args = parser.parse_args()

    # get hf token.
    hf_token: Optional[str] = args.hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ please provide huggingface token via --hf_token or hf_token env variable")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ model path {model_path} does not exist")
        return

    print(f"📁 model path: {model_path}")
    print(f"🚀 target repo: {args.repo_name}")
    print(f"🔒 private: {args.private}")
    print(f"🧪 test after upload: {args.test}")
    print(f"⏭️ skip conversion: {args.skip_conversion}")

    # validate model files.
    if not validate_model_files(model_path):
        return

    # convert llama checkpoint format to huggingface format.
    if not args.skip_conversion:
        if not convert_llama_checkpoint(model_path):
            print("❌ failed to convert model format")
            return

        # create tokenizer configuration files.
        create_tokenizer_files(model_path)
    else:
        print("⏭️ skipping model conversion as requested")

    # upload model.
    success = upload_model_to_hf(model_path, args.repo_name, hf_token, args.private)

    # test loading if requested.
    if success and args.test:
        test_model_loading_full(args.repo_name, hf_token)


if __name__ == "__main__":
    main()
