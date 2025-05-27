"""
llama model uploader package.
a comprehensive toolkit for converting and uploading fine-tuned llama-3.2 models to hugging face hub.
"""

__version__ = "1.0.0"
__author__ = "Juan Vásquez"
__email__ = "juanmvs@pm.me"

# import main functions for easy access.
from .file_validator import validate_model_files, list_model_files, print_file_summary
from .model_converter import convert_llama_checkpoint, convert_model_weights
from .tokenizer_creator import create_tokenizer_files
from .hf_uploader import upload_model_to_hf
from .model_tester import test_model_loading_full

__all__ = [
    "validate_model_files",
    "list_model_files",
    "print_file_summary",
    "convert_llama_checkpoint",
    "convert_model_weights",
    "create_tokenizer_files",
    "upload_model_to_hf",
    "test_model_loading_full",
    "detect_llama_version",
    "get_version_info",
    "LlamaVersion",
    "ModelSize",
]
