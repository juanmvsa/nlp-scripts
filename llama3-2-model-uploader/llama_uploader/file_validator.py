"""
file validation utilities for llama model uploads.
"""

from pathlib import Path
from typing import Union, List


def validate_model_files(model_path: Union[str, Path]) -> bool:
    """validate that all required model files are present."""
    model_path = Path(model_path)
    
    # check for pytorch checkpoint files (consolidated.*.pth).
    checkpoint_files = list(model_path.glob("consolidated.*.pth"))
    
    # check for params.json and tokenizer.model.
    required_files = ["params.json", "tokenizer.model"]
    
    missing_files: List[str] = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if not checkpoint_files:
        missing_files.append("model weights (consolidated.*.pth)")
    
    if missing_files:
        print(f"❌ missing required files: {missing_files}")
        return False
    
    print(f"✅ found {len(checkpoint_files)} checkpoint files and required metadata")
    return True


def list_model_files(model_path: Union[str, Path]) -> List[Path]:
    """list all files in model directory that should be uploaded."""
    model_path = Path(model_path)
    files_to_upload: List[Path] = []
    
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(model_path)
            # check if file should be ignored.
            ignore_patterns = [".git*", "__pycache__*", "*.pyc", "consolidated.*.pth", "checklist.chk"]
            if not any(rel_path.match(pattern) for pattern in ignore_patterns):
                files_to_upload.append(rel_path)
    
    return files_to_upload


def print_file_summary(model_path: Union[str, Path], files_to_upload: List[Path]) -> None:
    """print summary of files to be uploaded."""
    model_path = Path(model_path)
    
    print(f"📁 files to upload ({len(files_to_upload)}):")
    for file in sorted(files_to_upload):
        file_size = (model_path / file).stat().st_size / (1024 * 1024)  # mb.
        print(f"   - {file} ({file_size:.1f} mb)")
