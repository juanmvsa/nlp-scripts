#!/usr/bin/env python3
"""
setup script for llama-model-uploader.
note: this is mainly for compatibility - use pyproject.toml with uv for modern setup.
"""

from setuptools import setup, find_packages

# read the contents of readme file.
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="llama-model-uploader",
    version="1.0.0",
    description="A comprehensive toolkit for converting and uploading fine-tuned Llama models to Hugging Face Hub",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="juanmvs@pm.me",
    url="https://github.com/juanmvsa/nlp-scripts/llama-model-uploader",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "huggingface_hub>=0.19.0",
        "transformers>=4.43.0",
        "torch>=2.0.0",
        "tokenizers>=0.15.0",
        "requests>=2.25.0",
        "tqdm>=4.64.0",
        "packaging>=20.0",
    ],
    extras_require={
        "accelerate": [
            "accelerate>=0.24.0",
            "safetensors>=0.4.0",
        ],
        "spacy": [
            "spacy>=3.7.0",
            "spacy-transformers>=1.3.0",
        ],
        "sentencepiece": [
            "sentencepiece>=0.1.97,<0.2.0",
        ],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.0.290",
        ],
        "all": [
            "accelerate>=0.24.0",
            "safetensors>=0.4.0",
            "spacy>=3.7.0",
            "spacy-transformers>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llama-upload=llama_uploader.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="llama, huggingface, model, upload, machine-learning, nlp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llama-model-uploader/issues",
        "Source": "https://github.com/yourusername/llama-model-uploader",
        "Documentation": "https://github.com/yourusername/llama-model-uploader#readme",
    },
)
