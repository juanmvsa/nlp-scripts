# NLP Scripts

A collection of independent NLP utilities and tools for various text processing and machine learning tasks.

## Projects

### 1. llama-model-uploader
A comprehensive toolkit for converting and uploading fine-tuned Llama-3.2 models to Hugging Face Hub.

**Location:** `llama-model-uploader/`

**Installation:**
```bash
cd llama-model-uploader
uv sync
```

**Usage:**
```bash
uv run llama-upload --help
```

### 2. llama32-mlx-finetune
Llama 3.2 MLX fine-tuning script for Apple Silicon.

**Location:** `llama32-mlx-finetune/`

**Installation:**
```bash
cd llama32-mlx-finetune
uv sync
```

**Usage:**
```bash
uv run python main.py
```

### 3. text-normalization
Text preprocessing and normalization utilities for NLP pipelines.

**Location:** `text-normalization/`

**Installation:**
```bash
cd text-normalization
uv sync
```

**Usage:**
```bash
uv run python main.py
```

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

Each project can be installed independently using `uv`:

```bash
# Navigate to the desired project
cd <project-name>

# Install dependencies
uv sync

# Run the project
uv run <command>
```

## Development

Each project is self-contained with its own `pyproject.toml` file and can be developed independently.