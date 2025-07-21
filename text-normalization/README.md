# Text Normalization

Text preprocessing and normalization utilities for NLP pipelines. Includes comprehensive text cleaning, normalization, and preprocessing functions.

## Features

- HTML/XML tag removal
- Unicode normalization
- Case normalization
- Punctuation handling
- Whitespace normalization
- English token removal
- Spanish spell checking
- Text file processing utilities

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Usage

Run the text normalization pipeline:

```bash
uv run python main.py
```

The script will process text files from a specified folder and save the normalized output.

## Modules

- `text_preprocessing.py` - File reading utilities
- `normalize_text.py` - Core text normalization functions
- `remove_english_tokens.py` - English token removal
- `save_text_in_list.py` - Output utilities
- `main.py` - Main processing pipeline

## Requirements

- Python 3.9+
- NLTK
- BeautifulSoup4
- pyspellchecker

## Development

This project uses `pyproject.toml` for dependency management and can be developed independently from other projects in the repository.