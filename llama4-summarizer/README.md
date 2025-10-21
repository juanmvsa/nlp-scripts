# llama4-summarizer

document summarization tool using meta-llama/llama-4-maverick-17b-128e-instruct from huggingface.

## features

- summarize single documents or entire folders recursively.
- supports multiple file formats: txt, md, and pdf.
- leverages llama-4 maverick's native pdf parsing capabilities.
- executable with `uv` - no manual dependency management required.
- maintains folder structure when processing directories.

## requirements

- python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- huggingface account with access to meta-llama/llama-4-maverick-17b-128e-instruct (gated model)

## installation

1. install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. clone or download this repository:
```bash
git clone <repository-url>
cd llama4-summarizer
```

3. make the script executable:
```bash
chmod +x summarize.py
```

## usage

the script takes three arguments:
1. huggingface authentication token
2. input path (file or folder)
3. output path (file or folder)

### basic syntax

```bash
./summarize.py <HF_TOKEN> <INPUT> <OUTPUT>
```

or using uv run:

```bash
uv run summarize.py <HF_TOKEN> <INPUT> <OUTPUT>
```

### examples

#### summarize a single file

```bash
./summarize.py hf_yourtoken123 document.txt summary.txt
```

```bash
./summarize.py hf_yourtoken123 research_paper.pdf summary.txt
```

#### summarize all files in a folder

```bash
./summarize.py hf_yourtoken123 ./documents/ ./summaries/
```

this will:
- recursively find all `.txt`, `.md`, and `.pdf` files in `./documents/`
- process each file and generate summaries
- save summaries to `./summaries/` maintaining the original folder structure

#### example folder structure

input:
```
documents/
  ├── paper1.pdf
  ├── notes.md
  └── research/
      ├── article.txt
      └── report.pdf
```

output after running:
```bash
./summarize.py hf_token ./documents/ ./summaries/
```

```
summaries/
  ├── paper1.txt
  ├── notes.txt
  └── research/
      ├── article.txt
      └── report.txt
```

## getting your huggingface token

1. create an account at [huggingface.co](https://huggingface.co)
2. request access to [meta-llama/llama-4-maverick-17b-128e-instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)
3. once approved, go to [settings > access tokens](https://huggingface.co/settings/tokens)
4. create a new token with read permissions
5. use this token when running the script

## how it works

1. **file collection**: the script scans the input path for supported files (txt, md, pdf)
2. **model loading**: loads the llama-4 maverick model from huggingface
3. **processing**: for each file:
   - reads the content (pdf files are handled natively by the model)
   - generates a comprehensive summary using the llm
   - saves the summary to the output location
4. **progress tracking**: displays progress as `[current/total]` for batch processing

## dependencies

all dependencies are automatically managed by `uv`:

- transformers >= 4.40.0
- torch >= 2.0.0
- accelerate >= 0.27.0
- sentencepiece >= 0.2.0
- protobuf >= 3.20.0

## notes

- the first run will download the model (~17gb), which may take some time.
- processing speed depends on your hardware (gpu recommended).
- pdf files are processed using the model's native pdf parsing capabilities.
- empty or unreadable files are automatically skipped with a warning.

## troubleshooting

**error: huggingface authentication failed**
- ensure your token is valid and has read permissions.
- verify you have access to the gated model.

**error: out of memory**
- the model requires significant gpu memory.
- consider using a machine with more vram or reduce batch size.

**error: file not found**
- check that input paths exist and are accessible.
- ensure file extensions are lowercase or the script handles case-insensitive matching.

## license

this project uses the meta llama-4 maverick model, which is subject to meta's license terms.
