# 🦙 custom `llama 3.2` model uploader

A comprehensive Python toolkit for converting and uploading fine-tuned [llama 3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/) models to the [hugging face hub](https://huggingface.co/models). This tool automatically converts from  original Llama checkpoint format to huggingface-compatible format with full tokenizer support.

## ✨ Features

- 🔄 **Automatic Format Conversion**: Converts Llama checkpoint format to HuggingFace format
- 🎯 **Smart Tokenizer Handling**: Works with or without SentencePiece (this library has some issues in ARM-based Apple systems), includes fallback mechanisms
- 📦 **Modular Architecture**: Clean, reusable components for different tasks
- 🧠 **Memory Efficient**: Handles large models with intelligent memory management
- 🔒 **Secure Upload**: Supports private repositories and token authentication
- 🧪 **Built-in Testing**: Validates model loading after upload
- 📊 **Progress Tracking**: Clear status messages and file size reporting
- 🛡️ **Error Resilience**: Comprehensive error handling and fallback strategies
- ⚡ **UV-Powered**: Uses UV for fast, reliable dependency management

## 🚀 Quick Start

### Installation with [uv](https://huggingface.co/models) (recommended)

0. Install `uv` if you haven't already
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone this repository
```
git clone https://github.com/juanmvsa/nlp-scripts/tree/main/llama3-2-model-uploader
```

2. Create the virtual environment 
```
cd llama3-2-model-uploader
```

```
uv venv
```

```
source .venv/bin/activate  # on windows: .venv\Scripts\activate
```

```
uv pip install -e .
```

3. Install the necessary dependencies

Install with optional dependencies
```
uv pip install -e ".[all]"  # includes accelerate and sentencepiece
```

Set your `huggingface` token
```
export HF_TOKEN=your_huggingface_token_here
```

### Alternative Installation Methods

Install without `sentencepiece` (in case of build issues)
```bash
uv pip install -e ".[accelerate]"
```

Development installation
```
uv pip install -e ".[dev]"
```

Install specific extras
```
uv pip install -e ".[sentencepiece,accelerate]"
```

### Basic Usage

#### Using the installed command

```bash
llama3-2-upload \
  --model_path ./your_llama32_model \
  --repo_name your-username/your-model-name
```

#### or using python directly

```
uv run python upload_model.py \
  --model_path ./your_llama32_model \
  --repo_name your-username/your-model-name
```

### Full Example

```bash
llama3-2-upload \
  --model_path ./my_finetuned_llama \
  --repo_name johndoe/llama32-finetuned-model \
  --private \
  --test
```

## 📁 Project Structure

```
llama3-2-model-uploader/
├── upload_model.py          # 🎯 main entry point
├── file_validator.py        # 🔍 file validation utilities
├── model_converter.py       # 🔄 format conversion logic
├── tokenizer_creator.py     # 🎨 tokenizer file creation
├── hf_uploader.py           # ⬆️ huggingface hub integration
├── model_tester.py          # 🧪 model testing utilities
├── requirements.txt         # 📦 python dependencies
└── README.md               # 📖 documentation
```

## 📋 Requirements

### Input Files (Required)

Your `model` folder must contain:

```
your_model_folder/
├── consolidated.00.pth      # model weights (part 0)
├── consolidated.01.pth      # model weights (part 1)
├── ...                      # additional weight files
├── consolidated.XX.pth      # model weights (part X)
├── params.json              # model parameters
├── tokenizer.model          # sentencepiece tokenizer
└── checklist.chk           # optional (ignored during upload)
```

### Generated Files

The tool automatically creates:

- `config.json` - HuggingFace model configuration
- `pytorch_model.bin` - merged model weights
- `tokenizer_config.json` - tokenizer configuration
- `special_tokens_map.json` - special tokens mapping
- `tokenizer.json` - fast tokenizer (when possible)
- `README.md` - model card with usage instructions

## 🛠️ Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--model_path` | path to your model folder | ✅ |
| `--repo_name` | huggingface repo (`username/model-name`) | ✅ |
| `--hf_token` | huggingface token (or set `HF_TOKEN` env var) | ⚠️ |
| `--private` | make repository private | ❌ |
| `--test` | test model loading after upload | ❌ |
| `--skip_conversion` | skip format conversion | ❌ |

## 🔧 Advanced Usage

### `uv` commands

#### Environment Management

0. Create and activate the virtual environment
```bash
uv venv
```

Linux/MacOS
```
source .venv/bin/activate  
```

or (Windows)
```
.venv\Scripts\activate
```

1. Install project in development mode

```
uv pip install -e .
```

2. Install with all optional dependencies
```
uv pip install -e ".[all]"
```

3. Update dependencies
```
uv pip install --upgrade -e .
```

#### Running the script

0. Using the installed command (recommended)
```bash
llama3.2-upload --model_path ./model --repo_name user/model
```

1. Using `uv run` (runs in an isolated environment)
```
uv run llama3-2-upload --model_path ./model --repo_name user/model
```

# using python directly
```
uv run python upload_model.py --model_path ./model --repo_name user/model
```

### Environment Variables

0. Set your huggingface token
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

1. Run with the environment token
```
llama3-2-upload --model_path ./model --repo_name user/model
```

### Programmatic Usage

```python
from file_validator import validate_model_files
from model_converter import convert_llama_checkpoint
from hf_uploader import upload_model_to_hf

# validate files
if validate_model_files("./my_model"):
    # convert format
    convert_llama_checkpoint("./my_model")
    
    # upload to hub
    upload_model_to_hf("./my_model", "user/model", "hf_token")
```

### Testing Only

0. Upload without testing
```bash
llama3-2-upload --model_path ./model --repo_name user/model
```
  

1. Test an existing repository
```
uv run python -c "
from model_tester import test_model_loading_full
test_model_loading_full('user/model', 'your_token')
"
```

## 🐍 Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Core requirements

All dependencies are defined in the `pyproject.toml` file:

```toml
dependencies = [
    "huggingface_hub>=0.19.0",    # hub integration
    "transformers>=4.43.0",       # model compatibility
    "torch>=2.0.0",              # pytorch backend
    "tokenizers>=0.15.0",        # tokenizer handling
    "requests>=2.25.0",          # http requests
    "tqdm>=4.64.0",              # progress bars
    "packaging>=20.0",           # version parsing
]
```

### Optional Dependencies

```toml
[project.optional-dependencies]
# performance improvements
accelerate = [
    "accelerate>=0.24.0",
    "safetensors>=0.4.0",
]

# tokenizer conversion (may have build issues on some platforms)
sentencepiece = [
    "sentencepiece>=0.1.97,<0.2.0",
]

# development tools
dev = [
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pytest>=7.4.0",
    "ruff>=0.0.290",
]

# all optional dependencies
all = ["llama3-2-model-uploader[accelerate,sentencepiece]"]
```

### SentencePiece Installation Issues

If you encounter build errors with `sentencepiece` on MacOS ARM64:

#### Option 1: use conda
```bash
conda install -c conda-forge sentencepiece
```

Install without `sentencepiece` extra
```
uv pip install -e ".[accelerate]"  # 
```

#### option 2: use `homebrew` + `uv`

```
brew install protobuf
```

```
uv pip install -e ".[all]"
```

#### Option 3: skip `sentencepiece` (the script will use the defined fallbacks)

Install without `sentencepiece`
```
uv pip install -e ".[accelerate]"   
```

#### Option 4: use the pre-built wheel
```
uv pip install sentencepiece --only-binary=sentencepiece
```

## 📖 Usage Examples

### Example 1: Basic Upload

```bash
# simple upload with default settings
llama3-2-upload \
  --model_path ./llama32_finetuned \
  --repo_name myusername/llama32-chatbot
```

### Example 2: Private Model with Testing

```bash
# upload private model and test loading
llama3-2-upload \
  --model_path ./sensitive_model \
  --repo_name company/internal-model \
  --private \
  --test \
  --hf_token hf_xxxxxxxxxxxxxxxxxxxx
```

### Example 3: Skip Conversion

```bash
# upload pre-converted model files
llama3-2-upload \
  --model_path ./already_converted_model \
  --repo_name user/converted-model \
  --skip_conversion
```

### Example 4: Using `uv run`

```bash
# run without installing (uses temporary environment)
uv run --with llama-model-uploader llama3-2-upload \
  --model_path ./model \
  --repo_name user/model
```

## 🔍 Troubleshooting

### Common Issues

**❌ "Missing required files"**
```bash
# ensure your model folder contains:
ls your_model_folder/
# should show: consolidated.*.pth, params.json, tokenizer.model
```

**❌ "Token verification failed"**
```bash
# check your token has write permissions
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

**❌ "`sentencepiece` build failed"**
```bash
# install via conda instead
conda install -c conda-forge sentencepiece
# or skip sentencepiece (tool will use fallbacks)
```

**❌ "Out of memory during conversion"**
```bash
# the tool automatically handles this with fallbacks
# for very large models, it will copy the first checkpoint instead of merging
```

### Debug Mode

```bash
# run with python's verbose output
uv run python -v upload_model.py --model_path ./model --repo_name user/model

# check file sizes
uv run python -c "
from file_validator import list_model_files, print_file_summary
files = list_model_files('./your_model')
print_file_summary('./your_model', files)
"

# run with uv verbose mode
uv --verbose pip install -e .
```

## 🧪 Testing Your Upload

After uploading, test your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# load your uploaded model
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model")
model = AutoModelForCausalLM.from_pretrained("your-username/your-model")

# test generation
prompt = "Hello, I am a Llama model"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

