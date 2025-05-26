# 🦙 custom `llama 3.2` model uploader

a comprehensive python toolkit for converting and uploading fine-tuned [llama 3.2](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/) models to the [hugging face hub](https://huggingface.co/models).

this tool automatically converts from  original llama checkpoint format to huggingface-compatible format with full tokenizer support.

---

## ✨ features

- 🔄 **automatic format conversion**: converts llama checkpoint format to huggingface format
- 🎯 **smart tokenizer handling**: works with or without sentencepiece (this library has some issues in arm-based apple systems), includes fallback mechanisms
- 📦 **modular architecture**: clean, reusable components for different tasks
- 🧠 **memory efficient**: handles large models with intelligent memory management
- 🔒 **secure upload**: supports private repositories and token authentication
- 🧪 **built-in testing**: validates model loading after upload
- 📊 **progress tracking**: clear status messages and file size reporting
- 🛡️ **error resilience**: comprehensive error handling and fallback strategies
- ⚡ **`uv`-powered**: uses `uv` for fast, reliable dependency management

---

## 🚀 quick start

### installation with [uv](https://huggingface.co/models) (recommended)

#### 0. install `uv` if you haven't already
```bash
curl -lssf https://astral.sh/uv/install.sh | sh
```

#### 1. clone this repository
```
git clone https://github.com/juanmvsa/nlp-scripts/tree/main/llama3-2-model-uploader
```

#### 2. create the virtual environment 
```
cd llama3-2-model-uploader
```

```
uv venv
```

```
source .venv/bin/activate  # on windows: .venv\scripts\activate
```

```
uv pip install -e .
```

#### 3. install with optional dependencies
```
uv pip install -e ".[all]"  # includes accelerate and sentencepiece
```

#### 4. set your `huggingface` token
```
export hf_token=your_huggingface_token_here
```

### alternative installation methods

#### install without `sentencepiece` (in case of build issues)
```bash
uv pip install -e ".[accelerate]"
```

#### development installation
```
uv pip install -e ".[dev]"
```

#### install specific extras
```
uv pip install -e ".[sentencepiece,accelerate]"
```

---

### basic usage

#### using the installed command

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

### full example

```bash
llama3-2-upload \
  --model_path ./my_finetuned_llama \
  --repo_name johndoe/llama32-finetuned-model \
  --private \
  --test
```
---

## 📁 project structure

```
llama3-2-model-uploader/
├── upload_model.py          # 🎯 main entry point
├── file_validator.py        # 🔍 file validation utilities
├── model_converter.py       # 🔄 format conversion logic
├── tokenizer_creator.py     # 🎨 tokenizer file creation
├── hf_uploader.py           # ⬆️ huggingface hub integration
├── model_tester.py          # 🧪 model testing utilities
├── requirements.txt         # 📦 python dependencies
└── readme.md               # 📖 documentation
```

---

## 📋 requirements

### input files (required)

your `model` folder must contain:

```
your_model_folder/
├── consolidated.00.pth      # model weights (part 0)
├── consolidated.01.pth      # model weights (part 1)
├── ...                      # additional weight files
├── consolidated.xx.pth      # model weights (part x)
├── params.json              # model parameters
├── tokenizer.model          # sentencepiece tokenizer
└── checklist.chk           # optional (ignored during upload)
```

### generated files

the tool automatically creates:

- `config.json` - huggingface model configuration
- `pytorch_model.bin` - merged model weights
- `tokenizer_config.json` - tokenizer configuration
- `special_tokens_map.json` - special tokens mapping
- `tokenizer.json` - fast tokenizer (when possible)
- `readme.md` - model card with usage instructions

---

## 🛠️ command line options

| option | description | required |
|--------|-------------|----------|
| `--model_path` | path to your model folder | ✅ |
| `--repo_name` | huggingface repo (`username/model-name`) | ✅ |
| `--hf_token` | huggingface token (or set `hf_token` env var) | ⚠️ |
| `--private` | make repository private | ❌ |
| `--test` | test model loading after upload | ❌ |
| `--skip_conversion` | skip format conversion | ❌ |

---

## 🔧 advanced usage

### `uv` commands

### environment management

##### create and activate the virtual environment
```bash
uv venv
```

##### linux/macos
```
source .venv/bin/activate  
```

##### or (windows)
```
.venv\scripts\activate
```

#### install project in development mode

```
uv pip install -e .
```

#### install with all optional dependencies
```
uv pip install -e ".[all]"
```

#### update dependencies
```
uv pip install --upgrade -e .
```

#### running the script

##### 0. using the installed command (recommended)
```bash
llama3.2-upload --model_path ./model --repo_name user/model
```

##### using `uv run` (runs in an isolated environment)
```
uv run llama3-2-upload --model_path ./model --repo_name user/model
```

##### using python directly
```
uv run python upload_model.py --model_path ./model --repo_name user/model
```

#### environment variables

##### set your huggingface token
```bash
export hf_token=hf_xxxxxxxxxxxxxxxxxxxx
```

##### run with the environment token
```
llama3-2-upload --model_path ./model --repo_name user/model
```

### programmatic usage

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

### testing only

##### upload without testing
```bash
llama3-2-upload --model_path ./model --repo_name user/model
```

##### test an existing repository
```
uv run python -c "
from model_tester import test_model_loading_full
test_model_loading_full('user/model', 'your_token')
"
```

---

## 🐍 dependencies

this project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### core requirements

all dependencies are defined in the `pyproject.toml` file:

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

### optional dependencies

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

### `sentencepiece` installation issues

if you encounter build errors with `sentencepiece` on macos arm64:

#### option 1: use `conda`
```bash
conda install -c conda-forge sentencepiece
```

install without `sentencepiece` extra
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

#### option 3: skip `sentencepiece` (the script will use the defined fallbacks)

install without `sentencepiece`
```
uv pip install -e ".[accelerate]"   
```

#### option 4: use the pre-built wheel
```
uv pip install sentencepiece --only-binary=sentencepiece
```

---

## 📖 usage examples

### example 1: basic upload

```bash
# simple upload with default settings
llama3-2-upload \
  --model_path ./llama32_finetuned \
  --repo_name myusername/llama32-chatbot
```

### example 2: private model with testing

```bash
# upload private model and test loading
llama3-2-upload \
  --model_path ./sensitive_model \
  --repo_name company/internal-model \
  --private \
  --test \
  --hf_token hf_xxxxxxxxxxxxxxxxxxxx
```

### example 3: skip conversion

```bash
# upload pre-converted model files
llama3-2-upload \
  --model_path ./already_converted_model \
  --repo_name user/converted-model \
  --skip_conversion
```

### example 4: using `uv run`

```bash
# run without installing (uses temporary environment)
uv run --with llama-model-uploader llama3-2-upload \
  --model_path ./model \
  --repo_name user/model
```

---

## 🔍 troubleshooting

### common issues

**❌ "missing required files"**
```bash
# ensure your model folder contains:
ls your_model_folder/
# should show: consolidated.*.pth, params.json, tokenizer.model
```

**❌ "token verification failed"**
```bash
# check your token has write permissions
export hf_token=hf_xxxxxxxxxxxxxxxxxxxx
python -c "from huggingface_hub import hfapi; print(hfapi().whoami())"
```

**❌ "`sentencepiece` build failed"**
```bash
# install via conda instead
conda install -c conda-forge sentencepiece
# or skip sentencepiece (tool will use fallbacks)
```

**❌ "out of memory during conversion"**
```bash
# the tool automatically handles this with fallbacks
# for very large models, it will copy the first checkpoint instead of merging
```

### debug mode

#### run with python's verbose output
```bash
uv run python -v upload_model.py --model_path ./model --repo_name user/model
```

#### check file sizes
```
uv run python -c "
from file_validator import list_model_files, print_file_summary
files = list_model_files('./your_model')
print_file_summary('./your_model', files)
"
```

####  run with uv verbose mode
```
uv --verbose pip install -e .
```

---

## 🧪 testing your upload

after uploading, test your model:

```python
from transformers import autotokenizer, automodelforcausallm

# load your uploaded model
tokenizer = autotokenizer.from_pretrained("your-username/your-model")
model = automodelforcausallm.from_pretrained("your-username/your-model")

# test generation
prompt = "hello, i am a llama model"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=true)
print(response)
```

---

## 📄 license

this project is licensed under the mit license - see the [license](license) file for details.
