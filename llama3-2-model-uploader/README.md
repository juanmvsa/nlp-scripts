# 🦙 llama-3.2 Model Uploader

A comprehensive Python toolkit for converting and uploading fine-tuned Llama-3.2 models to Hugging Face Hub. This tool automatically converts from the original Llama checkpoint format to HuggingFace-compatible format with full tokenizer support.

## ✨ Features

- 🔄 **Automatic Format Conversion**: Converts Llama checkpoint format to HuggingFace format
- 🎯 **Smart Tokenizer Handling**: Works with or without SentencePiece (this library has issues with ARM-based Apple systems), includes fallback mechanisms
- 📦 **Modular Architecture**: Clean, reusable components for different tasks
- 🧠 **Memory Efficient**: Handles large models with intelligent memory management
- 🔒 **Secure Upload**: Supports private repositories and token authentication
- 🧪 **Built-in Testing**: Validates model loading after upload
- 📊 **Progress Tracking**: Clear status messages and file size reporting
- 🛡️ **Error Resilience**: Comprehensive error handling and fallback strategies

## 🚀 Quick Start

### Installation

1. Clone this repository
```bash
git clone <repository-url>
```
  
```
cd llama-model-uploader
```

2. Start a [uv](https://docs.astral.sh/uv/guides/projects/) project
```
uv init    
```
:w




2. Install the necessary dependencies
```
# install dependencies
pip install -r requirements.txt
```

```
# set your huggingface token
export HF_TOKEN=your_huggingface_token_here
```

### Basic Usage

```bash
python upload_model.py \
  --model_path ./your_llama32_model \
  --repo_name your-username/your-model-name
```

### Full Example

```bash
python upload_model.py \
  --model_path ./my_finetuned_llama \
  --repo_name johndoe/llama32-finance-model \
  --private \
  --test
```

## 📁 Project Structure

```
llama-model-uploader/
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

Your model folder must contain:

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

### Environment Variables

```bash
# set huggingface token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# run with environment token
python upload_model.py --model_path ./model --repo_name user/model
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

```bash
# upload without testing
python upload_model.py --model_path ./model --repo_name user/model

# test existing repository
python -c "
from model_tester import test_model_loading_full
test_model_loading_full('user/model', 'your_token')
"
```

## 🐍 Dependencies

### Core Requirements

```txt
huggingface_hub>=0.19.0    # hub integration
transformers>=4.43.0       # model compatibility
torch>=2.0.0              # pytorch backend
tokenizers>=0.15.0        # tokenizer handling
```

### Optional Dependencies

```txt
sentencepiece>=0.1.97     # tokenizer conversion (recommended)
accelerate>=0.24.0        # faster model loading
safetensors>=0.4.0        # safer serialization
```

### SentencePiece Installation Issues

If you encounter build errors with SentencePiece on macOS ARM64:

```bash
# option 1: use conda
conda install -c conda-forge sentencepiece

# option 2: use homebrew + pip
brew install protobuf
pip install sentencepiece

# option 3: skip sentencepiece (tool will use fallbacks)
pip install huggingface_hub transformers torch accelerate
```

## 📖 Usage Examples

### Example 1: Basic Upload

```bash
# simple upload with default settings
python upload_model.py \
  --model_path ./llama32_finetuned \
  --repo_name myusername/llama32-chatbot
```

### Example 2: Private Model with Testing

```bash
# upload private model and test loading
python upload_model.py \
  --model_path ./sensitive_model \
  --repo_name company/internal-model \
  --private \
  --test \
  --hf_token hf_xxxxxxxxxxxxxxxxxxxx
```

### Example 3: Skip Conversion

```bash
# upload pre-converted model files
python upload_model.py \
  --model_path ./already_converted_model \
  --repo_name user/converted-model \
  --skip_conversion
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

**❌ "SentencePiece build failed"**
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
python -v upload_model.py --model_path ./model --repo_name user/model

# check file sizes
python -c "
from file_validator import list_model_files, print_file_summary
files = list_model_files('./your_model')
print_file_summary('./your_model', files)
"
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

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a pull request

### Development Setup

```bash
# clone your fork
git clone https://github.com/yourusername/llama-model-uploader.git
cd llama-model-uploader

# install development dependencies
pip install -r requirements.txt
pip install black isort mypy  # formatting and type checking

# run type checking
mypy *.py

# format code
black *.py
isort *.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama-3.2 model architecture
- **Hugging Face** for the transformers library and model hub
- **The open-source community** for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llama-model-uploader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llama-model-uploader/discussions)
- **Documentation**: This README and inline code comments

---

**⭐ If this tool helped you, please consider giving it a star on GitHub!**
