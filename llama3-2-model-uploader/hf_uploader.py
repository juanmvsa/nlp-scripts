"""
huggingface hub upload utilities.
"""

from pathlib import Path
from typing import Union, List
from huggingface_hub import HfApi, create_repo, upload_folder
import traceback


def create_model_card(model_name: str, base_model: str = "meta-llama/Llama-3.2-3B", dataset_info: str = "") -> str:
    """create a basic model card for the uploaded model."""
    model_card = f"""---
license: llama3
base_model: {base_model}
tags:
- llama-3.2
- fine-tuned
- causal-lm
language:
- en
pipeline_tag: text-generation
---

# {model_name}

This is a fine-tuned version of {base_model}.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    torch_dtype=torch.float16,
    device_map="auto"
)

# generate text.
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

{dataset_info}

## Model Details

- **Base Model**: {base_model}
- **Model Type**: Causal Language Model
- **Language**: English
"""
    return model_card


def verify_hf_token(hf_token: str) -> bool:
    """verify huggingface token and return user info."""
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami(token=hf_token)
        print(f"✅ authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ token verification failed: {e}")
        return False


def create_hf_repository(repo_name: str, hf_token: str, private: bool = False) -> bool:
    """create or verify huggingface repository."""
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        print(f"✅ repository {repo_name} created/verified at {repo_url}")
        return True
    except Exception as e:
        print(f"❌ error creating repository: {e}")
        print(f"    make sure '{repo_name}' follows the format 'username/model-name'")
        return False


def upload_files_to_hf(model_path: Union[str, Path], repo_name: str, hf_token: str, files_to_upload: List[Path]) -> bool:
    """upload files to huggingface hub."""
    model_path = Path(model_path)
    
    if not files_to_upload:
        print("❌ no files found to upload!")
        return False
    
    try:
        print("⬆️ starting file upload...")
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=hf_token,
            commit_message="upload fine-tuned llama-3.2 model",
            ignore_patterns=[".git*", "__pycache__*", "*.pyc", "consolidated.*.pth", "checklist.chk"],
            run_as_future=False  # ensure synchronous upload.
        )
        print(f"🎉 successfully uploaded model to https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ error uploading files: {e}")
        print(f"    error type: {type(e).__name__}")
        print(f"    full traceback:\n{traceback.format_exc()}")
        return False


def verify_upload(repo_name: str, hf_token: str) -> bool:
    """verify upload by listing files in repository."""
    try:
        api = HfApi(token=hf_token)
        repo_files = api.list_repo_files(repo_id=repo_name, token=hf_token)
        print(f"✅ verified {len(repo_files)} files in repository:")
        for file in sorted(repo_files):
            print(f"   - {file}")
        return True
    except Exception as e:
        print(f"⚠️ could not verify uploaded files: {e}")
        return False


def upload_model_to_hf(model_path: Union[str, Path], repo_name: str, hf_token: str, private: bool = False) -> bool:
    """upload model to hugging face hub."""
    
    print(f"🚀 starting upload to {repo_name}")
    
    # verify token first.
    if not verify_hf_token(hf_token):
        return False
    
    # create repository.
    if not create_hf_repository(repo_name, hf_token, private):
        return False
    
    # create and save model card.
    model_card_content = create_model_card(repo_name)
    model_card_path = Path(model_path) / "README.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card_content)
    print("✅ created readme.md")
    
    # import here to avoid circular imports.
    from file_validator import list_model_files, print_file_summary
    
    # list files that will be uploaded.
    files_to_upload = list_model_files(model_path)
    print_file_summary(model_path, files_to_upload)
    
    # upload all files.
    if not upload_files_to_hf(model_path, repo_name, hf_token, files_to_upload):
        return False
    
    # verify upload by listing files in repo.
    verify_upload(repo_name, hf_token)
    
    return True
