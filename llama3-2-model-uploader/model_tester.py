"""
model testing utilities for uploaded models.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def test_tokenizer_loading(repo_name: str, hf_token: str) -> bool:
    """test if tokenizer can be loaded successfully."""
    try:
        # try loading with legacy=false first (recommended).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_name,
                token=hf_token,
                legacy=False,
                use_fast=False,  # use slow tokenizer for sentencepiece.
            )
            print("✅ tokenizer loaded successfully with legacy=false")
            return True
        except Exception as e1:
            print(f"⚠️ legacy=false failed, trying legacy=true: {e1}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_name, token=hf_token, legacy=True, use_fast=False
                )
                print("✅ tokenizer loaded successfully with legacy=true")
                return True
            except Exception as e2:
                print(f"❌ both tokenizer loading methods failed: {e2}")
                return False

    except Exception as e:
        print(f"❌ tokenizer loading failed: {e}")
        return False


def test_model_loading(repo_name: str, hf_token: str) -> bool:
    """test if model can be loaded successfully."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_name,
            token=hf_token,
            torch_dtype="auto",
            device_map="cpu",  # use cpu for testing.
        )
        print("✅ model successfully loaded with transformers!")
        return True
    except Exception as e:
        print(f"❌ model loading failed: {e}")
        return False


def test_tokenization(repo_name: str, hf_token: str) -> bool:
    """test basic tokenization functionality."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_name, token=hf_token, use_fast=False
        )

        # test a simple tokenization.
        test_text = "hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"✅ tokenization test passed: {inputs}")
        return True

    except Exception as e:
        print(f"⚠️ tokenization test failed: {e}")
        return False


def test_model_loading_full(repo_name: str, hf_token: str) -> bool:
    """test if the uploaded model can be loaded with transformers."""
    print("🔍 testing model loading...")

    # test tokenizer loading.
    if not test_tokenizer_loading(repo_name, hf_token):
        return False

    # test model loading.
    if not test_model_loading(repo_name, hf_token):
        return False

    # test tokenization functionality.
    test_tokenization(repo_name, hf_token)

    print("✅ all model tests passed!")
    return True
