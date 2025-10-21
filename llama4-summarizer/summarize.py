#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.40.0",
#     "torch>=2.0.0",
#     "accelerate>=0.27.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=3.20.0",
# ]
# ///
"""
document summarization script using meta-llama/llama-4-maverick-17b-128e-instruct.
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(hf_token: str):
    """
    load the llama-4 model and tokenizer from huggingface.

    args:
        hf_token: huggingface authentication token for gated model access.

    returns:
        tuple of (tokenizer, model).
    """
    model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

    print(f"loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("model loaded successfully.")
    return tokenizer, model


def read_document(file_path: str) -> str:
    """
    read the input document.

    args:
        file_path: path to the input document.

    returns:
        document content as string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"error: input file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"error reading file '{file_path}': {e}")
        sys.exit(1)


def generate_summary(tokenizer, model, document: str) -> str:
    """
    generate a summary of the document using the llama-4 model.

    args:
        tokenizer: the model tokenizer.
        model: the llama-4 model.
        document: the document content to summarize.

    returns:
        generated summary as string.
    """
    # create the prompt for summarization.
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

you are a helpful assistant that creates concise and accurate summaries of documents.<|eot_id|><|start_header_id|>user<|end_header_id|>

please provide a comprehensive summary of the following document:

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print("generating summary...")

    # tokenize the input.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # generate the summary.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # decode the output.
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # extract only the assistant's response.
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        summary = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        summary = summary.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
    else:
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = summary[len(prompt):].strip()

    print("summary generated successfully.")
    return summary


def write_summary(output_path: str, summary: str):
    """
    write the summary to the output file.

    args:
        output_path: path to the output file.
        summary: the generated summary.
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"summary saved to '{output_path}'.")
    except Exception as e:
        print(f"error writing to file '{output_path}': {e}")
        sys.exit(1)


def main():
    """
    main function to parse arguments and run the summarization pipeline.
    """
    parser = argparse.ArgumentParser(
        description="summarize documents using meta-llama/llama-4-maverick-17b-128e-instruct."
    )

    parser.add_argument(
        "hf_token",
        type=str,
        help="huggingface authentication token for gated model access."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="path to the document to summarize."
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="path and name of the output txt file."
    )

    args = parser.parse_args()

    # load the model and tokenizer.
    tokenizer, model = load_model(args.hf_token)

    # read the input document.
    document = read_document(args.input_file)

    # generate the summary.
    summary = generate_summary(tokenizer, model, document)

    # write the summary to the output file.
    write_summary(args.output_file, summary)

    print("\nsummarization complete!")


if __name__ == "__main__":
    main()
