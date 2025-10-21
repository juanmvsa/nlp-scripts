#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.40.0",
#     "torch>=2.0.0",
#     "accelerate>=0.27.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=3.20.0",
#     "spacy-layout>=0.0.12",
#     "spacy>=3.7.0",
#     "bitsandbytes>=0.41.0",
# ]
# ///
"""
document summarization script using meta-llama/llama-4-maverick-17b-128e-instruct.
"""

import argparse
import sys
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import spacy
from spacy_layout import spaCyLayout


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
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu device: {torch.cuda.get_device_name(0)}")
        print(f"gpu memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} gb")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    # configure 4-bit quantization (nf4) for maximum memory efficiency.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "70GiB", "cpu": "100GiB"}
    )

    print(f"model loaded with 4-bit nf4 quantization (~75% memory reduction).")

    # check where model parameters are loaded.
    print("\nmodel loaded successfully.")
    print("\nmodel device allocation:")
    device_map = model.hf_device_map if hasattr(model, 'hf_device_map') else {}
    if device_map:
        devices = set(device_map.values())
        # separate devices by type for proper handling.
        gpu_devices = []
        cpu_devices = []
        disk_devices = []

        for device in devices:
            if isinstance(device, int) or (isinstance(device, str) and device.startswith('cuda')):
                gpu_devices.append(device)
            elif device == 'cpu':
                cpu_devices.append(device)
            elif device == 'disk':
                disk_devices.append(device)

        # display in priority order: gpu, cpu, disk.
        for device in gpu_devices:
            device_id = device if isinstance(device, int) else device.split(':')[1]
            print(f"  ✓ gpu (cuda:{device_id})")
        for device in cpu_devices:
            print(f"  ⚠ cpu (slower performance expected)")
        for device in disk_devices:
            print(f"  ⚠ disk (significantly slower performance expected)")
    else:
        # fallback: check model device.
        model_device = next(model.parameters()).device
        if model_device.type == 'cuda':
            print(f"  ✓ gpu ({model_device})")
        else:
            print(f"  ⚠ {model_device.type}")

    return tokenizer, model


def read_document(file_path: str) -> str:
    """
    read the input document (txt, md, or pdf).

    note: for pdf files, uses spacy-layout to extract text.

    args:
        file_path: path to the input document.

    returns:
        document content as string.
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        print(f"error: input file '{file_path}' not found.")
        return ""

    try:
        # for pdf files, use spacy-layout to extract text.
        if file_path_obj.suffix.lower() == '.pdf':
            print(f"  extracting text from pdf using spacy-layout...")
            nlp = spacy.blank("en")
            layout = spaCyLayout(nlp)
            doc = layout(file_path)
            content = doc.text
            print(f"  extracted {len(content)} characters from pdf.")
            return content
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
    except Exception as e:
        print(f"error reading file '{file_path}': {e}")
        return ""


def collect_files(input_path: str) -> List[Path]:
    """
    collect all supported files from input path (file or directory).

    args:
        input_path: path to file or directory.

    returns:
        list of path objects for supported files.
    """
    supported_extensions = {'.txt', '.md', '.pdf'}
    path = Path(input_path)

    if not path.exists():
        print(f"error: input path '{input_path}' does not exist.")
        sys.exit(1)

    files = []

    if path.is_file():
        if path.suffix.lower() in supported_extensions:
            files.append(path)
        else:
            print(f"error: file '{path}' has unsupported extension. supported: {', '.join(supported_extensions)}")
            sys.exit(1)
    elif path.is_dir():
        # recursively find all supported files.
        for ext in supported_extensions:
            files.extend(path.rglob(f"*{ext}"))

        if not files:
            print(f"error: no supported files found in directory '{input_path}'.")
            sys.exit(1)

    return sorted(files)


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

    # tokenize the input with reduced max length to save memory.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # check inference device.
    inference_device = inputs['input_ids'].device
    if inference_device.type == 'cuda':
        print(f"  ✓ running inference on gpu ({inference_device})")
    else:
        print(f"  ⚠ running inference on {inference_device.type} (slower performance expected)")

    # generate the summary.
    with torch.no_grad():
        # clear cuda cache before generation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
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


def generate_output_path(input_file: Path, output_path: str, base_input_path: Path) -> str:
    """
    generate output file path for a given input file.

    args:
        input_file: path to the input file.
        output_path: user-specified output path.
        base_input_path: base input path (file or directory).

    returns:
        output file path as string.
    """
    output = Path(output_path)

    # if output is a directory or input is a directory, create structured output.
    if base_input_path.is_dir() or output.is_dir() or output_path.endswith('/'):
        # create relative path structure.
        relative_path = input_file.relative_to(base_input_path.parent if base_input_path.is_file() else base_input_path)

        # generate new filename: originalname_llama4_maverick_summary.txt.
        original_stem = input_file.stem
        new_filename = f"{original_stem}_llama4_maverick_summary.txt"

        # maintain directory structure but use new filename.
        output_file = output / relative_path.parent / new_filename
        return str(output_file)
    else:
        # single file output.
        return output_path


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
        "input",
        type=str,
        help="path to the document or folder to summarize (supports txt, md, pdf files)."
    )

    parser.add_argument(
        "output",
        type=str,
        help="path and name of the output txt file or directory for multiple summaries."
    )

    args = parser.parse_args()

    # collect all files to process.
    input_path = Path(args.input)
    files_to_process = collect_files(args.input)

    print(f"found {len(files_to_process)} file(s) to process.\n")

    # load the model and tokenizer.
    tokenizer, model = load_model(args.hf_token)

    # process each file.
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] processing: {file_path}")

        # read the document.
        document = read_document(str(file_path))

        if not document.strip():
            print(f"warning: skipping empty or unreadable file: {file_path}")
            continue

        # generate the summary.
        summary = generate_summary(tokenizer, model, document)

        # determine output path.
        output_path = generate_output_path(file_path, args.output, input_path)

        # write the summary.
        write_summary(output_path, summary)

    print("\n" + "="*50)
    print(f"summarization complete! processed {len(files_to_process)} file(s).")
    print("="*50)


if __name__ == "__main__":
    main()
