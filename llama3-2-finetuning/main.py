#!/usr/bin/env python3

from typing import Optional
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import json


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training"""
    dist.destroy_process_group()


class DistributedLlama70BTrainer:
    """Distributed trainer for Llama 3.2 70B"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-70B",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

    def setup_model_and_tokenizer(self) -> tuple:
        """Setup model and tokenizer for distributed training"""

        # Load tokenizer (only on rank 0 to avoid conflicts)
        if self.rank == 0:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = None

        # Broadcast tokenizer to all processes
        if self.world_size > 1:
            dist.barrier()
            if self.rank != 0:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        # Model setup with device mapping for multi-GPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  # We'll handle device placement manually
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )

        # Move model to appropriate device
        model = model.to(self.device)

        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Wrap with DDP for distributed training
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank])

        if self.rank == 0:
            model.print_trainable_parameters()

        return model, tokenizer

    def prepare_dataset(
        self, data_path: str, tokenizer, max_length: int = 1024
    ) -> Dataset:
        """Prepare dataset for distributed training"""

        # Load data
        with open(data_path, "r") as f:
            if data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        # Convert to proper format
        formatted_data = []
        for item in data:
            if "instruction" in item and "output" in item:
                text = f"<|user|>\n{item['instruction']}\n<|assistant|>\n{item['output']}<|end_of_text|>"
                formatted_data.append({"text": text})
            elif "text" in item:
                formatted_data.append(item)

        dataset = Dataset.from_list(formatted_data)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding=False, max_length=max_length
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, num_proc=4 if self.rank == 0 else 1
        )

        return tokenized_dataset

    def train(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        output_dir: str = "./llama-70b-distributed",
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """Run distributed training"""

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments for distributed setup
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Per GPU batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32 // self.world_size,  # Adjust for world size
            warmup_steps=50,
            learning_rate=1e-4,
            weight_decay=0.01,
            bf16=True,
            logging_steps=5,
            save_steps=250,
            eval_steps=250 if eval_dataset else None,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            group_by_length=True,
            optim="adamw_bnb_8bit",
            dataloader_num_workers=2,
            save_strategy="steps",
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=False,  # Problematic with DDP
            report_to="wandb" if self.rank == 0 else None,
            run_name=f"llama-70b-distributed-{self.world_size}gpu",
            ddp_find_unused_parameters=False,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            save_only_model=True,  # Important for LoRA + DDP
            # Distributed training specific
            local_rank=self.rank,
            ddp_backend="nccl",
            dataloader_drop_last=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Clear cache before training
        torch.cuda.empty_cache()

        # Start training
        if self.rank == 0:
            print("Starting distributed training...")

        trainer.train()

        # Save model (only on rank 0)
        if self.rank == 0:
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")


def main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Main distributed training function"""

    # Setup distributed training
    setup_distributed(rank, world_size)

    try:
        # Initialize trainer
        trainer = DistributedLlama70BTrainer(
            model_name=args.model_name, rank=rank, world_size=world_size
        )

        # Setup model and tokenizer
        model, tokenizer = trainer.setup_model_and_tokenizer()

        # Prepare dataset
        train_dataset = trainer.prepare_dataset(
            args.data_path, tokenizer, args.max_length
        )

        # Optional eval dataset
        eval_dataset = None
        if args.eval_data_path:
            eval_dataset = trainer.prepare_dataset(
                args.eval_data_path, tokenizer, args.max_length
            )

        # Train
        trainer.train(model, tokenizer, train_dataset, args.output_dir, eval_dataset)

    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed Llama 3.2 70B Fine-tuning"
    )
    parser.add_argument(
        "--model_name", default="meta-llama/Llama-3.2-70B", help="Model name"
    )
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--eval_data_path", help="Path to evaluation data")
    parser.add_argument(
        "--output_dir", default="./llama-70b-distributed", help="Output directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs",
    )

    args = parser.parse_args()

    # Launch distributed training
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            main, args=(args.world_size, args), nprocs=args.world_size, join=True
        )
    else:
        main(0, 1, args)

# Example launch commands:

# Single GPU:
# python distributed_training.py --data_path dataset.jsonl --world_size 1

# Multi-GPU (4 GPUs):
# python distributed_training.py --data_path dataset.jsonl --world_size 4

# With evaluation:
# python distributed_training.py --data_path train.jsonl --eval_data_path eval.jsonl --world_size 4

# Using torchrun (alternative):
# torchrun --nproc_per_node=4 distributed_training.py --data_path dataset.jsonl
