import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def build_prompt(example: dict) -> str:
    """Format each record into a concise instruction-response pair."""
    return (
        "### Detected Emotion & User Context\n"
        f"{example['input']}\n\n"
        "### Supportive Reply\n"
        f"{example['output']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for EmoCare.")
    parser.add_argument(
        "--dataset_path",
        default="assets/comfort_examples.jsonl",
        help="Path to the JSONL dataset.",
    )
    parser.add_argument(
        "--model_name",
        default="nreHieW/Llama-3.1-8B-Instruct",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/lora",
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Maximum sequence length for training samples.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps to simulate a larger batch.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    dataset = dataset.map(
        lambda example: {"text": build_prompt(example)}, remove_columns=dataset.column_names
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    fp16_enabled = cuda_available and not bf16_supported

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.04,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16_supported,
        fp16=fp16_enabled,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        padding_free=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()

