#!/usr/bin/env python3
"""Fine-tune Decimus LLM using QLoRA on orchestration training data.

Uses standard PEFT + TRL (no Unsloth dependency).

Usage (on a rented GPU with A100/A6000):
    pip install peft trl datasets bitsandbytes accelerate
    python finetune.py \
        --data train.jsonl \
        --eval eval.jsonl \
        --output /workspace/decimus-llm-v1 \
        --merge

Requirements:
    - GPU with >= 24GB VRAM (A100 40GB recommended)
    - transformers, peft, trl, datasets, bitsandbytes, accelerate, torch
"""

import argparse
from pathlib import Path


SYSTEM_PROMPT = (
    "You are Decimus LLM, an expert orchestration advisor trained on "
    "Rimsky-Korsakov's Principles of Orchestration. You help composers "
    "transform piano sketches into full orchestral scores by recommending "
    "instrument assignments, doublings, voicings, and textures. "
    "When asked for an orchestration plan, respond with structured JSON."
)


def format_chat_template(example, tokenizer):
    """Format an Alpaca-style record into Llama 3.1 chat format."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Decimus LLM")
    parser.add_argument("--data", type=Path, required=True,
                        help="Training data JSONL (Alpaca format)")
    parser.add_argument("--eval", type=Path, default=None,
                        help="Eval data JSONL (optional)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for model/adapter")
    parser.add_argument("--base-model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model from HuggingFace")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base model and save full fp16 weights")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in 4-bit
    print(f"Loading model in 4-bit: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    print(f"Applying LoRA (rank={args.lora_rank})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading training data: {args.data}")
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    print(f"Training samples: {len(dataset)}")

    eval_dataset = None
    if args.eval and args.eval.exists():
        eval_dataset = load_dataset("json", data_files=str(args.eval), split="train")
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=eval_dataset.column_names,
        )
        print(f"Eval samples: {len(eval_dataset)}")

    # Training arguments
    output_dir = str(args.output)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
    )

    trainer.train()

    # Save LoRA adapter
    lora_path = Path(output_dir) / "lora-adapter"
    print(f"Saving LoRA adapter -> {lora_path}")
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))

    # Optionally merge and save full model
    if args.merge:
        merged_path = Path(output_dir) / "merged-fp16"
        print(f"Merging LoRA into base model -> {merged_path}")

        from peft import AutoPeftModelForCausalLM

        # Reload the adapter and merge
        merge_model = AutoPeftModelForCausalLM.from_pretrained(
            str(lora_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        merge_model = merge_model.merge_and_unload()
        merge_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))

    print()
    print("=" * 50)
    print("Training complete!")
    print(f"  LoRA adapter: {lora_path}")
    if args.merge:
        print(f"  Merged model: {merged_path}")
    print()
    print("Next steps:")
    print("  1. Download the merged model to your local machine")
    print("  2. Build Docker image: cd deploy && docker build -t decimus-llm .")
    print("  3. Push to Docker Hub: docker push YOUR_USER/decimus-llm")
    print("  4. Create RunPod serverless endpoint pointing to the image")
    print("=" * 50)


if __name__ == "__main__":
    main()
