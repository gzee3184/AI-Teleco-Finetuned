"""
LoRA Training Script for 5G Fault Diagnosis

Fine-tunes Qwen 2.5 1.5B using LoRA to correct hybrid classifier errors.
Focuses on improving C1 detection without overfitting.

Anti-overfitting measures:
1. Early stopping on validation loss
2. Low LoRA rank (r=8)
3. Dropout (0.1)
4. Weight decay (0.01)
5. Focused dataset (error cases only)
"""

import os
os.environ['HF_HOME'] = '/export/scratch/abrar008/.cache/huggingface'

import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType


def load_data(data_path):
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def format_for_training(example, tokenizer):
    """Format a single example for training."""
    
    # Build the full conversation
    prompt = example['Input_Prompt']
    reasoning = example['Reasoning_Trace']
    label = example['Generated_Label']
    
    # Format as chat
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reasoning}  # Use raw reasoning trace as-is
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return {"text": text}


def create_dataset(data, tokenizer):
    """Create HuggingFace dataset from training data."""
    formatted = [format_for_training(ex, tokenizer) for ex in data]
    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--train_data", default="lora_data/train.json")
    parser.add_argument("--val_data", default="lora_data/val.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="lora_output")
    
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training config
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    
    args = parser.parse_args()
    
    print("="*60)
    print("LoRA FINE-TUNING FOR 5G FAULT DIAGNOSIS")
    print("="*60)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    print(f"\nConfiguring LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare data
    print(f"\nLoading training data from {args.train_data}")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    train_dataset = create_dataset(train_data, tokenizer)
    val_dataset = create_dataset(val_data, tokenizer)
    
    # Tokenize
    print("\nTokenizing datasets...")
    
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Training arguments
    print(f"\nSetting up training (epochs={args.num_epochs}, lr={args.learning_rate})")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        save_total_limit=2
    )
    
    # Early stopping callback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.01
        )
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks
    )
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        trainer.train()
        
        # Save final model
        print(f"\nSaving model to {args.output_dir}/final")
        model.save_pretrained(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        
        print("\nTraining complete!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save_pretrained(f"{args.output_dir}/interrupted")
        tokenizer.save_pretrained(f"{args.output_dir}/interrupted")
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    eval_results = trainer.evaluate()
    print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
    
    return eval_results


if __name__ == "__main__":
    main()
