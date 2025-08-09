"""Minimal example of training a custom model with DeepSpeed.

Only the dataset path and model path need to be provided.  The script loads a
causal language model, tokenizes the dataset and runs a simple training loop
using DeepSpeed for distributed execution.
"""

import argparse
from typing import Dict

import deepspeed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom model")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path or name of the model to fine-tune")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to a text dataset file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Where to store the trained model")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    dataset = load_dataset("text", data_files={"train": args.dataset_path})["train"]

    def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(batch["text"], truncation=True, padding="max_length",
                         max_length=tokenizer.model_max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    engine, _, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )

    train_loader = DataLoader(
        dataset, batch_size=engine.config.train_micro_batch_size_per_gpu, shuffle=True
    )

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(engine.device) for k, v in batch.items()}
        outputs = engine(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        engine.backward(loss)
        engine.step()
        if step % 10 == 0 and engine.global_rank == 0:
            print(f"step {step} loss {loss.item():.4f}")

    if engine.global_rank == 0:
        engine.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
