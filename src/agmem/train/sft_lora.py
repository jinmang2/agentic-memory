"""QLoRA SFT for the 0.5B extraction student (Phase 4, docs/06).

Targets RTX 2060 6GB: 4-bit base + LoRA r=16, batch 1, grad-accum 8.
Requires the 'train' extra: uv sync --extra train
(The llama.cpp daemon must be stopped during training — both need VRAM.)

    uv run python -m agmem.train.sft_lora --data data/sft/note.jsonl \
        --model Qwen/Qwen3-0.6B --out checkpoints/note-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl from distill_data.py")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, Trainer, TrainingArguments)
    except ImportError as exc:  # capability principle: explicit, actionable
        raise SystemExit(
            f"missing training deps ({exc.name}) — install with: uv sync --extra train"
        ) from exc

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_text(r: dict) -> dict:
        messages = [
            {"role": "system",
             "content": "You must respond with a single JSON object and nothing else."},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]
        return {"text": tok.apply_chat_template(messages, tokenize=False)}

    ds = Dataset.from_list([to_text(r) for r in rows])

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                               bnb_4bit_compute_dtype=torch.float16),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model, train_dataset=ds,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=args.lr, fp16=True, logging_steps=10,
            save_strategy="epoch", report_to=[]),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
