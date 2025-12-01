from typing import Dict, Any, List
from dataclasses import dataclass
import os

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.data.flickr8k import load_flickr8k
from src.models.qwen_vl_loader import Qwen3VLLoader
from src.models.lora_setup import apply_lora

USE_MOCK = os.environ.get("USE_MOCK", "True").lower() == "true"


@dataclass
class VLMDataCollator:
    """
    Data collator for vision-language captioning with Qwen3-VL chat format.

    For each example:
      - User: [image] + prompt_template (e.g. "Describe this image...")
      - Assistant: ground-truth caption (caption_0)

    We:
      - Build chat messages (user + assistant)
      - Use apply_chat_template to get the text with image placeholders
      - Encode via processor(text, images)
      - Compute labels so that ONLY the caption tokens are supervised.
    """

    processor: Any
    prompt_template: str

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]
        captions = [ex["caption_0"] for ex in batch]  # training target
        user_texts = [self.prompt_template.strip()] * len(batch)

        text_full_list: List[str] = []

        # 1) Build full chat messages (user + assistant) and text
        for user_text, caption in zip(user_texts, captions):
            messages_user = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            messages_full = messages_user + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": caption},
                    ],
                }
            ]

            text_full = self.processor.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
            text_full_list.append(text_full)

        # 2) Encode full conversations + images
        enc = self.processor(
            text=text_full_list,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = enc["input_ids"]           # [B, T]
        attention_mask = enc["attention_mask"] # [B, T]
        tokenizer = self.processor.tokenizer

        # 3) Initialize labels = -100 everywhere
        labels = torch.full_like(input_ids, -100)

        # Helper: find subsequence `subseq` inside `seq` (both are lists of ints)
        def find_subsequence(seq: List[int], subseq: List[int]):
            n, m = len(seq), len(subseq)
            if m == 0 or m > n:
                return None
            for start in range(n - m + 1):
                if seq[start:start + m] == subseq:
                    return start
            return None

        # 4) For each example, locate caption tokens in the full sequence
        for i, caption in enumerate(captions):
            # Only look at non-padded part
            valid_len = attention_mask[i].sum().item()
            full_ids = input_ids[i, :valid_len].tolist()

            # Tokenize caption as a plain sequence (no extra specials)
            tok = tokenizer(
                caption,
                add_special_tokens=False,
                return_tensors=None,
            )
            raw_ids = tok["input_ids"]

            # Handle both shapes: List[int] or List[List[int]]
            if isinstance(raw_ids[0], int):
                cap_ids = raw_ids  # already a flat list of ints
            else:
                cap_ids = raw_ids[0]  # first sequence in batch

            # Find where caption appears in the full sequence
            start = find_subsequence(full_ids, cap_ids)
            if start is None:
                # If we can't find the caption tokens, skip supervision for this example
                # (you could log here if you want)
                continue

            end = start + len(cap_ids)

            # Enable supervision only on the caption token span
            labels[i, start:end] = input_ids[i, start:end]

        enc["labels"] = labels
        enc["attention_mask"] = attention_mask

        return enc


def run_lora_training(cfg: Dict[str, Any]):
    """
    Full LoRA fine-tuning loop for Qwen3-VL on Flickr8k captioning.

    This should be run on a GPU machine (NOT WSL CPU).
    """

    print("[train] Loading Flickr8k dataset...")
    ds = load_flickr8k(
        max_train_samples=cfg["dataset"].get("max_train_samples"),
        max_val_samples=cfg["dataset"].get("max_val_samples"),
        max_test_samples=cfg["dataset"].get("max_test_samples"),
    )
    train_ds: Dataset = ds["train"]
    val_ds: Dataset = ds["validation"]
    print(f"[train] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Load real Qwen3-VL model + processor
    print("[train] Loading Qwen3-VL model...")
    loader = Qwen3VLLoader(cfg)
    model = loader.model
    processor = loader.processor

    # Apply LoRA
    model = apply_lora(model, cfg["peft"])

    # Build data collator
    prompt_template = cfg["generation"]["prompt_template"]
    data_collator = VLMDataCollator(
        processor=processor,
        prompt_template=prompt_template,
    )

    # TrainingArguments from config
    train_cfg = cfg["training"]

    training_args = TrainingArguments(
    output_dir=train_cfg["output_dir"],
    num_train_epochs=int(train_cfg["num_train_epochs"]),
    per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
    per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
    gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
    learning_rate=float(train_cfg["learning_rate"]),
    weight_decay=float(train_cfg["weight_decay"]),
    max_grad_norm=float(train_cfg["max_grad_norm"]),
    warmup_ratio=float(train_cfg["warmup_ratio"]),
    logging_steps=int(train_cfg["logging_steps"]),
    eval_strategy="steps",
    eval_steps=int(train_cfg["eval_steps"]),
    save_strategy="steps",
    save_steps=int(train_cfg["save_steps"]),
    save_total_limit=int(train_cfg["save_total_limit"]),
    fp16=bool(train_cfg.get("fp16", False)),
    bf16=bool(train_cfg.get("bf16", False)),
    remove_unused_columns=False,
    report_to=["wandb"] if cfg.get("wandb", {}).get("enabled", False) and not USE_MOCK else [],
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("[train] Starting training...")
    trainer.train(
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or False
    )

    print("[train] Training complete. Saving final model...")
    trainer.save_model(train_cfg["output_dir"])
    print(f"[train] Model saved to {train_cfg['output_dir']}")
