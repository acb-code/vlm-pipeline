from typing import Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.data.flickr8k import load_flickr8k
from src.models.qwen_vl_loader import Qwen3VLLoader
from src.models.lora_setup import apply_lora


@dataclass
class VLMDataCollator:
    """
    Data collator for vision-language captioning.

    - Takes raw HF samples with:
        image, caption_0..caption_4
    - Builds full text = prompt_template + "\n" + chosen caption
    - Uses processor to create model inputs
    - Masks out prompt tokens in labels so loss is only on caption.
    """

    processor: Any
    prompt_template: str

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]
        # For now, just use caption_0 as the target caption
        captions = [ex["caption_0"] for ex in batch]

        prompts = [self.prompt_template.strip()] * len(batch)
        full_texts = [p + "\n" + c for p, c in zip(prompts, captions)]

        # Processor will handle text + images jointly
        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Compute prompt lengths in tokens, using same tokenizer
        tokenizer = self.processor.tokenizer
        prompt_tokenized = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_lens = prompt_tokenized["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)

        # Labels: copy input_ids then mask out prompt tokens
        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lens):
            # Mask prompt part
            labels[i, : prompt_len] = -100

        inputs["labels"] = labels
        inputs["attention_mask"] = attention_mask

        return inputs


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
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        report_to=[],  # add "wandb" if you want later
        remove_unused_columns=False,  # crucial for vision inputs
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
