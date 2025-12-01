import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset
import yaml

# ----------------------------------------------------------------------
# 0. Make sure the project root is on sys.path so `src.*` imports work.
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # /root/vlm-pipeline
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.flickr8k import load_flickr8k
from src.models.qwen_vl_loader import Qwen3VLLoader
from src.training.trainer import VLMDataCollator  # updated collator


def load_config_from_yaml(name: str) -> Dict[str, Any]:
    """
    Load a YAML config from the `configs/` directory at project root.
    Example: load_config_from_yaml("flickr8k_captioning.yaml")
    """
    cfg_path = ROOT / "configs" / name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ------------------------------------------------------------------
    # 1. Load config, dataset, and processor
    # ------------------------------------------------------------------
    cfg = load_config_from_yaml("flickr8k_captioning.yaml")

    print("[debug] Loading Flickr8k dataset...")
    ds = load_flickr8k(
        max_train_samples=cfg["dataset"].get("max_train_samples", 8) or 8,
        max_val_samples=cfg["dataset"].get("max_val_samples", 8) or 8,
        max_test_samples=cfg["dataset"].get("max_test_samples", 8) or 8,
    )
    train_ds: Dataset = ds["train"]
    print(f"[debug] Train samples: {len(train_ds)}")

    # ------------------------------------------------------------------
    # 2. Load processor via Qwen3VLLoader (we only actually need processor)
    # ------------------------------------------------------------------
    print("[debug] Initializing Qwen3VLLoader (processor only)...")
    loader = Qwen3VLLoader(cfg)
    processor = loader.processor
    tokenizer = processor.tokenizer

    prompt_template = cfg["generation"]["prompt_template"]
    collator = VLMDataCollator(
        processor=processor,
        prompt_template=prompt_template,
    )

    # ------------------------------------------------------------------
    # 3. Take a small batch from train_ds and collate it
    # ------------------------------------------------------------------
    batch_size = 2
    raw_batch = [train_ds[i] for i in range(batch_size)]
    print(f"[debug] Building batch of size {batch_size}...")

    batch = collator(raw_batch)

    input_ids: torch.Tensor = batch["input_ids"]
    attention_mask: torch.Tensor = batch["attention_mask"]
    labels: torch.Tensor = batch["labels"]
    pixel_values = batch.get("pixel_values", None)

    print("[debug] Batch shapes:")
    print(f"  input_ids:      {tuple(input_ids.shape)}")
    print(f"  attention_mask: {tuple(attention_mask.shape)}")
    if pixel_values is not None:
        print(f"  pixel_values:   {tuple(pixel_values.shape)}")
    print(f"  labels:         {tuple(labels.shape)}")

    # ------------------------------------------------------------------
    # 4. Decode and visualize per example
    # ------------------------------------------------------------------
    os.makedirs("outputs/debug", exist_ok=True)
    debug_path = "outputs/debug/flickr8k_batch_inspect.txt"

    with open(debug_path, "w", encoding="utf-8") as f:
        for i in range(batch_size):
            f.write(f"=== EXAMPLE {i} ===\n")

            ex: Dict[str, Any] = train_ds[i]
            gt_captions = [
                ex["caption_0"],
                ex["caption_1"],
                ex["caption_2"],
                ex["caption_3"],
                ex["caption_4"],
            ]

            f.write("Ground-truth captions:\n")
            for idx, c in enumerate(gt_captions):
                f.write(f"  caption_{idx}: {c}\n")

            ids = input_ids[i]
            attn = attention_mask[i]
            lbls = labels[i]

            # Decode with and without skipping special tokens
            decoded_full = tokenizer.decode(
                ids[attn.bool()].tolist(),
                skip_special_tokens=False,
            )
            decoded_no_special = tokenizer.decode(
                ids[attn.bool()].tolist(),
                skip_special_tokens=True,
            )

            f.write("\n[Decoded input_ids (with special tokens)]:\n")
            f.write(decoded_full + "\n\n")

            f.write("[Decoded input_ids (skip_special_tokens=True)]:\n")
            f.write(decoded_no_special + "\n\n")

            # Show where labels start being non -100
            supervised_positions = (lbls != -100).nonzero(as_tuple=False).view(-1)
            if len(supervised_positions) > 0:
                first_supervised = supervised_positions[0].item()
                f.write(f"[Labels] first supervised token index: {first_supervised}\n")

                supervised_ids = ids[first_supervised : attn.sum()]
                decoded_supervised = tokenizer.decode(
                    supervised_ids.tolist(),
                    skip_special_tokens=True,
                )
                f.write("[Decoded supervised region (assistant caption region)]:\n")
                f.write(decoded_supervised + "\n")
            else:
                f.write("[Labels] No supervised tokens found (all -100?)\n")

            f.write("\n\n")

    print(f"[debug] Wrote batch inspection to: {debug_path}")


if __name__ == "__main__":
    main()
