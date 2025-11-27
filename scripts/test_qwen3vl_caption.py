#!/usr/bin/env python3
"""
Safe test: uses a mock VLM when on CPU / WSL to avoid crashes.

Run:
    uv run python scripts/test_qwen3vl_caption.py
"""

import yaml
from src.utils.paths import get_config_path
from src.data.flickr8k import load_flickr8k


# ====== Toggle between real and mock models ======
USE_MOCK = True  # <---- Change this to False ONLY on a GPU machine
# ==================================================

if USE_MOCK:
    from src.models.mock_vl_model import MockQwenVLModel as ModelClass
else:
    from src.models.qwen_vl_loader import Qwen3VLLoader as ModelClass


def main():
    # Load config
    cfg_path = get_config_path("flickr8k_captioning.yaml")
    cfg = yaml.safe_load(open(cfg_path, "r"))

    if USE_MOCK:
        print("\n[INFO] Running in MOCK MODE — safe for WSL CPU.")
    else:
        cfg["model"]["device"] = "cuda"
        cfg["model"]["torch_dtype"] = "float16"

    # Init model
    loader = ModelClass(cfg)

    # Load one image for testing pipeline
    ds = load_flickr8k(max_train_samples=1)
    img = ds["train"][0]["image"]

    print("Generating caption...")
    caption = loader.generate_caption(img)
    print("\n=== Caption Result ===")
    print(caption)
    print("======================")


if __name__ == "__main__":
    main()
