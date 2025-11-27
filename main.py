import argparse
import yaml
import os

from src.utils.paths import get_config_path
from src.utils.seed import set_global_seed
from src.data.flickr8k import load_flickr8k

from src.eval.baseline_eval import run_baseline_evaluation

# ===== MOCK MODE SWITCH =====
USE_MOCK = True  # <--- SAFE FOR WSL LOCAL DEV
# ============================

if USE_MOCK:
    from src.models.mock_vl_model import MockQwenVLModel as ModelClass
else:
    from src.models.qwen_vl_loader import Qwen3VLLoader as ModelClass


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Pipeline Entry Point")
    parser.add_argument("--config", type=str, default="flickr8k_captioning.yaml")
    parser.add_argument("--mode", type=str, choices=["baseline", "train", "eval"], default="baseline")
    return parser.parse_args()


def load_config(config_name: str) -> dict:
    path = get_config_path(config_name)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_global_seed(cfg["training"].get("seed", 42))

    if args.mode == "baseline":
        print("[main] Running baseline evaluation...")

        # Load validation dataset
        ds = load_flickr8k(
            max_train_samples=cfg["dataset"].get("max_train_samples"),
            max_val_samples=cfg["dataset"].get("max_val_samples"),
            max_test_samples=cfg["dataset"].get("max_test_samples"),
        )

        ds_val = ds["validation"]

        # Init model (mock or real)
        model = ModelClass(cfg)

        # Run eval
        run_baseline_evaluation(
            model=model,
            ds_val=ds_val,
            cfg=cfg,
            output_jsonl_path="outputs/baseline_predictions.jsonl",
        )

    elif args.mode == "train":
        print("[main] TRAIN MODE NOT IMPLEMENTED YET")

    elif args.mode == "eval":
        print("[main] EVAL MODE NOT IMPLEMENTED YET")


if __name__ == "__main__":
    main()
