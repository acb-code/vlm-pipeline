import argparse
import yaml

from src.utils.paths import get_config_path
from src.utils.seed import set_global_seed
from src.data.flickr8k import load_flickr8k
from src.eval.baseline_eval import run_baseline_evaluation
from src.training.trainer import run_lora_training

import wandb

# ===== MOCK MODE SWITCH =====
import os
USE_MOCK = os.environ.get("USE_MOCK", "True").lower() == "true"
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

    # W&B initialization
    if cfg.get("wandb", {}).get("enabled", False) and not USE_MOCK:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=cfg["wandb"]["run_name"]
        )
    else:
        print("[wandb] Disabled (mock mode or config)")


    if args.mode == "baseline":
        print("[main] Running baseline evaluation...")

        ds = load_flickr8k(
            max_train_samples=cfg["dataset"].get("max_train_samples"),
            max_val_samples=cfg["dataset"].get("max_val_samples"),
            max_test_samples=cfg["dataset"].get("max_test_samples"),
        )
        ds_val = ds["validation"]

        model = ModelClass(cfg)

        run_baseline_evaluation(
            model=model,
            ds_val=ds_val,
            cfg=cfg,
            output_jsonl_path="outputs/baseline_predictions.jsonl",
        )

    elif args.mode == "train":
        if USE_MOCK:
            print("[main] TRAIN requested but USE_MOCK=True. Refusing to run heavy training on this machine.")
            print("       Set USE_MOCK=False and run on your GPU box instead.")
            return

        print("[main] Running LoRA training...")
        run_lora_training(cfg)

    elif args.mode == "eval":
        print("[main] Eval mode is not implemented yet.")
    
    elif args.mode == "eval_trained":
        from src.eval.eval_trained import evaluate_lora_checkpoint

        if USE_MOCK:
            print("[main] Cannot evaluate trained model in mock mode.")
            return

        checkpoint = "outputs/checkpoints/last"  # or your chosen path
        evaluate_lora_checkpoint(cfg, checkpoint)


if __name__ == "__main__":
    main()
