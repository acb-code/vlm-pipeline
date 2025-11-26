import argparse
import yaml

from src.utils.paths import get_config_path
from src.utils.seed import set_global_seed
from src.data.flickr8k import load_flickr8k


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Pipeline Entry Point")
    parser.add_argument(
        "--config",
        type=str,
        default="flickr8k_captioning.yaml",
        help="Name of the config file in configs/",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "train", "eval"],
        default="baseline",
        help="What to run.",
    )
    return parser.parse_args()


def load_config(config_name: str) -> dict:
    config_path = get_config_path(config_name)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_global_seed(cfg["training"].get("seed", 42))

    if args.mode == "baseline":
        # For now, just prove dataset + config plumbing works
        ds = load_flickr8k(
            max_train_samples=cfg["dataset"].get("max_train_samples"),
            max_val_samples=cfg["dataset"].get("max_val_samples"),
            max_test_samples=cfg["dataset"].get("max_test_samples"),
        )
        print(ds)
        print("Baseline mode stub – model loading and eval will be added next.")
    elif args.mode == "train":
        print("Train mode stub – training loop will be added next.")
    elif args.mode == "eval":
        print("Eval mode stub – post-training evaluation will be added next.")


if __name__ == "__main__":
    main()
