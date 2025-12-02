#!/usr/bin/env python3
"""
Final evaluation + visualization for Flickr8k captioning.

Computes a simple corpus BLEU (no extra deps) for baseline vs finetuned predictions
and saves a before/after caption grid sampled from validation data.
"""

import argparse
import json
import math
import random
import textwrap
from pathlib import Path
from typing import List, Tuple, Callable

import matplotlib

# Headless rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.flickr8k import load_flickr8k  # noqa: E402
from src.utils.paths import get_config_path  # noqa: E402


def load_jsonl_predictions(path: Path) -> List[dict]:
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append(
                {
                    "prediction": obj["prediction"],
                    "references": obj.get("references") or obj.get("reference") or [],
                }
            )
    return preds


def extract_captions(example: dict) -> List[str]:
    """Return all caption_* fields, preserving order by suffix number."""
    caps = []
    for key in sorted(example.keys()):
        if key.startswith("caption_"):
            caps.append(example[key])
    return caps


def select_metric(
    references: List[str], probe_preds: List[str]
) -> Tuple[str, Callable[[List[str]], float], List[str]]:
    """
    Simple built-in metric: corpus BLEU (n=1..4) without extra deps.
    Uses first reference caption per image for scoring.
    """

    def corpus_bleu(preds: List[str]) -> float:
        tokenized = [
            (pred.lower().split(), ref.lower().split())
            for pred, ref in zip(preds, references)
        ]
        total_pred_lengths = sum(len(p) for p, _ in tokenized)
        total_ref_lengths = sum(len(r) for _, r in tokenized)

        def ngrams(tokens, n):
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        epsilon = 1e-9
        precisions = []

        for n in range(1, 5):
            num, denom = 0, 0
            for pred_tokens, ref_tokens in tokenized:
                pred_counts = {}
                for ng in ngrams(pred_tokens, n):
                    pred_counts[ng] = pred_counts.get(ng, 0) + 1

                ref_counts = {}
                for ng in ngrams(ref_tokens, n):
                    ref_counts[ng] = ref_counts.get(ng, 0) + 1

                overlap = 0
                for ng, c in pred_counts.items():
                    overlap += min(c, ref_counts.get(ng, 0))

                num += overlap
                denom += max(len(pred_tokens) - n + 1, 0)

            precisions.append((num + epsilon) / (denom + epsilon))

        # Brevity penalty
        bp = (
            1.0
            if total_pred_lengths > total_ref_lengths
            else math.exp(1 - (total_ref_lengths + epsilon) / (total_pred_lengths + epsilon))
        )
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
        return float(bp * geo_mean)

    # Probe
    corpus_bleu(probe_preds)
    return "bleu_simple", corpus_bleu, []


def render_caption_grid(samples: List[dict], out_path: Path):
    cols = 2
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()

    for ax, sample in zip(axes, samples):
        ax.imshow(sample["image"])
        ax.axis("off")
        ref_txt = textwrap.fill(f"Ref: {sample['reference']}", width=60)
        base_txt = textwrap.fill(f"Baseline: {sample['baseline']}", width=60)
        lora_txt = textwrap.fill(f"LoRA: {sample['lora']}", width=60)
        ax.set_title(f"{ref_txt}\n{base_txt}\n{lora_txt}", fontsize=9, loc="left")

    # Blank remaining axes if any
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Final eval + viz for captioning")
    parser.add_argument("--config", type=str, default="flickr8k_captioning.yaml")
    parser.add_argument(
        "--baseline_preds",
        type=str,
        default="outputs/baseline_predictions.jsonl",
    )
    parser.add_argument(
        "--lora_preds",
        type=str,
        default="outputs/lora_predictions.jsonl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="How many random examples to visualize.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/final_eval", help="Where to write metrics and figures."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config to match dataset slicing
    cfg_path = get_config_path(args.config)
    cfg = yaml.safe_load(open(cfg_path, "r"))
    ds = load_flickr8k(
        max_train_samples=cfg["dataset"].get("max_train_samples"),
        max_val_samples=cfg["dataset"].get("max_val_samples"),
        max_test_samples=cfg["dataset"].get("max_test_samples"),
    )
    ds_val = ds["validation"]

    baseline = load_jsonl_predictions(Path(args.baseline_preds))
    lora = load_jsonl_predictions(Path(args.lora_preds))

    total = min(len(ds_val), len(baseline), len(lora))
    if total == 0:
        raise ValueError("No overlapping samples found between dataset and prediction files.")

    # Truncate everything to shared length
    ds_val = ds_val.select(range(total))
    images = [ex["image"] for ex in tqdm(ds_val, desc="Loading images")]
    ref_texts = [extract_captions(ex)[0] for ex in ds_val]
    baseline_text = [p["prediction"] for p in baseline[:total]]
    lora_text = [p["prediction"] for p in lora[:total]]

    # Pick metric once, then apply to both
    metric_name, scorer, diag = select_metric(
        references=ref_texts,
        probe_preds=baseline_text[: min(8, total)],
    )

    baseline_score = scorer(baseline_text)
    lora_score = scorer(lora_text)
    metrics_path = out_dir / "metrics_final.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metric": metric_name,
                "baseline": baseline_score,
                "lora": lora_score,
                "diagnostics": diag,
            },
            f,
            indent=2,
        )

    print(f"[final_eval] {metric_name}: baseline={baseline_score:.4f}, lora={lora_score:.4f}")
    if diag:
        print("[final_eval] Diagnostics:", " | ".join(diag))

    # Visualization
    sample_count = min(args.num_samples, total)
    sample_ids = random.sample(range(total), sample_count)
    samples = []
    for idx in sample_ids:
        ex = ds_val[idx]
        caps = extract_captions(ex)
        samples.append(
            {
                "image": ex["image"],
                "reference": caps[0] if caps else "",
                "baseline": baseline_text[idx],
                "lora": lora_text[idx],
            }
        )

    viz_path = out_dir / "captions_grid.png"
    render_caption_grid(samples, viz_path)
    print(f"[final_eval] Saved visualization to {viz_path}")
    print(f"[final_eval] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
