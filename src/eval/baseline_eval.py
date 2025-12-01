import json
import os
from typing import Dict, Any
from tqdm import tqdm
from datasets import Dataset
from evaluate import load as load_metric
from src.utils.wandb_utils import wandb_safe_log


def run_baseline_evaluation(
    model,
    ds_val: Dataset,
    cfg: Dict[str, Any],
    output_jsonl_path: str = "outputs/baseline_predictions.jsonl"
):
    """
    Evaluate a model (mock or real) on the Flickr8k validation set using BLEU.

    Args:
        model: Mock or real Qwen3-VL model with generate_caption(image)
        ds_val: HuggingFace validation split dataset
        cfg: full config dictionary
        output_jsonl_path: where to save qualitative predictions
    """

    print("[baseline_eval] Starting baseline evaluation...")
    print(f"[baseline_eval] Validation samples: {len(ds_val)}")

    # Load BLEU metric
    bleu = load_metric("bleu")

    predictions = []
    references = []

    # For logging qualitative examples
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    out_f = open(output_jsonl_path, "w", encoding="utf-8")

    for ex in tqdm(ds_val, desc="Evaluating"):
        image = ex["image"]
        # Use all five captions for BLEU reference
        ref_texts = [
            ex["caption_0"],
            ex["caption_1"],
            ex["caption_2"],
            ex["caption_3"],
            ex["caption_4"],
        ]

        pred = model.generate_caption(image).strip()

        predictions.append(pred)
        references.append(ref_texts)   # BLEU expects list of lists

        # Save qualitative sample
        out_f.write(json.dumps({
            "prediction": pred,
            "reference": ref_texts
        }) + "\n")

    out_f.close()

    # Compute BLEU
    bleu_score = bleu.compute(predictions=predictions, references=references)

    print("\n=== Baseline Evaluation Results ===")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print("===================================")

    wandb_safe_log({"baseline_bleu": bleu_score["bleu"]})


    return bleu_score
