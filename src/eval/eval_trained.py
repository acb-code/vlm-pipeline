import json
from pathlib import Path

from datasets import Dataset
from evaluate import load as load_metric
from tqdm import tqdm

from src.data.flickr8k import load_flickr8k
from src.models.qwen_vl_loader import Qwen3VLLoader
from src.utils.wandb_utils import wandb_safe_log


def evaluate_lora_checkpoint(
    cfg,
    checkpoint_path: str,
    output_jsonl: str = "outputs/lora_predictions.jsonl",
    metrics_path: str = "outputs/metrics_lora.json",
):
    """
    Load a Qwen3-VL base model + LoRA adapter and evaluate on Flickr8k val split.
    """

    print("[eval_trained] Loading Qwen3-VL with LoRA adapter...")
    loader = Qwen3VLLoader(cfg, lora_adapter_dir=checkpoint_path)
    # loader.model is now the PEFT-wrapped model
    loader.model.eval()

    # Load dataset
    print("[eval_trained] Loading Flickr8k dataset...")
    ds = load_flickr8k(
        max_train_samples=cfg["dataset"].get("max_train_samples"),
        max_val_samples=cfg["dataset"].get("max_val_samples"),
        max_test_samples=cfg["dataset"].get("max_test_samples"),
    )
    ds_val: Dataset = ds["validation"]
    print(f"[eval_trained] Validation samples: {len(ds_val)}")

    bleu = load_metric("bleu")
    predictions = []
    references = []

    # Ensure output dir exists
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for ex in tqdm(ds_val, desc="Evaluating finetuned model"):
            image = ex["image"]
            refs = [
                ex["caption_0"],
                ex["caption_1"],
                ex["caption_2"],
                ex["caption_3"],
                ex["caption_4"],
            ]

            # Use the LoRA-augmented model for captioning
            pred = loader.generate_caption(image).strip()

            predictions.append(pred)
            references.append(refs)

            out_f.write(json.dumps({"prediction": pred, "references": refs}) + "\n")

    bleu_score = bleu.compute(predictions=predictions, references=references)

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump({"bleu": bleu_score["bleu"]}, f, indent=2)

    print(f"[eval_trained] BLEU: {bleu_score['bleu']:.4f}")
    print("[eval_trained] Results saved in outputs/")

    # log under a distinct key
    wandb_safe_log({"lora_bleu": bleu_score["bleu"]})

    return bleu_score
