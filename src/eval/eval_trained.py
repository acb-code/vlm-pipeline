import json
from datasets import Dataset
from evaluate import load as load_metric
from peft import PeftModel

from src.data.flickr8k import load_flickr8k
from src.models.qwen_vl_loader import Qwen3VLLoader


def evaluate_lora_checkpoint(cfg, checkpoint_path, output_jsonl="outputs/lora_predictions.jsonl"):

    print(f"[eval_trained] Loading base Qwen3-VL model...")
    loader = Qwen3VLLoader(cfg)
    base_model = loader.model

    print(f"[eval_trained] Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    processor = loader.processor

    # Load dataset
    ds = load_flickr8k(
        max_train_samples=cfg["dataset"].get("max_train_samples"),
        max_val_samples=cfg["dataset"].get("max_val_samples"),
        max_test_samples=cfg["dataset"].get("max_test_samples"),
    )
    ds_val: Dataset = ds["validation"]

    bleu = load_metric("bleu")
    predictions = []
    references = []

    out_f = open(output_jsonl, "w", encoding="utf-8")

    for ex in ds_val:
        image = ex["image"]
        refs = [
            ex["caption_0"], ex["caption_1"],
            ex["caption_2"], ex["caption_3"],
            ex["caption_4"]
        ]

        # Use the LoRA-augmented model for captioning
        pred = loader.generate_caption(image).strip()

        predictions.append(pred)
        references.append(refs)

        out_f.write(json.dumps({"prediction": pred, "references": refs}) + "\n")

    out_f.close()

    bleu_score = bleu.compute(predictions=predictions, references=references)

    # Save metrics
    with open("outputs/metrics_lora.json", "w") as f:
        json.dump({"bleu": bleu_score["bleu"]}, f, indent=2)

    print(f"[eval_trained] BLEU: {bleu_score['bleu']:.4f}")
    print("[eval_trained] Results saved in outputs/")
