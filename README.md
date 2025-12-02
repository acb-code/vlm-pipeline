# VLM Fine-Tuning Pipeline

Small, config-driven pipeline for adapting Qwen3-VL to image–text tasks (currently Flickr8k captioning). Built to make it easy to swap datasets, run LoRA training on a GPU pod, and compare baseline vs finetuned performance.

## Current Results
- Base model: `Qwen/Qwen3-VL-8B-Instruct`
- Task: Flickr8k image captioning (2k train / 500 val, 1 epoch)
- LoRA config: r=16, alpha=32, dropout=0.05, target_modules: q/k/v/o/gate/up/down
- Metrics (val BLEU-4, max_new_tokens=64):
  - Baseline (zero-shot): **0.1009**
  - Finetuned LoRA: **0.1294** (+28% relative)

Artifacts to keep: `outputs/flickr8k_qwen3vl_lora/` (LoRA weights), `outputs/baseline_predictions.jsonl`, `outputs/lora_predictions.jsonl`, `outputs/metrics_lora.json`.

## Install
Prereqs: Python 3.10+, `uv` (`pip install uv`), and a CUDA build of torch/torchvision on GPU machines.

Clone and install deps (torch is installed separately to match your GPU):
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128  # adjust for your CUDA
uv sync
```

## Usage
All commands run from repo root.

### Local mock (no GPU, safe on WSL)
```bash
export USE_MOCK=True
uv run python main.py --mode baseline --config flickr8k_captioning.yaml
```
Outputs: `outputs/baseline_predictions.jsonl`

### GPU training + eval (e.g., RunPod RTX 5090)
```bash
export USE_MOCK=False
bash setup_gpu_env.sh          # installs uv deps, leaves your CUDA torch intact
wandb login                    # optional
uv run python main.py --mode train --config flickr8k_captioning.yaml
uv run python main.py --mode eval  --config flickr8k_captioning.yaml
```
Outputs:
- LoRA adapter: `outputs/flickr8k_qwen3vl_lora/`
- Finetuned preds/metrics: `outputs/lora_predictions.jsonl`, `outputs/metrics_lora.json`

## Notes
- Config-driven: edit `configs/flickr8k_captioning.yaml` to change dataset sizes, LoRA ranks, learning rate, or generation params.
- HF dataset: `jxie/flickr8k` (5 captions/image); training currently uses `caption_0`.
- Inference uses the correct Qwen3-VL chat + image template internally; no <image> token in the prompt.
- Next optimization (planned): swap HF generate() for vLLM to enable batched decoding, paged KV cache, and higher throughput on the same GPU.
