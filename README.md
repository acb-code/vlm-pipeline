# VLM Fine-Tuning Pipeline

Small, config-driven pipeline for adapting Qwen3-VL to image–text tasks (currently Flickr8k captioning). Built to make it easy to swap datasets, run LoRA training on a GPU pod, and compare baseline vs finetuned performance.

## Results (full pipeline run)
- Base model: `Qwen/Qwen3-VL-8B-Instruct`; Task: Flickr8k captioning (2k train / 500 val, 1 epoch); LoRA: r=16, alpha=32, dropout=0.05 on q/k/v/o/gate/up/down.
- Eval command: `uv run python scripts/final_eval_viz.py --config flickr8k_captioning.yaml --num_samples 8`.
- Metrics (val `bleu_simple`, max_new_tokens=64): Baseline **0.0362**, LoRA **0.0587** (+62% relative). HF `evaluate` BLEU-4 from the same run: Baseline **0.1009**, LoRA **0.1294** (+28%).
- Visual QA: `outputs/final_eval/captions_grid.png` (below) shows val examples with reference, baseline caption, and LoRA caption. The LoRA consistently produces shorter, affordance-style sentences that match the training data format.
- Qualitative takeaways: LoRA shifts the captioning style prior (brevity, simpler verbs, present-progressive phrasing); object grounding stays similar with occasional color/detail drops; the consistent style change confirms the adapter is applied at inference.

![Baseline vs LoRA captions](outputs/final_eval/captions_grid.png)

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

### Final eval + visualization
Run a consolidated eval using a simple built-in corpus BLEU (no extra deps) plus a before/after grid:
```bash
uv run python scripts/final_eval_viz.py --config flickr8k_captioning.yaml --num_samples 8
```
Outputs to `outputs/final_eval/`:
- `metrics_final.json`: metric name used (bleu_simple) and baseline vs LoRA scores.
- `captions_grid.png`: random val images with reference, baseline caption, and LoRA caption for quick visual QA.

## Weights & Biases (optional)
- Enable by leaving `wandb.enabled: true` in `configs/flickr8k_captioning.yaml` (default) and logging in with `wandb login` or `export WANDB_API_KEY=...`. Mock mode (`USE_MOCK=True`) automatically disables logging to keep CPU runs light.
- Configure project/run names via the config keys: `wandb.project`, `wandb.entity` (username or team), `wandb.run_name`. Example override: `WANDB_PROJECT=my_proj WANDB_ENTITY=my_team uv run python main.py --mode train --config flickr8k_captioning.yaml`.
- Logged signals: training metrics from `Trainer`, baseline BLEU (`baseline_eval.py`), and finetuned BLEU (`eval_trained.py`), all gated behind the config so offline runs do not error.
- To disable entirely: set `wandb.enabled: false` in the config or `WANDB_DISABLED=true` in the environment.

## Generated Artifacts
- `outputs/flickr8k_qwen3vl_lora/`: PEFT LoRA adapter directory. Load with `Qwen3VLLoader(cfg, lora_adapter_dir=...)` for inference or reuse in downstream scripts; keep alongside the base checkpoint.
- `outputs/baseline_predictions.jsonl`: One JSON line per val image from the zero-shot model: `{"prediction": "...", "reference": ["ref1", ...]}`. Great for quick spot-checking model behavior.
- `outputs/lora_predictions.jsonl`: Same format as baseline but produced by the finetuned model. Compare side-by-side with the baseline file to see qualitative gains.
- `outputs/metrics_lora.json`: Stores scalar metrics like BLEU from the finetuned run. Use it to track runs or plot progress without re-evaluating.
- Re-generate any of the above by rerunning the commands in the Usage section; files are overwritten in place.

## BLEU Metric
- We report BLEU-4 on the Flickr8k validation split using all five human captions per image as references (`evaluate.load("bleu")`).
- Score range is 0–1; higher is better. In this setup, 0.10 → 0.13 reflects a meaningful quality bump, but absolute values stay low because captioning is diverse and BLEU is precision-focused.
- Small deltas (±0.01) are expected noise across seeds; sustained gains of ~0.02+ usually indicate a real improvement.
- Use BLEU here primarily for directional comparisons (baseline vs finetuned); pair with qualitative reads of the JSONL files above to judge caption fidelity and fluency.

## Notes
- Config-driven: edit `configs/flickr8k_captioning.yaml` to change dataset sizes, LoRA ranks, learning rate, or generation params.
- HF dataset: `jxie/flickr8k` (5 captions/image); training currently uses `caption_0`.
- Inference uses the correct Qwen3-VL chat + image template internally; no <image> token in the prompt.
- Next optimization (planned): swap HF generate() for vLLM to enable batched decoding, paged KV cache, and higher throughput on the same GPU.
