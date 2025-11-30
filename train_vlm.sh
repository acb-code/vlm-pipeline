# ===============================================================
#  Launch VLM LoRA Training (RunPod RTX5090)
# ===============================================================

set -e

echo "---------------------------------------------------------------"
echo " Starting VLM LoRA Training on GPU"
echo "---------------------------------------------------------------"

# Force real model mode
export USE_MOCK=False
export CUDA_VISIBLE_DEVICES=0

# Optional: Helps prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training
uv run python main.py --mode train --config flickr8k_captioning.yaml
