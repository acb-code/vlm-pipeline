#!/usr/bin/env bash
# GPU environment setup for VLM training (RunPod / CUDA box)
set -euo pipefail

TORCH_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "---------------------------------------------------------------"
echo " [1/5] Installing uv (and upgrading pip)"
echo "---------------------------------------------------------------"
pip install --upgrade pip
pip install uv

echo "---------------------------------------------------------------"
echo " [2/5] Syncing project dependencies with uv"
echo "---------------------------------------------------------------"
uv sync

echo "---------------------------------------------------------------"
echo " [3/5] Ensuring pip inside .venv"
echo "---------------------------------------------------------------"
uv run python -m ensurepip --upgrade

echo "---------------------------------------------------------------"
echo " [4/5] Installing CUDA torch/torchvision into .venv (if missing)"
echo "---------------------------------------------------------------"
# Install torch only if not already importable in the venv
uv run python - <<EOF || uv run python -m pip install torch==2.8.0 --index-url "${TORCH_INDEX}"
import torch, platform
print("Torch already in .venv:", torch.__version__, "CUDA:", torch.version.cuda, "platform:", platform.platform())
EOF
# Always ensure torchvision matches the CUDA wheel index
uv run python -m pip install --index-url "${TORCH_INDEX}" torchvision

echo "---------------------------------------------------------------"
echo " [5/5] Version check"
echo "---------------------------------------------------------------"
uv run python - <<'EOF'
import torch, torchvision
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

echo "---------------------------------------------------------------"
echo " Setup complete. You can now run: bash train_vlm.sh"
echo "---------------------------------------------------------------"
