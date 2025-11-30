# ===============================================================
#  GPU Environment Setup for VLM Training (RunPod RTX5090)
# ===============================================================

set -e

echo "---------------------------------------------------------------"
echo " [1/5] Installing uv package manager"
echo "---------------------------------------------------------------"
pip install --upgrade pip
pip install uv

echo "---------------------------------------------------------------"
echo " [2/5] Installing CUDA 12.1 PyTorch + Torchvision"
echo "---------------------------------------------------------------"
pip install torch==2.8.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

echo "---------------------------------------------------------------"
echo "  Torch Version Check"
echo "---------------------------------------------------------------"
python3 - <<'EOF'
import torch, torchvision
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

echo "---------------------------------------------------------------"
echo " [3/5] Syncing Python dependencies using uv"
echo "---------------------------------------------------------------"
uv sync

echo "---------------------------------------------------------------"
echo " [4/5] Creating outputs/ directory"
echo "---------------------------------------------------------------"
mkdir -p outputs

echo "---------------------------------------------------------------"
echo " [5/5] Environment setup complete!"
echo " You can now run: bash train_vlm.sh"
echo "---------------------------------------------------------------"
