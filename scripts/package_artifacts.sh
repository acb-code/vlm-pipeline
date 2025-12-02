#!/usr/bin/env bash
# Package training/eval artifacts into a portable tarball.
# Usage: bash scripts/package_artifacts.sh [TAG] [CONFIG_PATH]
# - TAG (optional): name suffix, default = timestamp
# - CONFIG_PATH (optional): default = configs/flickr8k_captioning.yaml

set -euo pipefail

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${2:-configs/flickr8k_captioning.yaml}"
ARTIFACT_DIR="artifacts_${TAG}"

echo "[package] Using config: ${CONFIG_PATH}"
echo "[package] Collecting artifacts into: ${ARTIFACT_DIR}"

mkdir -p "${ARTIFACT_DIR}"

# Copy config for reproducibility
cp "${CONFIG_PATH}" "${ARTIFACT_DIR}/"

# Discover LoRA adapter dir from config
ADAPTER_DIR=$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os, yaml
cfg_path = os.environ["CONFIG_PATH"]
cfg = yaml.safe_load(open(cfg_path, "r"))
print(cfg["training"]["output_dir"])
PY
)

if [ -d "${ADAPTER_DIR}" ]; then
  echo "[package] Adding LoRA adapter: ${ADAPTER_DIR}"
  cp -r "${ADAPTER_DIR}" "${ARTIFACT_DIR}/"
else
  echo "[package] WARNING: Adapter dir not found: ${ADAPTER_DIR}"
fi

# Optional prediction/metric files
for f in outputs/baseline_predictions.jsonl outputs/lora_predictions.jsonl outputs/metrics_lora.json; do
  if [ -f "${f}" ]; then
    echo "[package] Adding ${f}"
    cp "${f}" "${ARTIFACT_DIR}/"
  fi
done

TAR_NAME="${ARTIFACT_DIR}.tar.gz"
tar -czf "${TAR_NAME}" "${ARTIFACT_DIR}"
echo "[package] Created ${TAR_NAME}"
echo "[package] Done."
