#!/usr/bin/env bash
set -e

export USE_MOCK=False
export CUDA_VISIBLE_DEVICES=0

uv run python main.py --mode eval --config flickr8k_captioning.yaml
