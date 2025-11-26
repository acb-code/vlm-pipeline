import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Return the root directory of the project.

    Assumes this file lives in src/utils/ and the project root is two levels up.
    Adjust if you move things around.
    """
    return Path(__file__).resolve().parents[2]


def get_config_path(name: str) -> Path:
    """
    Get the path to a config file in configs/.

    Example:
        get_config_path("flickr8k_captioning.yaml")
    """
    root = get_project_root()
    return root / "configs" / name


def get_data_dir() -> Path:
    """
    Return the base data directory.
    """
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """
    Directory where raw datasets live (e.g., HF downloads, archive extractions).
    """
    return get_data_dir() / "raw"


def get_output_dir(subdir: str = "") -> Path:
    """
    Directory for model outputs, logs, checkpoints, etc.

    Args:
        subdir: Optional subdirectory, e.g. "flickr8k_qwen3vl_lora"
    """
    base = get_project_root() / "outputs"
    return base / subdir if subdir else base
