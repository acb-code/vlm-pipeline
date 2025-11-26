from typing import Dict, Optional

from datasets import DatasetDict, load_dataset


def load_flickr8k(
    split_train: str = "train",
    split_val: Optional[str] = None,
    split_test: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load the Flickr8k dataset using HuggingFace datasets and return a DatasetDict
    with train/validation/test splits.

    The HF 'flickr8k' dataset only has a 'train' split by default, so we
    create val/test splits from it if needed.

    Args:
        split_train: Name of the train split (if already present). If using default HF 'flickr8k',
                     this should remain 'train'.
        split_val:   Name of validation split. If None and the dataset has no such split,
                     a val split will be created from train.
        split_test:  Name of test split. If None and the dataset has no such split,
                     a test split will be created from train/val.
        max_*_samples: Optional limits for quick experiments.

    Returns:
        A DatasetDict with 'train', 'validation', and 'test' splits.
    """
    ds = load_dataset("flickr8k")

    # The HF `flickr8k` dataset comes as a single 'train' split.
    # We'll create validation + test splits if they're not provided.
    if isinstance(ds, DatasetDict):
        if "validation" not in ds or "test" not in ds:
            ds = _create_val_test_splits(ds["train"])
    else:
        # Just in case; treat as a single split and re-split.
        ds = _create_val_test_splits(ds)

    # Apply optional subsampling
    if max_train_samples is not None:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))

    if max_val_samples is not None:
        ds["validation"] = ds["validation"].select(
            range(min(max_val_samples, len(ds["validation"])))
        )

    if max_test_samples is not None:
        ds["test"] = ds["test"].select(range(min(max_test_samples, len(ds["test"]))))

    return ds


def _create_val_test_splits(train_ds, val_ratio: float = 0.1, test_ratio: float = 0.1) -> DatasetDict:
    """
    Create validation and test splits from a single train split.

    Args:
        train_ds: The original train Dataset.
        val_ratio: Proportion of samples for validation.
        test_ratio: Proportion of samples for test.

    Returns:
        DatasetDict with 'train', 'validation', and 'test'.
    """
    assert 0 < val_ratio < 1
    assert 0 < test_ratio < 1
    assert val_ratio + test_ratio < 1

    # First split off validation
    train_val = train_ds.train_test_split(test_size=val_ratio, shuffle=True)
    train_split = train_val["train"]
    val_split = train_val["test"]

    # Then split off test from the remaining train
    train_test = train_split.train_test_split(test_size=test_ratio / (1 - val_ratio), shuffle=True)
    final_train = train_test["train"]
    test_split = train_test["test"]

    return DatasetDict(
        {
            "train": final_train,
            "validation": val_split,
            "test": test_split,
        }
    )
