from typing import Optional
from datasets import DatasetDict, load_dataset


def load_flickr8k(
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load the Flickr8k dataset from 'jxie/flickr8k', which already contains
    train/validation/test splits and the following columns:

        - image: PIL.Image
        - caption: str
        - split: 'train' | 'val' | 'test'

    All we need to do is select splits and optionally limit sample counts.
    """

    raw = load_dataset("jxie/flickr8k")

    # The dataset comes as:
    # raw["train"], raw["validation"], raw["test"]
    train_ds = raw["train"]
    val_ds = raw["validation"]
    test_ds = raw["test"]

    if max_train_samples:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))

    if max_val_samples:
        val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))

    if max_test_samples:
        test_ds = test_ds.select(range(min(max_test_samples, len(test_ds))))

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })
