from src.data.flickr8k import load_flickr8k


def main():
    print("Downloading Flickr8k via HuggingFace datasets...")
    ds = load_flickr8k(max_train_samples=10, max_val_samples=10, max_test_samples=10)
    print(ds)
    print("Done. Dataset is cached in the HF datasets cache directory.")


if __name__ == "__main__":
    main()
