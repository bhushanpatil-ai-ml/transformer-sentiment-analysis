import os
import re
import string
import pandas as pd
from datasets import load_dataset


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)              # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)     # remove URLs
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text


def load_imdb_data(save_path: str = "data/raw") -> None:
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)

    print("IMDb dataset downloaded and saved successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


def preprocess_and_save(
    train_path: str = "data/raw/train.csv",
    test_path: str = "data/raw/test.csv",
    output_path: str = "data/processed"
) -> None:
    os.makedirs(output_path, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["clean_review"] = train_df["text"].astype(str).apply(clean_text)
    test_df["clean_review"] = test_df["text"].astype(str).apply(clean_text)

    train_df["review_length"] = train_df["clean_review"].apply(lambda x: len(x.split()))
    test_df["review_length"] = test_df["clean_review"].apply(lambda x: len(x.split()))

    train_df.to_csv(os.path.join(output_path, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test_processed.csv"), index=False)

    print("Processed files saved successfully.")
    print(f"Processed train shape: {train_df.shape}")
    print(f"Processed test shape: {test_df.shape}")


if __name__ == "__main__":
    load_imdb_data()
    preprocess_and_save()