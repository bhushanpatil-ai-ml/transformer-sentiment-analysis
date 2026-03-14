import os
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_transformer():
    os.makedirs("models/transformer", exist_ok=True)
    os.makedirs("models/transformer/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    train_df = train_df[["clean_review", "label"]].dropna()
    test_df = test_df[["clean_review", "label"]].dropna()

    # CPU-friendly subset for faster local training
    train_df = train_df.sample(n=4000, random_state=42)
    test_df = test_df.sample(n=1000, random_state=42)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(batch):
        return tokenizer(
            batch["clean_review"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="models/transformer/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    print("\nTransformer Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    with open("outputs/metrics/transformer_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Transformer Evaluation Results:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    trainer.save_model("models/transformer")
    tokenizer.save_pretrained("models/transformer")

    print("\nTransformer model and tokenizer saved successfully.")


if __name__ == "__main__":
    train_transformer()