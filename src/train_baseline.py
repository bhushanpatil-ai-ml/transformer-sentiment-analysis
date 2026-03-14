import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_baseline(
    train_path: str = "data/processed/train_processed.csv",
    test_path: str = "data/processed/test_processed.csv",
    model_dir: str = "models",
    output_dir: str = "outputs/metrics",
    plots_dir: str = "outputs/plots",
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["clean_review"]
    y_train = train_df["label"]

    X_test = test_df["clean_review"]
    y_test = test_df["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Baseline Model Performance:")
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1_score:  {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    with open(os.path.join(output_dir, "baseline_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Baseline Model Performance:\n")
        f.write(f"accuracy:  {accuracy:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall:    {recall:.4f}\n")
        f.write(f"f1_score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_confusion_matrix.png"))
    plt.show()

    joblib.dump(pipeline, os.path.join(model_dir, "baseline_tfidf_logreg.pkl"))
    print("\nBaseline model saved successfully.")
    print("Confusion matrix plot saved successfully.")


if __name__ == "__main__":
    train_baseline()