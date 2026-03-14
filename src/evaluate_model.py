import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_baseline_model(
    model_path: str = "models/baseline_tfidf_logreg.pkl",
    test_path: str = "data/raw/test.csv",
    output_dir: str = "outputs/metrics",
    plots_dir: str = "outputs/plots",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    if "text" not in test_df.columns or "label" not in test_df.columns:
        raise ValueError("Test CSV must contain 'text' and 'label' columns.")

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Evaluation Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")
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
    plt.title("Baseline Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    image_path = os.path.join(plots_dir, "baseline_confusion_matrix.png")
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {image_path}")


if __name__ == "__main__":
    evaluate_baseline_model()