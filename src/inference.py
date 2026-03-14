import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


MODEL_PATH = "models/transformer"


def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    label_map = {0: "Negative", 1: "Positive"}
    return label_map[prediction]


def main():
    tokenizer, model = load_model()

    print("Sentiment Analysis Inference")
    print("----------------------------")

    while True:
        text = input("\nEnter a review (or type 'exit'): ")

        if text.lower() == "exit":
            break

        sentiment = predict_sentiment(text, tokenizer, model)

        print(f"Predicted Sentiment: {sentiment}")


if __name__ == "__main__":
    main()