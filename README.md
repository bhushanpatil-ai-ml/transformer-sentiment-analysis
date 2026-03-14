# 🤖 Transformer Sentiment Analysis – Advanced NLP Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![NLP](https://img.shields.io/badge/NLP-Transformers-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# 📌 Project Overview

This project implements an **end-to-end Natural Language Processing pipeline** for **sentiment analysis of movie reviews** using both:

• Traditional Machine Learning  
• Transformer-based Deep Learning  

The project compares the performance of a **TF-IDF + Logistic Regression baseline model** with a **DistilBERT transformer model**.

The pipeline includes:

• Data preprocessing  
• Exploratory Data Analysis (EDA)  
• Baseline ML model training  
• Transformer fine-tuning  
• Model evaluation  
• Real-time sentiment inference  

The goal is to build a **robust sentiment classification system** capable of understanding natural language and predicting whether a review is **positive or negative**.

---

# 🎯 Problem Statement

Understanding customer sentiment from text reviews is essential for:

• Product review platforms  
• Social media monitoring  
• Customer feedback analysis  
• Recommendation systems  
• Market sentiment tracking  

Manual analysis of thousands of reviews is inefficient.  
Therefore, this project builds an **automated NLP pipeline** that can classify review sentiment using machine learning and deep learning techniques.

---

# 📊 Dataset

Dataset used:

**IMDb Movie Reviews Dataset**

Source: HuggingFace Datasets Library

Dataset characteristics:

• 50,000 movie reviews  
• Balanced dataset  
• Binary sentiment classification  

Label Encoding:

```
0 → Negative
1 → Positive
```

Dataset split:

• Training set  
• Testing set

---

# 🛠 Technologies Used

• Python  
• Pandas  
• NumPy  
• Scikit-learn  
• Matplotlib  
• Seaborn  
• HuggingFace Transformers  
• PyTorch  
• Datasets Library  

---

# 📂 Project Structure

```
transformer-sentiment-analysis/

├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Cleaned dataset
│
├── models/
│   ├── baseline_model.pkl      # TF-IDF + Logistic Regression
│   └── transformer/            # DistilBERT trained model
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── outputs/
│   ├── plots/                  # Visualizations
│   └── metrics/                # Evaluation metrics
│
├── src/
│   ├── data_preprocessing.py   # Text cleaning & dataset preparation
│   ├── train_baseline.py       # Classical ML model
│   ├── train_transformer.py    # Transformer training
│   ├── evaluate_model.py       # Model evaluation
│   ├── inference.py            # Sentiment prediction script
│   └── utils.py                # Helper functions
│
├── requirements.txt
└── README.md
```

---

# 🔎 Data Preprocessing

Text preprocessing steps include:

• Lowercasing text  
• Removing HTML tags  
• Removing URLs  
• Removing numbers  
• Removing punctuation  
• Removing extra whitespace  

Example transformation:

```
Original:
"This movie was AMAZING!!! 10/10"

Processed:
"this movie was amazing"
```

---

# 🔬 Exploratory Data Analysis

The EDA notebook analyzes:

• Dataset structure  
• Class distribution  
• Review length distribution  
• Word frequency patterns  
• Sentiment balance  

EDA helps understand the dataset before model training.

---

# 🤖 Machine Learning Models

Two models were implemented and compared.

---

# 1️⃣ Baseline Model

Model:

**TF-IDF + Logistic Regression**

Pipeline:

```
Text → TF-IDF Vectorizer → Logistic Regression
```

Advantages:

• Fast training  
• Strong traditional NLP baseline  
• Simple and interpretable

---

# 2️⃣ Transformer Model

Model used:

**DistilBERT**

DistilBERT is a compressed version of BERT that:

• retains strong language understanding  
• is faster and lighter than BERT  
• performs well on text classification tasks

Model architecture:

```
Input Text
   ↓
Tokenizer
   ↓
DistilBERT Encoder
   ↓
Classification Head
   ↓
Sentiment Prediction
```

---

# 📈 Model Evaluation Metrics

Models were evaluated using:

• Accuracy  
• Precision  
• Recall  
• F1 Score  
• Classification Report  
• Confusion Matrix  

These metrics help measure model performance and classification quality.

---

# 🏆 Model Performance

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

Example results:

| Model | Accuracy | F1 Score |
|------|------|------|
Baseline (TF-IDF + Logistic Regression) | ~0.89 | ~0.89 |
DistilBERT Transformer | ~0.83 | ~0.83 |

The baseline model performed slightly better due to limited training resources.

With full GPU training, transformer models generally outperform classical approaches.

---

# 🔍 Inference (Real-Time Prediction)

The project includes an interactive sentiment prediction script.

Run:

```
python src/inference.py
```

Example:

```
Enter a review: This movie was fantastic
Predicted Sentiment: Positive

Enter a review: The film was boring and terrible
Predicted Sentiment: Negative

```

## Example Prediction

Input:
"This movie was absolutely fantastic and inspiring!"

Output:
Sentiment: Positive

---

# ⚙ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/transformer-sentiment-analysis.git
cd transformer-sentiment-analysis
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶ Running the Project

Preprocess dataset:

```
python src/data_preprocessing.py
```

Train baseline model:

```
python src/train_baseline.py
```

Train transformer model:

```
python src/train_transformer.py
```

Run inference:

```
python src/inference.py
```

---

# 🚀 Future Improvements

Possible future enhancements:

• Train transformer model using GPU  
• Hyperparameter tuning with Optuna  
• Experiment tracking using MLflow  
• Deploy model using FastAPI  
• Docker containerization  
• Build a web UI for sentiment prediction  

---

# 👨‍💻 Author

**Bhushan Patil**

AI / Machine Learning Engineer  
Pune, Maharashtra, India

---

# ⭐ If you found this project useful

Consider giving it a **star on GitHub**.