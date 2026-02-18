# 🚗 Sentiment Analysis BERT Model — FYP

A **BERT-based sentiment analysis model** fine-tuned on an **Uber reviews dataset**, developed as part of the Final Year Project (FYP) for an **AI-based Transport App** at the **National University of Modern Languages (NUML)**.

---

## ✨ Overview

This project fine-tunes a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model to classify user sentiments from Uber ride reviews. The model is designed to integrate into a university transport application, enabling real-time feedback analysis to improve service quality.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🧠 **BERT Fine-Tuning** | Transfer learning on a pre-trained BERT model for domain-specific sentiment classification |
| 📊 **Uber Reviews Dataset** | Trained on real-world Uber ride review data for practical relevance |
| 🏷️ **Sentiment Classification** | Classifies reviews into sentiment categories (e.g., Positive, Negative, Neutral) |
| 🎓 **University FYP** | Developed as part of an AI-based transport application for NUML |
| 🐍 **Pure Python** | Entire pipeline implemented in Python |

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Deep Learning Framework**: PyTorch / TensorFlow
- **NLP Model**: BERT (`bert-base-uncased` or similar)
- **Libraries**: Hugging Face Transformers, scikit-learn, Pandas, NumPy, Matplotlib
- **Environment**: Jupyter Notebook

---

## 📁 Project Structure

```
MODEL_FYP.ipynb/
├── MODEL_FYP.ipynb        # Main notebook — data loading, preprocessing, BERT fine-tuning, evaluation
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- GPU recommended for faster training (Google Colab works great)

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmedimtiaz-github/MODEL_FYP.ipynb.git
cd MODEL_FYP.ipynb
```

### 2. Install Dependencies

```bash
pip install transformers torch torchvision pandas numpy scikit-learn matplotlib tqdm
```

### 3. Run the Notebook

```bash
jupyter notebook MODEL_FYP.ipynb
```

Or open in **Google Colab** for GPU-accelerated training.

---

## 📈 Model Pipeline

```
Uber Reviews Dataset
        │
        ▼
┌─────────────────┐
│  Data Cleaning   │  ← Remove noise, handle missing values
│  & Preprocessing │  ← Tokenization with BERT tokenizer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BERT Model      │  ← Fine-tune pre-trained BERT
│  Fine-Tuning     │  ← Add classification head
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation      │  ← Accuracy, Precision, Recall, F1-Score
│  & Metrics       │  ← Confusion Matrix
└────────┬────────┘
         │
         ▼
   Sentiment Predictions
   (Positive / Negative / Neutral)
```

---

## 📊 Evaluation Metrics

The model is evaluated using standard NLP classification metrics:

- **Accuracy** — Overall correctness
- **Precision** — Positive predictive value
- **Recall** — Sensitivity / True positive rate
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of predictions

---

## 🎓 About the Project

This model is a core component of an **AI-based university transport application** developed as a Final Year Project at **NUML (National University of Modern Languages)**. The sentiment analysis module processes user feedback on transport services, enabling:

- 📋 Automated review classification
- 📈 Service quality monitoring
- 🔔 Alert generation for negative sentiment trends
- 📊 Dashboard-ready sentiment summaries

---

## 🤝 Author

**M. Ahmed Imtiaz**
- GitHub: [@Ahmedimtiaz-github](https://github.com/Ahmedimtiaz-github)
- University: National University of Modern Languages (NUML)