# MODEL_FYP.ipynb
Sentiment Analysis of Uber Reviews Using BERT

This repository contains the code and documentation for a sentiment analysis pipeline built using a fine-tuned BERT model on the Uber Reviews dataset. This project was developed as part of my Final Year Project (FYP) at NUML University under the BS Artificial Intelligence program.

🚀 Project Overview

Name: Feedback Flow (Sentiment Analysis Module)Role: Backend Lead & Model DeveloperObjective: Fine-tune a pre-trained BERT model to classify Uber customer reviews into positive, negative, and neutral sentiments, and serve predictions via a RESTful API built with Flas

Set up a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

🧰 Usage

1. Data Preprocessing
Clean and preprocess the raw Uber review dataset:
python src/train.py --mode preprocess --input data/uber_reviews_raw.csv --output data/uber_reviews_cleaned.csv

2. Model Training
Fine-tune the BERT model on the preprocessed dataset:
python src/train.py --mode train \
    --input data/uber_reviews_cleaned.csv \
    --model_name bert-base-uncased \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --output_dir models/

3. Batch Prediction

Run batch predictions on new data:
Payload: {"review": "Your review text here."}
Response: {"sentiment": "positive"}

📊 Model Performance
Accuracy
93%
Precision
0.88
Recall
0.90
F1-Score
0.89

🔍 Methodology
Data Collection & Cleaning: Removed duplicates, non-English reviews, and neutral noise.
Tokenization: Used BertTokenizer from Hugging Face Transformers.
Fine-tuning: Employed BertForSequenceClassification with cross-entropy loss.
Evaluation: Split data into train (80%), validation (10%), and test (10%) sets.

📝 Contributions
Muhammad Ahmed Imtiaz – Data preprocessing, model fine-tuning, API development.
Supervisor: DR Moiz ullah Ghauri, Department of Computer Science, NUML University.

📚 References
Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers

📞 Contact
Email: ahmedimtqureshi@gmail.com
LinkedIn: [https://linkedin.com/in/YourLinkedInProfile](https://www.linkedin.com/in/muhammad-ahmed-imtiaz-68332b282?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
GitHub: (https://github.com/Ahmedimtiaz-github)

🪪 License
This project is licensed under the MIT License - see the LICENSE file for details.


