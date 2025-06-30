# email-spam-classifier
Implementing a ML model to detect SMS spam messages using NLP

## Project Structure
sms-spam-detection/
|
├── sms-spam-detection.ipynb
├── app.py
├── model.pkl
├── vectorizer.pkl
└── README.md

---

## Dataset
The dataset used consists of SMS messages labled as **spam** or **ham** (not spam). It goes through preprocessing steps such as:
- Lowercasing
- Stop word removal
- Punctuation removal
- Stemming
- Tokenization

![Feature Extraction](assets/feature-pairplot.png)

Dataset Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Model Training
