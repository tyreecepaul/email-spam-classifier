# sms-spam-classifier
Implementing a ML model to detect SMS spam messages using NLP

![SMS Final Screenshot](assets/spam-email.png)

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

Dataset Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


### Feature Pairplot
![Feature Extraction](assets/feature-pairplot.png)

### Spam Word Cloud
![Spam WC](assets/spam-wc.png)

### Ham Word Cloud
![Ham WC](assets/ham-wc.png)

### Spam Histogram
![Spam Hist](assets/spam-hist.png)

### Ham Histogram
![Ham Hist](assets/ham-hist.png)

## Model Training
A set of classifiersd are trained using Scikit-learn. Each model is evaluated based on accuracy, precision and recall.

![Model Comparison](assets/algo-catplot.png)

## Deployment + Final Result
![SMS Spam Message](assets/spam-email.png)
![SMS Ham Message](assets/ham-email.png)

## Credit
Youtube Tutorial: https://www.youtube.com/watch?v=YncZ0WwxyzU&t=4248s
