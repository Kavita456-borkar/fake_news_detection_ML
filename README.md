📰 Fake News Detection Using Machine Learning
📌 Project Description

This project builds a Fake News Detection System using Machine Learning and Natural Language Processing (NLP).

The goal of this project is to automatically classify news articles as:

0 → Fake News

1 → Real News

With the rapid spread of misinformation online, automated fake news detection systems are essential. This project demonstrates how text data can be processed and used to train a machine learning model for accurate classification.

🎯 What I Did in This Project
1️⃣ Data Collection

Used datasets such as:

train.tsv

fake.csv

The dataset contained different truth levels like:

false

barely-true

half-true

mostly-true

true

pants-fire

2️⃣ Data Cleaning & Preprocessing

To prepare the text data for machine learning, I performed:

Removed unnecessary columns

Converted text to lowercase

Removed punctuation and special characters

Removed stopwords using NLTK

Applied lemmatization to reduce words to their base form

Converted multi-class labels into binary classification (Real/Fake)

Label Mapping:

Original Label	Converted To
false	0
pants-fire	0
barely-true	0
half-true	1
mostly-true	1
true	1
3️⃣ Feature Engineering

Used TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer

Converted textual data into numerical feature vectors

This allows machine learning algorithms to understand text data

4️⃣ Model Building

Split the dataset into:

Training set

Testing set

Trained a PassiveAggressiveClassifier

The model learns from the training data and adjusts weights for correct classification

Why Passive Aggressive?

Works well for large-scale text classification

Efficient and fast

Good performance for binary classification problems

5️⃣ Model Evaluation

Evaluated the model using:

Accuracy Score

Confusion Matrix

Precision

Recall

F1-Score

This helped measure how well the model distinguishes between fake and real news.

6️⃣ Model Saving

Saved the trained model using pickle

This allows reuse of the model without retraining

Example:

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

To load later:

model = pickle.load(open('model.pkl', 'rb'))
🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

NLTK

Matplotlib

Seaborn

Pickle

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install dependencies:

pip install pandas numpy scikit-learn nltk matplotlib seaborn

Download NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
📁 Project Structure
detecting_fakenews.ipynb
train.tsv
fake.csv
model.pkl
README.md
🚀 Future Improvements

Try advanced models (Logistic Regression, Random Forest, XGBoost)

Implement Deep Learning models (LSTM, BERT)

Perform hyperparameter tuning

Deploy as a web application using Flask or Streamlit

Use a larger and more diverse dataset

🎯 Conclusion

This project demonstrates how Natural Language Processing and Machine Learning techniques can be combined to detect fake news effectively. It provides a strong foundation for building real-world misinformation detection systems.
