from flask import Flask, render_template, request, jsonify
import re
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

app = Flask(__name__)

# Load your trained SVM model and TFIDF vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

tag_map = defaultdict(lambda : wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV

def preprocess_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    words = word_tokenize(text)
    wordnet_lemmatizer = WordNetLemmatizer()
    cleaned_words = [wordnet_lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(words) if word not in stopwords.words("english") and word.isalpha()]
    return ' '.join(cleaned_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    sentiment= None
    review = None
    if request.method == 'POST':
        review = request.form['text']

        # Preprocess the input text
        cleaned_review = preprocess_text(review)

        # Vectorize the cleaned review
        review_vectorized = tfidf_vectorizer.transform([cleaned_review])

        # Make a prediction using the SVM model
        prediction = svm_model.predict(review_vectorized)

        sentiment = "positive" if prediction[0] == 1 else "negative"

        return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
