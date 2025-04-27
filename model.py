import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings

from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import regexp_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


warnings.filterwarnings('ignore')

''' reading data '''
df = pd.read_csv(r"C:\Users\ravi teja gundami\OneDrive\Desktop\IMDB Dataset.csv")

''' dropping duplicate rows '''
df.drop_duplicates(inplace=True)

''' converting every word into lower '''
def lower_word(word):
    return word.lower()

df['review'] = df['review'].map(lower_word)

def regex_(raw_text):
    find_html = re.compile('<.*?>')
    clean_text = re.sub(find_html, '', raw_text)
    return clean_text

''' apply regex_ in review '''
df.review = df.review.apply(lambda x: regex_(x))

''' Running WhiteSpace tokenizer  '''
w_token = WordPunctTokenizer()
df["review_tokenized"] = [w_token.tokenize(t) for t in df["review"]]

'''Define POS tags '''
tag_map = defaultdict(lambda : wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV

''' Stopwords removal & WordNet lemmatization '''

for idx, t in enumerate(df.review_tokenized):
    if idx % 100 == 0:
        print(idx)

    word_ls = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(t):
        if word not in stopwords.words("english") and word.isalpha():
            word_p = wordnet_lemmatizer.lemmatize(word, tag_map[tag[0]])
            word_ls.append(word_p)
    df.loc[idx, "review_tokenized_cleaned"] = str(word_ls)

''' dropping null values '''
df.dropna(inplace=True)

''' train test split '''
X_train, X_test, y_train, y_test = train_test_split(df['review_tokenized_cleaned'], df['sentiment'], test_size=0.25, random_state=0)

''' LabelEncoding '''
enc = LabelEncoder()
y_train = enc.fit_transform(y_train)
y_test = enc.transform(y_test)

''' TFIDF vector '''
tfidf = TfidfVectorizer(max_features = 5000)

''' fit on data '''
tfidf.fit(df.review_tokenized_cleaned)

''' fit on X_train '''
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

''' SVM '''
svm_model = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")

''' fit on data '''
svm_model.fit(X_train, y_train)

''' prediction '''
pred_lg= svm_model.predict(X_test)

print("Accuracy Score: ", accuracy_score(y_test, pred_lg))

print("Confusion Matrix: ", confusion_matrix(y_test, pred_lg))

print("Classification Report: \n", classification_report(y_test, pred_lg))

pickle.dump(svm_model,open('svm_model.pkl',"wb"))
pickle.dump(tfidf,open('tfidf_vectorizer.pkl',"wb"))
