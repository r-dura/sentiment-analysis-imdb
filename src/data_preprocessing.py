import numpy as np
import pandas as pd
import string
import re
from collections import Counter
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

def tokenize_and_clean(text):
    stop_words = set(stopwords.words('english'))
    # Remove the "br" token (which are breaks)
    text = re.sub(r'<br\s*/?>|br', ' ', text)
    tokens = text.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word.lower() in stop_words]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

# Building bag of words for all reviews
def build_full_bow(reviews):
    full_bow = Counter()
    for review in reviews:
        tokens = tokenize_and_clean(review)
        bow = Counter(tokens)
        full_bow.update(bow)
    return full_bow

# Tokenizes reviews and filters out word not present in total bag of words
def preprocess_reviews(reviews, full_bow):
    processed_reviews = []
    for review in reviews:
        tokens = tokenize_and_clean(review)
        filtered_tokens = [word for word in tokens if word in full_bow]
        processed_review = ' '.join(filtered_tokens)
        processed_reviews.append(processed_review)
    return processed_reviews

# Converts the vocabulary to a fixed-length vector with size of vocab (17738) + 1 elements
def prepare_data(df, test_size=0.2, random_state=42):
    full_bow = build_full_bow(df['review'])
    processed_reviews = preprocess_reviews(df['review'], full_bow)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_reviews)
    # encode training data set
    X = tokenizer.texts_to_matrix(processed_reviews, mode='freq')
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test