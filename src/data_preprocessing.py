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
def build_bow(reviews):
    full_bow = Counter()
    for review in reviews:
        tokens = tokenize_and_clean(review)
        bow = Counter(tokens)
        full_bow.update(bow)
    return full_bow

# Tokenizes each review in data set given the total bag of words from dataset
def preprocess_reviews(reviews, bow):
    processed_reviews = []
    for review in reviews:
        tokens = tokenize_and_clean(review)
        filtered_tokens = [word for word in tokens if word in bow]
        processed_review = ' '.join(filtered_tokens)
        processed_reviews.append(processed_review)
    return processed_reviews

def prepare_data(df, test_size=0.2, random_state=42):
    # split data into training and testing
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['sentiment'])

    # Filter bow only using training data to keep tokens with a min occurence greater than 2
    train_bow = build_bow(train_df['review'])
    filtered_bow = {token: train_bow[token] for token in train_bow if train_bow[token] > 10}

    # Process training and testing reviews
    processed_train_reviews = preprocess_reviews(train_df['review'], filtered_bow)
    processed_test_reviews = preprocess_reviews(test_df['review'], filtered_bow)

    # Vectorize the data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_train_reviews)
    
    # encode training and testing data sets
    X_train = tokenizer.texts_to_matrix(processed_train_reviews, mode='freq')
    y_train = train_df['sentiment'].values

    X_test = tokenizer.texts_to_matrix(processed_test_reviews, mode='freq')
    y_test = test_df['sentiment'].values

    return X_train, X_test, y_train, y_test