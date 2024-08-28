import numpy as np
import pandas as pd
import string
import re
from collections import Counter
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer


class TextPreprocessor:
    def __init__(self, min_occurrence):
        self.min_occurrence = min_occurrence
        self.stop_words = set(stopwords.words('english'))
        self.filtered_bow = None
        self.tokenizer = None
        
    def tokenize_and_clean(self, text):
        # Remove the "br" token (which are breaks)
        text = re.sub(r'<br\s*/?>|br', ' ', text)
        tokens = text.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [word.translate(table) for word in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if not word.lower() in self.stop_words]
        tokens = [word.lower() for word in tokens if len(word) > 1]
        return tokens

    # Building bag of words for all reviews
    def build_bow(self, reviews):
        bow = Counter()
        for review in reviews:
            tokens = self.tokenize_and_clean(review)
            temp_bow = Counter(tokens)
            bow.update(temp_bow)
        return bow

    # Tokenizes each review in data set given the total bag of words from dataset and filters
    def preprocess_reviews(self, reviews):
        processed_reviews = []
        for review in reviews:
            tokens = self.tokenize_and_clean(review)
            filtered_tokens = [word for word in tokens if word in self.filtered_bow]
            processed_review = ' '.join(filtered_tokens)
            processed_reviews.append(processed_review)
        return processed_reviews

    def filter_bow(self, bow):
        filtered_bow = {token: bow[token] for token in bow if bow[token] > self.min_occurrence}
        return filtered_bow

    def fit(self, reviews):
        train_bow = self.build_bow(reviews)
        self.filtered_bow = self.filter_bow(train_bow)
        processed_reviews = self.preprocess_reviews(reviews)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(processed_reviews)

    def transform(self, reviews):
        processed_reviews = self.preprocess_reviews(reviews)
        return self.tokenizer.texts_to_matrix(processed_reviews, mode='freq')

# def prepare_data(df, test_size=0.2, min_occurance = 2):
#     # split data into training and testing
#     train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentiment'])

#     # Filter bow only using training data to keep tokens with a min occurence greater than 2
#     train_bow = build_bow(train_df['review'])
#     filtered_bow = {token: train_bow[token] for token in train_bow if train_bow[token] > min_occurance}

#     # Tokenize training and testing reviews and filter by training bag of words
#     processed_train_reviews = preprocess_reviews(train_df['review'], filtered_bow)
#     processed_test_reviews = preprocess_reviews(test_df['review'], filtered_bow)

#     # Vectorize the data
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(processed_train_reviews)
    
#     # encode training and testing data sets
#     X_train = tokenizer.texts_to_matrix(processed_train_reviews, mode='freq')
#     y_train = train_df['sentiment'].values

#     X_test = tokenizer.texts_to_matrix(processed_test_reviews, mode='freq')
#     y_test = test_df['sentiment'].values

#     # X_train.to_csv('data/processed/train_processed.csv', index=False)
#     # X_test.to_csv('data/processed/test_processed.csv', index=False)

#     return X_train, X_test, y_train, y_test