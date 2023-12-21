# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the pre-trained model
model_path = 'task1_model2'
model = tf.keras.models.load_model(model_path)

# Load the test data
df_test = pd.read_csv("../Test Dataset/behaviour_simulation_test_company.csv")

# Convert 'date' column to datetime format and extract date, month, year, and time features
df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')
df_test['date_only'] = df_test['date'].dt.date
df_test['time_only'] = df_test['date'].dt.time
df_test = df_test.drop(columns=['date'])

df_test['date_only'] = pd.to_datetime(df_test['date_only'])
df_test['day'] = df_test['date_only'].dt.day
df_test['month'] = df_test['date_only'].dt.month
df_test['year'] = df_test['date_only'].dt.year

# Convert to string format
df_test['day'] = df_test['day'].astype(str)
df_test['month'] = df_test['month'].astype(str)
df_test['year'] = df_test['year'].astype(str)
df_test['time_only'] = df_test['time_only'].astype(str)

import emoji

# Function to remove emojis from a text
def remove_emojis(text):
    return emoji.demojize(text)

# Apply the function to the 'content' column
df_test['content'] = df_test['content'].apply(remove_emojis)

# Remove any remaining emoji-related text (e.g., :smile:)
df_test['content'] = df_test['content'].str.replace(":\w+:", " ")

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Tokenize content using nltk
df_content = df_test['content']
df_content = df_content.astype(str)
df_content = df_content.dropna()
df_content = df_content.apply(word_tokenize)

df_test['tokenized_content'] = df_content

import string

nltk.download('stopwords')

from nltk.corpus import stopwords

# Define a set of stopwords and punctuations
stopwords = set(['hyperlink'] + ['mention'] + ['<'] + ['>'])
print(stopwords)

# Function to remove English stopwords
def remove_english_stopwords_func(text):
    t = [token for token in text if token.lower() not in stopwords]
    text = ' '.join(t)
    return text

df_test['content_without_stopwords'] = df_test['tokenized_content'].apply(remove_english_stopwords_func)

# Function to remove hashtags from a text
def remove_hashtags(text):
    return re.sub(r'\# \w+', '', text)

# Apply the function to the 'content' column
df_test['content_without_stopwords_and_hastags'] = df_test['content_without_stopwords'].apply(remove_hashtags)

# Get relevant columns for test data
tweets_test_1 = df_test['username']
tweets_test_2 = df_test['content_without_stopwords_and_hastags']
tweets_test_4 = df_test['day']
tweets_test_5 = df_test['month']
tweets_test_6 = df_test['year']
tweets_test_7 = df_test['time_only']
tweets_test_8 = df_test['inferred company']

# Concatenate text data with date and time features
X_test_text = tweets_test_2 + tweets_test_8
X_test_date_time = df_test[['day', 'month', 'year', 'time_only']]
X_test = pd.concat([X_test_text, X_test_date_time], axis=1)

# Convert to TensorFlow dataset
X_test_text_ds = tf.convert_to_tensor(X_test[0].astype(str))

# Convert date and time features to numerical values
X_test['day'] = X_test['day'].astype(int)
X_test['month'] = X_test['month'].astype(int)
X_test['year'] = X_test['year'].astype(int)
X_test['time_only'] = X_test['time_only'].apply(lambda x: int(x.replace(':', '')))

X_test_date_time_ds = tf.convert_to_tensor(X_test[['day', 'month', 'year', 'time_only']].values)

# Vectorize phrases
max_features = 10000
sequence_length = 250

text_vectorizer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=max_features,
    output_mode='int',
    vocabulary=None,
    output_sequence_length=sequence_length)

# Vectorize test data
features_test = X_test_text_ds
text_vectorizer.adapt(features_test)

# Preprocess test data
def preprocess(x):
    return text_vectorizer(x)

X_test_vectorized = preprocess(X_test_text_ds)

# Make predictions using the loaded model
predictions = model.predict([X_test_text_ds, X_test_date_time_ds])
predictions = predictions.flatten()