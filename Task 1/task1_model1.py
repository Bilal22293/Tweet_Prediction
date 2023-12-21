# -*- coding: utf-8 -*-

from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd

df = pd.read_csv("../Test Dataset/behaviour_simulation_test_time.csv")
y_pred = pd.DataFrame(columns=["id","likes"])
uname = pd.read_csv("username_before_1780.csv")
df_var = pd.read_csv("../EDA/username_central_tendencies.csv")

uname = uname.drop(columns ="Unnamed: 0")
dfx = df[df["username"].isin(uname["username"])]

unamey = df_var[df_var["variance"]>=1780]
unamey = unamey["username"]
unamey = pd.DataFrame(unamey)
dfy = df[df["username"].isin(unamey["username"])]
y_pred["id"]=dfx["id"]
y_pred["likes"]=0

for index,row in dfx.iterrows():
  temp = df_var[df_var["username"]==row["username"]]
  if(temp["Type of Data"].values[0]=="Skewed"):
    y_pred["likes"][index] = temp["median"].values[0]
  else:
    y_pred["likes"][index] = temp["mean"].values[0]


import tensorflow as tf

model_path = 'task1_model2'
model = tf.keras.models.load_model(model_path)

df_test = dfy

# Assuming df_test is your DataFrame
df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')

# Create new columns for date and time
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
import re

# Function to remove emojis from a text
def remove_emojis(text):
    return emoji.demojize(text)

# Apply the function to the 'content' column
df_test['content'] = df_test['content'].apply(remove_emojis)

# Remove any remaining emoji-related text (e.g., :smile: )
df_test['content'] = df_test['content'].str.replace(":\w+:", " ")

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Assuming you have already loaded your dataset into a variable 'df'
df_test.dropna(inplace=True)

df_content = df_test['content']

df_content = df_content.astype(str)
df_content = df_content.dropna()
df_content = df_content.apply(word_tokenize)

df_test['tokenized_content'] = df_content

import string

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords_and_punctuations = set(['hyperlink']+['mention']+['<']+['>'])
print(stopwords_and_punctuations)

def remove_english_stopwords_func(text):
    t = [token for token in text if token.lower() not in stopwords_and_punctuations]
    text = ' '.join(t)
    return text

df_test['content_without_stopwords_and_punctuations'] = df_test['tokenized_content'].apply(remove_english_stopwords_func)

df_test

# Get tweets for test data
tweets_test_1 = df_test['username']
tweets_test_2 = df_test['content_without_stopwords_and_punctuations']
tweets_test_4 = df_test['day']
tweets_test_5 = df_test['month']
tweets_test_6 = df_test['year']
tweets_test_7 = df_test['time_only']
tweets_test_8 = df_test['inferred company']

X_test = tweets_test_1 + ' ' + tweets_test_2 + ' ' + tweets_test_4 + tweets_test_5 + tweets_test_6 + ' ' + tweets_test_7 + tweets_test_8

X_test

# Concatenate text data with date and time features


# Convert to TensorFlow dataset
X_test_ds = tf.convert_to_tensor(X_test)

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
features_test = X_test_ds
text_vectorizer.adapt(features_test)

def preprocess(x):
    return text_vectorizer(x)

X_test_ds

# Preprocess test data
X_test_vectorized = preprocess(X_test_ds)

predictions = model.predict(X_test_ds)

