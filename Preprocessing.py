# Import necessary libraries
import os
import numpy as np
import pandas as pd
from types import new_class
from scipy.stats import skew

# Change directory to the location of the dataset
os.chdir('/Train Data/')

"""##Sorting the Main Dataset by Username"""

df2=pd.read_csv('behaviour_simulation_train.csv')

# Sort the DataFrame by the 'username' column
df_sorted = df2.sort_values(by='username')

# Define the path where you want to save the new CSV file
filepath = '/Preprocess/sorted_by_username.csv'

# Save the sorted DataFrame as a new CSV file
df_sorted.to_csv(filepath, index=False)

"""##Removing Outliers from the Dataset"""

df1=pd.read_csv('/Preprocess/sorted_by_username.csv')

# Define a function 'remove_outliers' to remove outliers from a group
def remove_outliers(group):
    # Calculate the first quartile (q1) of 'likes' in the group
    q1 = group['likes'].quantile(0.25)

    # Calculate the third quartile (q3) of 'likes' in the group
    q3 = group['likes'].quantile(0.75)

    # Calculate the interquartile range (IQR) of 'likes' in the group
    iqr = q3 - q1

    # Calculate the lower bound for outlier detection
    lower_bound = q1 - 1.5 * iqr

    # Calculate the upper bound for outlier detection
    upper_bound = q3 + 1.5 * iqr

    # Return a filtered group containing data points within the bounds
    return group[(group['likes'] >= lower_bound) & (group['likes'] <= upper_bound)]

# Group the DataFrame 'df' by 'username' and apply the 'remove_outliers' function to each group
filtered_groups = df1.groupby('username', group_keys=False).apply(remove_outliers)

# Reset the index of the filtered DataFrame
filtered_df = filtered_groups.reset_index(drop=True)

# Get the length of the filtered DataFrame
len(filtered_df)

filtered_df

filepath = "/Preprocess/behaviour_simulation_train_preprocess.csv"
filtered_df.to_csv(filepath,index=False)

"""##Removing Stopwords, Hastags, Emojis and NAN values from our dataset"""

# Read the dataset into a Pandas DataFrame
df = pd.read_csv('/Preprocess/behaviour_simulation_train_preprocess.csv')

# Display the DataFrame
df

# Check for missing values in the DataFrame
df.isna().sum()

# Drop rows with missing values
df.dropna(inplace=True)

df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Create new columns for date and time
df['date_only'] = df['date'].dt.date
df['time_only'] = df['date'].dt.time
df = df.drop(columns=['date'])

# Install the 'emoji' library
!pip install emoji

# Import necessary libraries for working with emojis
import emoji
import re

# Function to remove emojis from a text
def remove_emojis(text):
    text =  emoji.demojize(text)

    # Remove any remaining emoji-related text using a regular expression
    text = re.sub(r':\w+(-\w+)*:', ' ', text)

    return text

# Apply the function to the 'content' column
df['content'] = df['content'].apply(remove_emojis)

# Import necessary libraries for text processing
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Assuming you have already loaded your dataset into a variable 'df'
df.dropna(inplace=True)

# Tokenize the 'content' column
df_content = df['content']
df_content = df_content.astype(str)
df_content = df_content.dropna()
df_content = df_content.apply(word_tokenize)

# Create a new column 'tokenized_content'
df['tokenized_content'] = df_content

# Import necessary libraries for working with stopwords
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

# Define custom stopwords
stopwords = set(['hyperlink']+['<']+['>']+['mention'])

# Function to remove stopwords from a list of tokens
def remove_stopwords_func(text):
    t = [token for token in text if token.lower() not in stopwords]
    text = ' '.join(t)
    return text

# Apply the function to the 'tokenized_content' column
df['content_without_stopwords'] = df['tokenized_content'].apply(remove_stopwords_func)

# Function to remove hashtags from a text
def remove_hashtags(text):
    return re.sub(r'\# \w+', '', text)

# Apply the function to the 'content' column
df['content_without_stopwords_and_hastags'] = df['content_without_stopwords'].apply(remove_hashtags)

# Display the DataFrame
df.drop(columns=["content_without_stopwords"],inplace=True)

filepath = "/Preproess/preprocess_final_behaviour_dataset.csv"
df.to_csv(filepath,index=False)

