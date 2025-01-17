
import pandas as pd
df=pd.read_csv('../Preprocess/preprocess_final_behaviour_dataset.csv')

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df
# Convert the 'date' column to datetime format
df['date_only'] = pd.to_datetime(df['date_only'])

# Create a new column 'month-year' with the desired format as string for better readability
df['month-year'] = df['date_only'].dt.strftime('%Y-%m')

# Group the data by 'month-year' and 'username' to count the total likes for each user in each month
monthly_likes = df.groupby(['month-year', 'username'])['likes'].sum().reset_index()

# Get unique months for subplots
unique_months = monthly_likes['month-year'].unique()

# Determine the number of rows and columns for subplots
num_rows = (len(unique_months) + 1) // 2
num_cols = 2

# Adjust the size of the figure (you might need to tweak these numbers based on your screen resolution and actual data)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))  # Adjust the figure size here
axes = axes.flatten()

# Increase the overall font size for readability
plt.rcParams.update({'font.size': 10})  # Adjust the font size here

# Iterate through unique months and create subplots
for i, month in enumerate(unique_months):
    ax = axes[i]
    group = monthly_likes[monthly_likes['month-year'] == month]
    top_10_users = group.sort_values(by='likes', ascending=False).head(10)
    x = top_10_users['username']
    y = top_10_users['likes']

    # You might want to adjust the bar width here if necessary
    ax.bar(x, y, width=0.5)  # Adjust the width of the bars
    ax.set_xlabel('Username')
    ax.set_ylabel('Total Likes')
    ax.set_title(f'Top 10 Users in {month}')
    ax.set_xticklabels(x, rotation=90, ha='right')  # Rotate the labels for better fit

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Remove empty subplots if necessary
for i in range(len(unique_months), num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout(pad=3.0)  # Adjust the padding here
plt.show()

"""# Boxplots of 'Likes' by Usernames (First 816 Users)"""


# Get the 'username' column from the DataFrame 'df' and convert it to a list
user = df['username'].tolist()

# Create an empty list 'username' to store unique usernames
username = []

# Loop through each username in the 'user' list
for u in user:
    # Check if the username 'u' is not already in the 'username' list
    if not u in username:
        # If 'u' is not in 'username', add it to the 'username' list
        username.append(u)

username=username[:816] #taking first 816 rows dataset

# Create a subplots grid with 136 rows and 6 columns, setting the figure size
fig, ax = plt.subplots(136, 6, figsize=(20, 454))

# Initialize an index counter
i = 0

# Iterate through the subplots grid
for x in ax:
    for y in x:
        # Create a boxplot for the 'likes' column of the DataFrame filtered by the current username
        y.boxplot(df[df['username'] == username[i]]['likes'].tolist())

        # Set the x-axis label to the current username
        y.set_xlabel(username[i])

        # Increment the index counter
        i += 1

# Save the figure as 'boxplot1.png'
fig.savefig('boxplot1.png')

# Display the plot
plt.show()

"""# Boxplots of 'Likes' by Usernames (Second Batch of 816 Users)"""

# Import necessary libraries
import matplotlib.pyplot as plt

# Get the 'username' column from the DataFrame 'df' and convert it to a list
user = df['username'].tolist()

# Create an empty list 'username' to store unique usernames
username = []

# Loop through each username in the 'user' list to extract unique usernames
for u in user:
    if not u in username:
        username.append(u)

# Select the second batch of 816 unique usernames (if there are at least 1632 unique usernames)
username = username[816:1632]

# Create a subplots grid with 136 rows and 6 columns, setting the figure size
fig, ax = plt.subplots(136, 6, figsize=(20, 454))

# Initialize an index counter
i = 0

# Iterate through the subplots grid
for x in ax:
    for y in x:
        # Create a boxplot for the 'likes' column of the DataFrame filtered by the current username
        y.boxplot(df[df['username'] == username[i]]['likes'].tolist())

        # Set the x-axis label to the current username
        y.set_xlabel(username[i])

        # Increment the index counter
        i += 1

# Save the figure as 'boxplot2.png'
fig.savefig('boxplot2.png')

# Display the plot
plt.show()

"""# Boxplots of 'Likes' by Usernames (Remaining Users)"""

# Import necessary libraries
import matplotlib.pyplot as plt

# Get the 'username' column from the DataFrame 'df' and convert it to a list
user = df['username'].tolist()

# Create an empty list 'username' to store unique usernames
username = []

# Loop through each username in the 'user' list to extract unique usernames
for u in user:
    if not u in username:
        username.append(u)

# Select the remaining unique usernames (from index 1632 to the end)
usern = username[1632:]

# Create a subplots grid with 136 rows and 6 columns, setting the figure size
fig, ax = plt.subplots(136, 6, figsize=(20, 454))

# Initialize an index counter
i = 0

# Iterate through the subplots grid
for x in ax:
    for y in x:
        # Create a boxplot for the 'likes' column of the DataFrame filtered by the current username
        y.boxplot(df[df['username'] == usern[i]]['likes'].tolist())

        # Set the x-axis label to the current username
        y.set_xlabel(usern[i])

        # Increment the index counter
        i += 1

# Save the figure as 'boxplot3.png'
fig.savefig('boxplot3.png')

# Display the plot
plt.show()

filtered_df['date'] = pd.to_datetime(filtered_df['date'])

filtered_df['month'] = filtered_df['date'].dt.month

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# Filter only the numeric columns in filtered_df
numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns

# Exclude the 'id' column if it's present
if 'id' in numeric_columns:
    numeric_columns = numeric_columns.drop('id')

cols_per_row = 4

n_rows = (len(numeric_columns) + cols_per_row - 1) // cols_per_row

plt.figure(figsize=(cols_per_row * 4, n_rows * 4))

for i, column in enumerate(numeric_columns):
    plt.subplot(n_rows, cols_per_row, i + 1)
    sb.histplot(filtered_df[column], kde=True)
    plt.title(f"Univariate plot of {column}")
    plt.legend({column})

plt.tight_layout()
plt.show()

# Import necessary libraries
from scipy.stats import skew

# Initialize empty lists to store computed statistics
mean_likes = []
num_tweets = []
median_likes = []
var_likes =[]
skv = []

# Iterate through each username in the 'username' list
for u in username:
    # Filter the 'filtered_df' DataFrame to get 'likes' for the current username and convert it to a NumPy array
    likes = filtered_df[filtered_df['username'] == u]['likes'].tolist()
    likes = np.array(likes)

    # Compute mean and median of 'likes' for the current username
    mean = np.mean(likes)
    median = np.median(likes)
    vari = np.var(likes)

    # Calculate the number of tweets for the current username
    n_tweet = len(filtered_df[filtered_df['username'] == u])

    # Calculate the skewness of 'likes' for the current username
    sk_val = skew(likes)

    # Determine if the data is skewed or symmetric based on skewness value
    if sk_val > 0.5 or sk_val < -0.5:
        sk = 'Skewed'
    else:
        sk = 'Symmetric'

    # Append computed values to respective lists
    mean_likes.append(mean)
    median_likes.append(median)
    var_likes.append(vari)
    num_tweets.append(n_tweet)
    skv.append(sk)

# Create a dictionary with computed statistics
data_dict = {'username': username, 'mean': mean_likes, 'median': median_likes,'variance':var_likes, 'number of tweets': num_tweets, 'Type of Data': skv}

# Create a DataFrame from the dictionary
filtered_dff = pd.DataFrame(data_dict)

# Save the DataFrame to a CSV file
filtered_dff.to_csv("eda_sorted_username.csv")

