
# Team-59 Adobe-Mid-Prep: Inter IIT Tech Meet 12.0 <img src="assets/adobe_logo.png" height="40">

## Overview 🌟
Welcome to our project repository! Here, we've developed a suite of Python scripts tailored for robust data analysis, preprocessing, and Natural Language Processing (NLP). These scripts are meticulously organized into training, inference, and final execution stages, ensuring a seamless workflow for your data processing needs.

## Repository Directory Structure

```
README
EDA/
    - Boxplots
    - Top_10_username_plots
    - EDA.py
    - CSV's
Pre-processing/
    - preprocessing.py
Task 1/
    - python notebooks of models
    - required datasets
Task 2/
    - python notebooks of models
    - required datasets
Test Dataset/
    - required datasets
team_59_results
    - result datasets
assests/
    - adobe_logo.png
```

### Project Structure 📁
Our project consists of the following key scripts:

1. **EDA**: The EDA.py file in the EDA folder provides a detailed analysis of user engagement in our dataset, highlighting the top 10 users each month based on likes, creating boxplots for likes distribution per user, and computing key statistics like mean, median, variance, and data symmetry to understand user engagement trends. All of the boxplots are contained within the "Boxplot" folder. The csv's calculated 
2. **Preprocessing**: The preprocessing.py file in the Preprocess folder focuses on refining our dataset for better analysis. It starts by eliminating outliers in the 'likes' for each username, followed by cleansing the 'content' column through the removal of emojis, hashtags, and tokenization. Additionally, custom stop words are identified and removed to streamline the data for subsequent analysis. The date and time is also split as date_only and time_only.
3. **Task 1**: This folder contains the required python files for the models being used in task 1 to predict the number of likes
4. **Task 2**: This folder contains the required python files for the models being used in task 2 to predict the tweet content
5. **team_59_results 2**: Contains results


## Acknowledgements 💖
A heartfelt shoutout to Team 59! Your collaborative efforts and dedication were pivotal in bringing this Inter IIT Project to fruition.
