
# Team-59 Adobe-Mid-Prep: Inter IIT Tech Meet 12.0 !Adobe <img src="assets/adobe_logo.png" height="40">

## Overview 🌟
Welcome to our project repository! Here, we've developed a suite of Python scripts tailored for robust data analysis, preprocessing, and Natural Language Processing (NLP). These scripts are meticulously organized into training, inference, and final execution stages, ensuring a seamless workflow for your data processing needs.

### Project Structure 📁
Our project consists of the following key scripts:

1. **EDA**: The EDA.py file in the EDA folder provides a detailed analysis of user engagement in our dataset, highlighting the top 10 users each month based on likes, creating boxplots for likes distribution per user, and computing key statistics like mean, median, variance, and data symmetry to understand user engagement trends.
2. **Preprocessing**: The preprocessing.py file in the Preprocess folder focuses on refining our dataset for better analysis. It starts by eliminating outliers in the 'likes' for each username, followed by cleansing the 'content' column through the removal of emojis, hashtags, and tokenization. Additionally, custom stop words are identified and removed to streamline the data for subsequent analysis.
3. **Task 1**: This folder contains the required python files for the models being used in task 1 to predict the number of likes
4. **Task 2**: This folder contains the required python files for the models being used in task 2 to predict the tweet content

## Getting Started 🚀
To dive into our project, follow these steps:

1. Clone the Repository:
   ```shell
   git clone https://github.com/TeamNumber59/Adobe-Inter-IIT-Tech-Meet-12.0.git
   ```
2. Run Setup Script
   ```shell
   chmod +x setup.sh
   ./setup.sh
   ```
3. Set Up a Virtual Environment:
   ```shell
   python -m venv venv
   ```
4. Install Dependencies:
   ```shell
   pip install -r requirements.txt
   ```

## Usage Guide 🛠️
- **EDA.py**: Load your dataset to commence exploratory data analysis. Utilize the script to uncover key trends and distribution patterns.
- **Preprocessing.py**: Employ this script to refine your data, managing any missing values and standardizing features.

## License 📄
This project is proudly licensed under the [MIT License](LICENSE).

## Acknowledgements 💖
A heartfelt shoutout to Team 59! Your collaborative efforts and dedication were pivotal in bringing this Inter IIT Project to fruition.
