import pandas as pd
from transformers import pipeline
from PIL import UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rake_nltk import Rake
import nltk
import gdown

nltk.download('stopwords')
nltk.download('punkt')

#Loading image to text pre-trained model
captioner = pipeline("image-to-text",model="Salesforce/blip2-opt-2.7b", device=0)

gdown.download('https://drive.google.com/file/d/1jtkeNyvka2dkC8b12s1NMqCVXTF9uWg4', 'content_simulation_train.csv', quiet=True)
df=pd.read_csv('content_simulation_train.csv')

#Caption based om thumbnails and image urls
for index, row in df.iterrows():
    if 'Video' in row['media'] or 'Gif' in row['media']:
        url1 = row['media']
        image_url1 = url1
        start_index1 = image_url1.find("thumbnailUrl='")
        if start_index1 != -1:
            end_index1 = image_url1.find("'", start_index1 + len("thumbnailUrl='"))
            full_url1 = image_url1[start_index1 + len("thumbnailUrl='"):end_index1]
            try:
                text_image1 = captioner(full_url1)[0]
                temp_str1 = list(text_image1.values())
                df.at[index,'imagetotext'] =temp_str1[0]
            except UnidentifiedImageError as e:
                print(f"UnidentifiedImageError: {e} for {full_url1}")
                # Perform error handling or set a default value for df['imagetotext'] here if needed
            except Exception as ex:
                print(f"An error occurred: {ex} for {full_url1}")
                # Perform additional error handling for other exceptions if needed
    elif 'Photo' in row['media']:
        url1 = row['media']
        image_url1 = url1
        start_index1 = image_url1.find("fullUrl='")
        if start_index1 != -1:
            end_index1 = image_url1.find("'", start_index1 + len("fullUrl='"))
            full_url1 = image_url1[start_index1 + len("fullUrl='"):end_index1]
            try:
                text_image1 = captioner(full_url1)[0]
                temp_str1 = list(text_image1.values())
                df.at[index,'imagetotext'] =temp_str1[0]
            except UnidentifiedImageError as e:
                print(f"UnidentifiedImageError: {e} for {full_url1}")
                # Perform error handling or set a default value for df['imagetotext'] here if needed
            except Exception as ex:
                print(f"An error occurred: {ex} for {full_url1}")
                # Perform additional error handling for other exceptions if needed



# Save the updated CSV file with word embeddings
df.to_csv('image_to_text1.csv', index=False)

df=pd.read_csv('image_to_text1.csv')

#Extracting keywords from the caption generated using RAKE
def extract_keywords_rake(text):
    # Create a Rake object
    rake = Rake()

    # Extract keywords from the text
    rake.extract_keywords_from_text(text)

    # Get the ranked keywords
    keywords = rake.get_ranked_phrases()

    return keywords
keywords=[]
for i in range(len(df)):
  text=df['imagetotext'].tolist()[i]
  keywords.append(extract_keywords_rake(text))


df['keywords']=keywords



# Step 1: Ensure GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 2: Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
model = AutoModelForCausalLM.from_pretrained('stabilityai/stablelm-zephyr-3b').to(device)


pred=[]

like=df['likes'].tolist()
user=df['user'].tolist()


for i in range(len(keywords)):
    data = {
        "Keywords":keywords[i],
        "Likes": like[i],
        "Username": user[i]
    }

    # System prompt
    sys_prompt = f"""
    You are a tweet generating bot which generates catchy tweets based on the given keywords:
    You will have the following data to generate the tweet.
    1) Keywords
    2) Number of likes in the tweet
    3) Username in the tweet

    Do not specify the number of likes in the tweet and make it as unique as possible. Also use relevant emojis and hashtags. The tweet length should be between 20 to 80.
    Generate a tweet around the specified keywords.
    And I want you to write the generted tweet within quotes.
    Do not give me any steps, and do not give me any explanation of how you wrote the tweet.
    Do not give me how you wrote the tweet and why you wrote what you wrote.
    Give me only the tweet that you wrote as output, nothing else.

    Here is your data: {data}

    """

    prompt = [{'role': 'user', 'content': sys_prompt}]

    # Step 3: Prepare input data
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(device)

    # Step 4: Model inference
    tokens = model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.8,
        do_sample=True
    )

    txt=tokenizer.decode(tokens[0], skip_special_tokens=True)
    pred.append(txt)

def selection(s):
    r=s.rfind('<|assistant|>')
    x=s.rfind('<|endoftext|>')
    return (s[r+len('<|assistant|>'):x])

'''
!!!removing quotes!!!
'''
def remove_quotes(input_string):
    return input_string.strip('"\'')

pred1=[]
for pp in pred:
  pred1.append(selection(pp))
pred2=[]
for pp in pred1:
  pred2.append(remove_quotes(pp))
#Save predictions
df['predictions']=pred2