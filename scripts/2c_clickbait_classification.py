print('======= This script classifies articles as clickbait =======')
print('Importing necessary libraries and stopwords...')
import os
import pandas as pd
from transformers import pipeline

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm

# Download NLTK resources
from nltk import download
download("stopwords")
download("punkt")

# Load German stopwords
stop_words = set(stopwords.words('german'))

# Set Hugging Face token
os.environ["HF_TOKEN"] = "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"

# Load the preprocessed data
df = pd.read_csv('./data/data_scraped.csv')

#### Removing Stopwords ####
print('Removing stopwords from text features ...')

def remove_stopwords(text):
    """
    Remove stopwords from a given text.

    Parameters:
    text (str): Input text containing words to be processed.

    Returns:
    str: Processed text with stopwords removed.
    """
    if isinstance(text, str):        
        words = word_tokenize(text)
        # Remove punctuation and special characters
        words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
        # Remove stopwords
        words = [word.lower() for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    else:
        return text

def remove_stopwords_from_columns(df, columns):
    """
    Remove stopwords from specified columns in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing text columns to be processed.
    columns (list): List of column names in the DataFrame to process.

    Returns:
    DataFrame: DataFrame with specified columns processed to remove stopwords.
    """
    for col in tqdm(columns, desc="Removing stopwords"):
        df[col] = df[col].apply(remove_stopwords)
    return df

# Columns to clean from stopwords
columns_to_clean = ['h1', 'abstract', 'meta_title', 'meta_description']
df = remove_stopwords_from_columns(df, columns_to_clean)

### Text classification ###

data = df

# Initialize the text classification pipeline with the specified model
print('Apply clickbait classifier ...')

pipe = pipeline("text-classification", model="Stremie/roberta-base-clickbait")

def classify_headline(headline):
    """
    Classify a headline using the pre-trained model.

    Parameters:
    headline (str): The headline text to classify.

    Returns:
    tuple: A tuple containing the predicted label and score.
    """
    result = pipe(headline)[0]
    label = result['label']
    score = result['score']
    return label, score

# Apply the classification function to the 'h1' column
tqdm.pandas(desc="Classifying headlines")
data[['label', 'score']] = data['h1'].progress_apply(classify_headline).apply(pd.Series)

# Save the classified data to a new CSV file
data.to_csv('./data/clickbait.csv', encoding='utf-8', index=False)
print('File is saved as data/clickbait.csv')
print('======== Processing complete ========')