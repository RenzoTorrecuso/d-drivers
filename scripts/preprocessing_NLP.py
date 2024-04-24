# Importing necessary libraries
print('======= This script prepares the data for natural language processing =======')
print('Importing necessary libraries and stopwords...')
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import PowerTransformer

# Download NLTK resources
from nltk import download
download("stopwords")
download("punkt")

# Load German stopwords
stop_words = set(stopwords.words('german'))

#### Reading file #####
print('Reading the input file...')
# Read CSV file into a DataFrame
if os.path.exists('./data/data_nlp_A.csv'):
    df = pd.read_csv('./data/data_nlp_A.csv')
else:
    df = pd.read_csv('./data/data_nlp.csv')  # Reading alternative file

#### Scaling ####
print('Scaling target variables with Power Transformer ...')

# Initialize PowerTransformer
scaler = PowerTransformer()

# Transform target variables
target_columns = ['external_impressions', 'external_clicks', 'ctr']
scaled_columns = [col + '_scaled' for col in target_columns]
df[scaled_columns] = scaler.fit_transform(df[target_columns])

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
    for col in columns:
        df[col] = df[col].apply(remove_stopwords)
    return df

# Columns to clean from stopwords
columns_to_clean = ['h1', 'abstract', 'meta_title', 'meta_description']
df = remove_stopwords_from_columns(df, columns_to_clean)

#### Encoding ####
print('Encoding categorical values ....')

# List of categorical columns to encode
categorical = ['sentiment_abstract', 'sentiment_meta_title', 'video_player_types', 'clickbait_label', 'title_has_colon', 'media_type']

# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical, prefix=categorical, drop_first=True)

#### Saving file ####
print('Saving file ....')

# Save the preprocessed DataFrame to a new CSV file
df_encoded.to_csv('./data/preprocessing_nlp.csv', encoding='utf-8', index=False)
print('File is saved as data/preprocessing_nlp.csv')
print('======== Processing complete ========')