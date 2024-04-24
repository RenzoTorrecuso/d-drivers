import os
import pandas as pd
from transformers import pipeline

# Set Hugging Face token
os.environ["HF_TOKEN"] = "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"

# Load the preprocessed data
data = pd.read_csv('./data/preprocessing_nlp.csv')

# Initialize the text classification pipeline with the specified model
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
data[['label', 'score']] = data['h1'].apply(classify_headline).apply(pd.Series)

# Save the classified data to a new CSV file
data.to_csv('./clickbait.csv', encoding='utf-8', index=False)