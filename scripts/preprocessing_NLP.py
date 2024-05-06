# Importing necessary libraries
print('======= This script prepares the data for natural language processing =======')
print('Importing necessary libraries and stopwords...')
import pandas as pd
import os
from sklearn.preprocessing import PowerTransformer

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