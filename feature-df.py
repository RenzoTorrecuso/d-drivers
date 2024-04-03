## For explanations see ./notebooks/Feature_eng_clara.ipynb

import pandas as pd
from timeit import timeit

# Read file:
print('======== This script engineers relevant features based on cleaned file ========')
# print('The run should take approx. 30 seconds.')
# print()
print('Reading the file...')

df = pd.read_csv('data/full_data.csv')

print('Reading complete. \nCreating the features...')

### Feature Engineering ###

## Target Variable
# Calculate the Click through rate based on external clicks and impressions
df['ctr'] = df['external_clicks'] / df['external_impressions'] *100

## Features
# Function to extract last part of URL
def extract_last_part(url):
    return url.rsplit('/', 1)[-1]

# Apply the function to create a new column
df.loc[:, 'url_text'] = df['url'].apply(extract_last_part)

# # URL keywords into list 
# tbd

# # drop duplicates
# tbd

# Title length
df['title_len'] = df['title'].str.len()

# H1 length
df['h1_len'] = df['h1'].str.len()

# Abstract length
df['abstract_len'] = df['abstract'].str.len()

# URL length
df['url_len'] = df['url_text'].str.len()

## Does the page contain a video or image
#df['media_type'] = 

# One hot encode category & type
#tbd

### Final Touch ###
# df = df['version_id_ong', 'page_id', 'publish_date', 'date', 'last_update',
#              'url_text', 'page_name', 'title', 'h1', 'abstract', 
#              'classification_product', 'classification_type', 
#              'authors', 'author_scraped',
#              'image_url', 'word_count', 'words_scraped', 
#              'daily_likes', 'daily_dislikes', 'video_play', 'page_impressions', 'clickouts', 
#              'external_clicks', 'external_impressions', 'ctr'
#              ]

### Writing to the file ###
print('Writing the final data frame to file...')
df.to_csv('./data/data_features.csv', encoding='utf-8', index=False)
print('The full dataframe with features is saved as ./data/data_features.csv')

print('======== Processing complete ========')