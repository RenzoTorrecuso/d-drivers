## For explanations see ./notebooks/Feature_eng_clara.ipynb

import pandas as pd

# Read file:
print('======== This script engineers relevant features based on cleaned file ========')
# print()
print('Reading the file...')

df_perf = pd.read_csv('data/full_data.csv')
df_scrape = pd.read_csv('data/full_scraped.csv')

print('Reading complete. \nCreating the features...')

### Feature Engineering ###

## Target Variable ##

# Calculate the Click through rate based on external clicks and impressions
df_perf['ctr'] = df_perf['external_clicks'] / df_perf['external_impressions'] *100

## Features ##

# Function to extract last part of URL and clean it
def extract_last_part(url):
    url_text = url.rsplit('/', 1)[-1]
    cleaned_url = url_text.split('_')[0]
    cleaned_url_list = cleaned_url.split('-')
    return cleaned_url_list

# Apply the function to create a new column
df_scrape['url_text'] = df_scrape['url'].apply(extract_last_part)

# Sum up all list items per ongoing Version ID and merge with original df
df_feat = pd.merge(df_scrape, df_scrape.groupby('page_id')['url_text'].apply(lambda x: list(set(sum(x, [])))).reset_index(name='merged_url'), on='page_id', how='left')

#Transform media column
def media_type(df, media_type):
    if 'img-wrapper' in media_type or any(item in media_type for item in ['image-gallery', 'mb-lg-7', 'mb-8']):
        return 'img'
    elif any(item in media_type for item in ['mb-3', 'video-player', 'recobar']):
        return 'video'
    else:
        return 'other'

df_feat['media_type'] = df_scrape['media_type'].apply(lambda x: media_type(df_feat, x))

# Title length
df_feat['meta_title_len'] = df_feat['meta_title'].str.len()

# Meta description length
df_feat['meta_desc_len'] = df_feat['meta_description'].str.len()

# H1 length
df_feat['h1_len'] = df_feat['h1'].str.len()

# Abstract length
df_feat['abstract_len'] = df_feat['abstract'].str.len()

# URL length
df_feat['merged_url_len'] = df_feat['merged_url'].str.len()

### Merging ###
col = ['page_id', 'url', 'date', 'version_id', 'publish_date',
       'word_count', 'classification_product', 'classification_type', 'page_name', 'authors','title',
       'daily_likes', 'daily_dislikes', 'video_play', 'page_impressions',
       'clickouts', 'external_clicks', 'external_impressions','ctr']

on = ['url','page_id']

df_merge = pd.merge(df_perf[col],df_feat,how='left',on=on)
#df_merge.drop(['url_x'], axis=1, inplace=True)

df_merge.rename({
    'date_x':'date',
    #'url_y':'url',
    'date_y':'scraped_date'}
    , axis=1, inplace=True)

order = ['page_id','version_id','date','publish_date','url','page_name','authors','title',
         'word_count','classification_product','classification_type',
         'h1','author','scraped_date','abstract','main_text_length','meta_title','meta_description','meta_image_url','media_type','page_img_size','url_text','merged_url',
         'meta_title_len','meta_desc_len','daily_likes','daily_dislikes','video_play',
         'page_impressions','clickouts','external_clicks','external_impressions','ctr']

df_merge = df_merge[order]

### Writing to the file ###
print('Writing the final data frame to file...')
df_merge.to_csv('./data/data_features.csv', encoding='utf-8', index=False)
print('The full dataframe with features is saved as ./data/data_features.csv')

print('======== Processing complete ========')