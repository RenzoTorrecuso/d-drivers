## For explanations see ./notebooks/Cleaning-categorising-katja.ipynb

import pandas as pd
### from timeit import timeit

# Read part of the malformatted file:
print('=== This is the script that combines and tidies up the raw data ===')
print('Reading the first file')

df1 = pd.read_excel('data/data_d-drivers_2024-03-24.xlsx', sheet_name='data',
                    usecols=['PAGE_EFAHRER_ID', 'DATE', 'PAGE_CANONICAL_URL', 'PAGE_AUTHOR', 'WORD_COUNT', 'CLICKOUTS']
                    )
print('Reading the second file')
#Read everything from the new file:
df2 = pd.read_excel('data/data_d-drivers_2024-03-26.xlsx', sheet_name='data')

print('Reading complete. \n Cleaning up the dataframes...')
df1.columns = [col.lower() for col in df1.columns]
df2.columns = [col.lower() for col in df2.columns]

df1.rename({
           #'impressions': 'page_impressions',
           'page_efahrer_id': 'page_id',
           'published_at': 'publish_date',
           'page_canonical_url': 'url',
           'page_author': 'authors', 
            }, axis=1, inplace=True)

df2.rename({
           'impressions': 'page_impressions',
           'page_efahrer_id': 'page_id',
           'published_at': 'publish_date',
           'page_canonical_url': 'url',
           'page_author': 'authors', 
            }, axis=1, inplace=True)

# Eliminate mistakes from the table
df1.drop(78658, inplace=True)
df2.drop(40600, inplace=True)

### Merging ###

# Using the `left` merging: we already know that `df1` is malformatted
key_columns = ['page_id', 'date', 'url', 'authors', 'word_count']

print('Merging...')
df = pd.merge(left=df2, right=df1, on=key_columns, how='left') 

### Imputing ###
print('Imputing...')
df = df.sort_values(['page_id', 'date', 'publish_date', 'url'])\
    .reset_index(drop=False)
df.rename({'index': 'old_index'}, axis=1, inplace=True)

# reshuffling columns
df = df[['old_index', 'page_id', 'date', 'publish_date', 'word_count',
         # 'publish_date_equal_to_date', # we don't need this one anymore (and never needed?)
       'url', 'page_name', 'classification_product', 'classification_type',
       'title', 'authors', 'daily_likes', 'daily_dislikes', 
       'video_play', 'page_impressions', 'clickouts', 
       'external_clicks', 'external_impressions'
        ]]

### Imputing ###
# Filling in missing publishing dates: 
# First, we assume that when the current `date` precedes the published_date
# the article was initially published long ago, where "long ago" is 1. Jan 2018
# and today it has been scheduled for an update
df.loc[df.date < df.publishing_date, 'publishing_date'] = pd.Timestamp('2018-01-01 00:00:00')

# second, we assume that whern there is date of publication at all, 
# the articles were published on 1. Jan 2018 (Roughly 33% of all articles)

versions = df.groupby(['page_id'], as_index=False, sort=True)[
    ['page_id', 'date', 'publishing_date', 'word_count']
    ].ffill()
# which articles don't have the publishing date?
no_publ_date = versions[versions.publishing_date.isna()].page_id.unique()

df.loc[df.page_id.isin(no_publ_date), 'publishing_date'] = pd.Timestamp('2018-01-01 00:00:00')
# Forward-filling the word count and publishing dates for each article.
# !! Assuming that when the word counts do not change unless otherwise specified !! 
versions2 = df.groupby(['page_id'], as_index=False, sort=True)[
    ['page_id', 'date', 'publishing_date', 'word_count']
    ].ffill().drop_duplicates()

# merging the imputed columns back into the df
df_imputed = pd.merge(df[df.columns.drop('publishing_date').drop('word_count')], # drop the non-imputed columns
                      versions2,
                      on=['page_id', 'date'], how='left')

### Including the scraped data ###
# THANKS @CLARA 
df_scraped = pd.read_csv('../data/scraping_no_duplicates.csv')
df_scraped.columns = [col.lower() for col in df_scraped.columns]

df_scraped.rename({
           #'impressions': 'page_impressions',
           'words': 'words_scraped',
           'page_efahrer_id': 'page_id',
           'page_canonical_url': 'url',
           'author': 'author_scraped',
           'current_title': 'h1'
            }, axis=1, inplace=True)

col_to_merge = ['page_id', 'url']
df_full = pd.merge(left=df_imputed, right=df_scraped, on=col_to_merge, how='left')
# May drop some columns
#df_full = df_full.drop(['old_index'], axis=1)

### Writing to the file ###
print('Writing the final data frame to file')
df_full.to_csv('./data/full_data.csv', encoding='utf-8', index=False)
print('The full dataframe is saved as ./data/full_data.csv')

# User side features: those which the reader sees
# user_side_features = ['page_id', 'date', 'publish_date', 'word_count', 'words_scraped', 
#                          'page_title', 'page_name', 'title', 'h1', 'authors',  
#                          'classification_product', 'classification_type']

# [['page_id', 'date', 'publish_date', 'word_count', 'words_scraped', 
                        #  'page_title', 'page_name', 'title', 'h1', 'authors', 'video_play', 
                        #  'classification_product', 'classification_type', 
                        #  'page_impressions', 'clickouts',
                        #  'external_clicks', 'external_impressions']]
#df_eda_user = df_full.groupby(user_side_features)

#df_eda_user.to_csv('./data/data_eda_user-side.csv', encoding='utf-8', index=False)
#print('The dataframe for pages with a few variations is saved as ./data/data_eda_static.csv')

#df_full[['page_id']].to_csv('./data/data_eda_static.csv', encoding='utf-8', index=False)
#print('The dataframe for pages with a few variations is saved as ./data/data_eda_static.csv')


print('Processing complete')
