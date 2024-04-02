## For explanations see ./notebooks/Cleaning-categorising-katja.ipynb

import pandas as pd
### from timeit import timeit

# Read part of the malformatted file:
print('=== This is the script that combines and tidies up the raw data ===')
print('The run should take approx. 30 seconds.')
print()
print('Reading the first file...')

df1 = pd.read_excel('data/data_d-drivers_2024-03-24.xlsx', sheet_name='data',
                    usecols=['PAGE_EFAHRER_ID', 'DATE', 'PAGE_CANONICAL_URL', 'PAGE_AUTHOR', 'WORD_COUNT', 'CLICKOUTS']
                    )
print('Reading the second file...')
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
# First, we assume that when the current `date` precedes the publish_date
# the article was initially published long ago, where "long ago" is 1. Jan 2018
# and today it has been scheduled for an update
df.loc[df.date < df.publish_date, 'publish_date'] = pd.Timestamp('2018-01-01 00:00:00')

# second, we assume that whern there is date of publication at all, 
# the articles were published on 1. Jan 2018 (Roughly 33% of all articles)

versions = df.groupby(['page_id'], as_index=False, sort=True)[
    ['page_id', 'date', 'publish_date', 'word_count']
    ].ffill()
# which articles don't have the publishing date?
no_publ_date = versions[versions.publish_date.isna()].page_id.unique()

df.loc[df.page_id.isin(no_publ_date), 'publish_date'] = pd.Timestamp('2018-01-01 00:00:00')
# Forward-filling the word count and publishing dates for each article.
# !! Assuming that when the word counts do not change unless otherwise specified !! 
versions2 = df.groupby(['page_id'], as_index=False, sort=True)[
    ['page_id', 'date', 'publish_date', 'word_count']
    ].ffill().drop_duplicates()

# merging the imputed columns back into the df
df_imputed = pd.merge(df[df.columns.drop('publish_date').drop('word_count')], # drop the non-imputed columns
                      versions2,
                      on=['page_id', 'date'], how='left')

df_imputed = df_imputed.sort_values(['page_id', 'date', 'publish_date']) # just in case, should be already sorted

df_imputed['word_count'] = df_imputed.groupby(['page_id', 'date'])['word_count'].ffill()

# Impute the still missing word counts with 0
# -> In the future: take the value of the `word count (scraped)` 
# or the mean value for the given category! 
df_imputed['word_count'] = df_imputed['word_count'].fillna(0)

df_imputed['publish_date'] = df_imputed.groupby(['page_id', 'date'])['publish_date'].ffill()
df_imputed['publish_date'] = df_imputed.groupby(['page_id'])['publish_date'].ffill()

### Version count ###
print('''Calculating version IDs
      Hint: version changes when any of the folowing change: word count, publish_date or the authors''')

temp = df_imputed[['page_id', 'word_count', 'publish_date', 'authors']].drop_duplicates()
temp = temp.fillna({'word_count': 0, 'publish_date': pd.Timestamp('2018-01-01 00:00')})
temp = temp.drop_duplicates()
temp = temp.sort_values('publish_date')

wc_versions = temp.groupby('page_id')['word_count'].transform(lambda x: pd.factorize(x)[0])
publish_versions = temp.groupby('page_id')['publish_date'].transform(lambda x: pd.factorize(x)[0])
authors_versions = temp.groupby('page_id')['authors'].transform(lambda x: pd.factorize(x)[0])

version_count = 10000*wc_versions + 1000*publish_versions + 1*authors_versions
temp['ver_id_wc'] = wc_versions
temp['ver_id_pub'] = publish_versions
temp['ver_id_auth'] = authors_versions
temp['version_id_raw'] = version_count
temp['version_id'] = temp.groupby('page_id')['version_id_raw'].transform(lambda x: pd.factorize(x)[0])

df_imputed = pd.merge(df_imputed, temp.drop(['ver_id_wc', 'ver_id_pub', 'ver_id_auth', 'version_id_raw'], axis=1),
         on=['page_id', 'word_count', 'publish_date', 'authors'],
         how='left')

### Including the scraped data ###
# THANKS @CLARA 
print('Merging with the scraped data...')

df_scraped = pd.read_csv('data/scraping_no_duplicates.csv')
df_scraped.columns = [col.lower() for col in df_scraped.columns]

df_scraped.rename({
           #'impressions': 'page_impressions',
           'words': 'words_scraped',
           'page_efahrer_id': 'page_id',
           'page_canonical_url': 'url',
           'author': 'author_scraped',
           'current_title': 'h1'
            }, axis=1, inplace=True)

merge_keys = ['page_id', 'url']
df_full = pd.merge(left=df_imputed, right=df_scraped, on=merge_keys, how='left')
# May drop some columns
#df_full = df_full.drop(['old_index'], axis=1)

### Final touch
df_full = df_full[['old_index', 'page_id', 'date', 'url', 'version_id', 'publish_date',
       'word_count', 'words_scraped', 'classification_product', 'classification_type', 
       'page_name', 'authors', 'author_scraped',
       'title', 'h1', 'abstract', 'last_update', 'image_url',
       'daily_likes', 'daily_dislikes', 'video_play', 'page_impressions',
       'clickouts', 'external_clicks', 'external_impressions']]

### Writing to the file ###
print('Writing the final data frame to file')
df_full.to_csv('./data/full_data.csv', encoding='utf-8', index=False)
print('The full dataframe is saved as ./data/full_data.csv')

print('Processing complete')
