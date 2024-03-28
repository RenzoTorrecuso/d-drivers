## For explanations see /notebooks/Cleaning-categorising-katja.ipynb

import pandas as pd

# Read part of the malformatted file:
df1 = pd.read_excel('../data/data_d-drivers_2024-03-24.xlsx', sheet_name='data',
                    use_cols=['PAGE_EFAHRER_ID', 'DATE', 'PAGE_CANONICAL_URL', 'PAGE_AUTHOR', 'WORD_COUNT', 'CLICKOUTS']
                    )
#Read everything from the new file:
df2 = pd.read_excel('../data/data_d-drivers_2024-03-26.xlsx', sheet_name='data')

df1.columns = [col.lower() for col in df1.columns]
df2.columns = [col.lower() for col in df2.columns]

df1.rename({
           #'impressions': 'page_impressions',
           'page_efahrer_id': 'page_id',
           'published_at': 'publishing_date',
           'page_canonical_url': 'url',
           'page_author': 'authors', 
            }, axis=1, inplace=True)

df2.rename({
           'impressions': 'page_impressions',
           'page_efahrer_id': 'page_id',
           'published_at': 'publishing_date',
           'page_canonical_url': 'url',
           'page_author': 'authors', 
            }, axis=1, inplace=True)

# Eliminate mistakes from the table

df1.drop(78658, inplace=True)
df2.drop(40600, inplace=True)

# Merging
# Using the `left` merging: we already know that `df1` is malformatted
key_columns = ['page_id', 'date', 'url', 'authors', 'word_count']
df = pd.merge(left=df2, right=df1, on=key_columns, how='left') 

### Including the scraped data ###


# Writing to the file

df.to_csv('./data/full_data.csv')
