import pandas as pd

df = pd.read_csv('data/data_aggr.csv', parse_dates=['date', 'publish_date'])

# ------------------------------------------------------------ #
# Aggregate on page_id level
# ------------------------------------------------------------ #

df = df.drop(['date'],axis=1)\
                .groupby(['page_id'], as_index=False)\
                .agg(
                        {'url':'last',
                        'version_id': 'max',
                        'publish_date': 'max',
                        'word_count': 'mean',
                        'classification_product': 'first',
                        'classification_type': 'first',
                        'page_name': 'first',
                        'title': 'first',
                        'authors': 'last', 
                        'external_clicks': 'sum', 
                        'external_impressions': 'sum',
                        'daily_likes': 'sum',
                        'daily_dislikes': 'sum',
                        'video_play': 'sum',
                        'page_impressions': 'sum',
                        'clickouts': 'sum'
                        }
                )

# Rename multiple columns
df.rename(columns={'version_id': 'no_versions',
                   'publish_date': 'last_publish_date',
                   'authors': 'last_author'
                   }, inplace=True)

print('Writing to the file...')
df.to_csv('data/data_aggr_page_id.csv', index=False)

print('Aggregated data saved as "data/data_aggr_page_id.csv" ')
print()
print('======== Processing complete ========')
