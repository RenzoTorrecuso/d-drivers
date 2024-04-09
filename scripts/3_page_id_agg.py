import pandas as pd

df = pd.read_csv('data/data_aggr.csv', parse_dates=['date', 'publish_date'])

# ------------------------------------------------------------ #
# Aggregate on page_id level
# ------------------------------------------------------------ #

df = df.groupby(['page_id'], as_index=False)\
        .agg(
        {
                'date':'count',
                'url':'last',
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
df.rename(columns={'date': 'n_days', # N observations for the given article
                   'version_id': 'no_versions',
                   'publish_date': 'last_publish_date',
                   'authors': 'last_author',
                   'daily_likes': 'likes_n_days', # the current likes and dislikes would make more sense
                   'daily_dislikes': 'total_likes_n_days', # These two columns make sense only on the daily basis
                   }, inplace=True)                        #           ...should we drop them?

print('Writing to the file...')
df.to_csv('data/data_aggr_page_id.csv', index=False)

print('Aggregated data saved as "data/data_aggr_page_id.csv" ')
print()
print('======== Processing complete ========')
