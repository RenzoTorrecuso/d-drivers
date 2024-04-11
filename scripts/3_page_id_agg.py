import pandas as pd

df = pd.read_csv('data/data_aggr.csv', parse_dates=['date', 'publish_date'])


def aggregate_strings(url_column):
    """Collects all string values which have occured for the key 
    into a string of unique elements separated by ';' """

    # Step 1: Split the strings in each row by the delimiter ';' to get individual elements
    url_column = url_column.apply(lambda x: x.split(';'))
    
    # Step 2: Merge all the lists together
    all_elements = sum(url_column, [])
    
    # Step 3: Remove duplicates
    unique_elements = list(set(all_elements))
    
    # Step 4: Concatenate the unique elements into a single string
    result = ';'.join(unique_elements)
    
    return result

# ------------------------------------------------------------ #
# Aggregate on page_id level
# ------------------------------------------------------------ #

df = df.groupby(['page_id'], as_index=False)\
        .agg(
        {
                'date':'count',
                'url': aggregate_strings,
                'version_id': 'max',
                'publish_date': ['max', 'min'],
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
df.columns = [col[0] + '_' + col[1] if col[0] == 'publish_date' else col[0] for col in df.columns]

df.insert(3, 'n_urls', df.url.apply(lambda urllist: len(urllist.split(';'))))
df.insert(5, 'age', (pd.Timestamp('2024-04-01 00:00') - df.publish_date_min).apply(lambda td: td.days))

# Rename multiple columns
df.rename(columns={'date': 'n_days', # N observations for the given article
                   'version_id': 'no_versions',
                   'publish_date_max': 'last_publish_date',
                   'authors': 'last_author',
                   'daily_likes': 'likes_n_days', # the current likes and dislikes would make more sense
                   'daily_dislikes': 'dislikes_n_days', # These two columns make sense only on the daily basis
                   }, inplace=True)                        #           ...should we drop them?

print('Writing to the file...')
df.to_csv('data/data_aggr_page_id.csv', index=False)

print('Aggregated data saved as "data/data_aggr_page_id.csv" ')
print()
print('======== Processing complete ========')
