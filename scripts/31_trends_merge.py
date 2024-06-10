import pandas as pd

# Read file:
print('======== This script adds the google trends features to the main file ========')

print('Reading the file...')

df = pd.read_csv('data/data_features.csv')
df_trend_match = pd.read_csv('data/data_trends_classified.csv')

merge_keys = ['page_id']

# Perform the third merge with df_trend_match
df = pd.merge(left=df, right=df_trend_match[['predicted_probability', 'predicted_query_label','query_score', 'page_id']], how='left', on=merge_keys)

df.rename(columns={
            'predicted_probability': 'google_trend_prob',
            'predicted_query_label': 'google_trend_label',
            'query_score': 'google_trend_score'
            }, 
        inplace=True)

### Writing to the file ###
print('Writing the final data frame to file...')
df.to_csv('data/data_features_gt.csv', encoding='utf-8', index=False)
print('The full dataframe including google trends features is saved as data/data_features_gt.csv')

print('======== Processing complete ========')