import pandas as pd
import json

df = pd.read_csv('data/data_nlp_A.csv')

df_features = df[['page_id', 'external_impressions', 'ctr', 
                  'video_player_types', 'media_type', 'meta_title', 
                 'sentiment_meta_title', 
                 'sentiment_abstract',
                'clickbait_label', 'clickbait_prob', 
                'abstract', 
              ]].copy()

df_features.rename({'page_id': "ID", 
                    'abstract': "Abstract", 
                    'video_player_types': "Video player types", 
                    'media_type': "Media type", 
                    'sentiment_abstract': "Sentiment abstract", 
                    'meta_title': "Meta title", 
                    'sentiment_meta_title': "Sentiment: meta title",
                    'clickbait_label': "Clickbait", 
                    'clickbait_prob': "Clickbait confidence",
                    'external_impressions': "Page impressions",
                    'ctr': "Click-through"}, axis=1, inplace=True)
df_features.loc[:, 'const'] = 1

df_overview = df[['page_id',
                'external_impressions',
                'ctr',
                'date_min',
                'age',
                'scraped_word_count', 
                'classification_product', 
                'classification_type',
                'n_days', 
                'no_versions',
                'date_scraped',
                'mean_version_lifetime', 
                'n_urls',
                'author_list',
                'scraped_author', 
                'total_likes_n_days', 
                'daily_likes_median', 
                'total_dislikes_n_days',
                'daily_dislikes_median', 
                'video_play',   
                'media_type',
                'meta_title_len', 'meta_desc_len', 'h1_len', 'abstract_len', 'merged_url_len', 'title_has_colon',
                'clickbait_prob', 'clickbait_label', 
                #'google_trend_prob', 'google_trend_label', 'google_trend_score',
                'video_player_types', 'sentiment_abstract', 'confidence_abstract',
                'sentiment_meta_title', 'confidence_meta_title']].copy()

df_features.to_csv('data/sl_app/eda_scatters.csv')

df_overview.rename({
        'page_id': "ID",
        'n_days': "N readings", 
        'date_min': "Earliest date", 
        'n_urls': "N URLs total", 
        'age': "Age (days)", 
        'no_versions': "N versions", #'last_publish_date', #'word_count',
        'classification_product': "Topic", 
        'classification_type': "Type",
        'author_list': "Authors", 
        'external_impressions': "Page impressions",
        'total_likes_n_days': "Likes total", 
        'daily_likes_median': "Likes daily median", 
        'total_dislikes_n_days': "Dislikes total",
        'daily_dislikes_median': "Dislikes daily median",
        'video_play': "Video plays",
        'ctr': "Click-through", 
        'mean_version_lifetime': "Mean version lifetime", 
        'scraped_author': "Author last", 
        'date_scraped': "Last update date", 
        'scraped_word_count': "Word count current",
        'media_type': "Media type",
        'meta_title_len': "Meta title length", 
        'meta_desc_len': "Meta description length",
        'h1_len': "H1 length", 
        'abstract_len': "Abstract length", 
        'merged_url_len': "Unique words in URL", 
        'title_has_colon': "Title has colon",
        'clickbait_prob': "Clickbait confidence", 
        'clickbait_label': "Clickbait", 
        #'google_trend_prob': "Related term confidence", 
        #'google_trend_label': "Related term", 
        #'google_trend_score',
        'video_player_types': "Video player type", 
        'sentiment_abstract': "Sentiment: abstract", 
        'confidence_abstract': "Sentiment conf: abstract",
        'sentiment_meta_title': "Sentiment: title", 
        'confidence_meta_title': "Sentiment conf: title"}, axis=1, inplace=True)
df_overview.loc[:, 'const'] = 1

### Mask author names

authors_map = json.load(open('data/codes/authors.json', 'r', encoding='utf-8'))
authors_map = {key.lower(): value for key, value in authors_map.items()}

df_overview['Author last mask'] = df_overview['Author last'].str.lower()
df_overview['Author last mask'] = df_overview['Author last mask'].str\
                                    .replace('/', ',').replace(' & ', ', ').replace(' und ', ', ')

for auth in authors_map.keys():
    df_overview['Author last mask'] = df_overview['Author last mask'].str.replace(auth, authors_map[auth.lower()])

df_overview.to_csv('data/sl_app/eda_total.csv')