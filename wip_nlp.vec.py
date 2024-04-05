import pandas as pd
import nltk
nltk.download("stopwords") 
nltk.download("punkt")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string

# Import stopwords to remove from text
stop_words = set(stopwords.words('german'))

# Read csv and get rid of unnecessary columns
df = pd.read_csv('./data/data_features.csv')
df_nlp = df.drop(['author','date','meta_image_url','media_type','page_img_size','url_text','meta_title_len','meta_desc_len', 'h1_len','abstract_len', 'merged_url_len'],axis=1)

# Remove stopwords from data
def remove_stopwords(text):
    if isinstance(text, str):        
        words = word_tokenize(text)
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        return ' '.join([word for word in words if word.lower() not in stop_words])
    else:
        return text

def remove_stopwords_from_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(remove_stopwords)
    return df

columns_to_clean = ['h1','abstract','meta_title','meta_description','merged_url']
df_nlp = remove_stopwords_from_columns(df_nlp, columns_to_clean)

# Vectorize preprocessed text
df_nlp_vec = df_nlp.copy()
df_nlp_vec.drop(['url','page_id'],axis=1,inplace=True)
df_nlp_vec.fillna('', inplace=True)

def vectorize_text(column, df):
    col = df[column]
    vect = CountVectorizer().fit(col)
    transformed = vect.transform(col)
    for i, feature_name in enumerate(vect.get_feature_names_out()):
        df[f'{column}_{feature_name}'] = transformed[:, i].toarray().flatten()  # Convert to dense array
    return df

# Save results -> to be finetuned
for item in columns_to_clean:
    df = vectorize_text(column=item, df=df_nlp_vec)
    filename = f'../data/nlp_features_{item}.csv'
    df.to_csv(filename, encoding='utf-8', index=False)

# Merge to one csv
# tbd