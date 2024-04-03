import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

col = ['title', 'h1', 'abstract', 'url_text']

# def vectorize_text(column, df=df):
#     col = df[column]
#     vect = CountVectorizer().fit(col)
#     transformed = vect.transform(col)
#     for i, feature_name in enumerate(vect.get_feature_names_out()):
#         df[f'{column}_{feature_name}'] = transformed[:, i].toarray().flatten()  # Convert to dense array
#     return df

# for item in ['title', 'h1', 'abstract', 'url_text']:
#     df = vectorize_text(column=item, df=df)
#     filename = f'../data/full_data_{item}.csv'
#     df.to_csv(filename, encoding='utf-8', index=False)


# df.to_csv('./data/full_data_vectorized.csv', encoding='utf-8', index=False)


# sorted(vect.vocabulary_.items(), key=lambda x: x[1])[:20]


# print("Text dataframe shape = {}".format(col.shape))
# print("Vocabulary length = {}".format(len(vect.vocabulary_)))