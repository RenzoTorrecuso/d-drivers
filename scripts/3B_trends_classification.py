## For explanations see ./notebooks/googel_trends_classification.ipynb
##  INPUT (required): data_features.csv, related_queries.csv
##  OUTPUT: data_trends_classified.csv
## Runtime: this script will take more than 48 hours to run on an average CPU (as of April 2024)
## Recommendation: run this on a GPU or on google colab with T4 environment (see comments in the notebook)

from transformers import pipeline
import pandas as pd
from tqdm import tqdm

###################################
# 0. Functions for selecting score with highest probability and looping over dataframe
###################################

def get_predictions_score(prediction):
    """
    Description:
        Function for returning the labels and scores with the highes prediction probability
        from a given classification pipeline
    Args:
        prediction (pipeline): classification pipeline

    Returns:
        max_labe (str): label with highest probability
        max_probability (float): highest probability
    """
    pred_labels = prediction['labels']
    pred_scores = prediction['scores']
    
    # Find the index of the label with the highest probability
    max_index = pred_scores.index(max(pred_scores))
    
    # Extract the label and its corresponding probability
    max_label = pred_labels[max_index]
    max_probability = pred_scores[max_index]
    
    return max_label, max_probability

def trends_classify(filter, df_labels, df_gscore, classifier, limititer=0):
    """Function for labelling dataframe with labels and scores based on highest prediction probability
    Args:
        filter (str): filter string used for filtering dataframe by certain classification_product
        df_labels (dataframe): dataframe df_labels containing all labels to be filtered by filter.
        df_gscore (dataframe): dataframe df_gscore containing all articles with text to be classified, to be filtered by filter.
        classifier (pipeline): classifier for labelling.
        limititer (int): Defaults to 0. If int is given iteration over rows will stop at value of int
    Returns:
        df_gscore_iter: dataframe df_score with two new columns (label and proba) and several labelingresults per row
    """
    iter = filter

    df_labels_per_category = df_labels[df_labels['classification_product'] == iter]
    candidate_labels = df_labels_per_category['query'].astype(str).tolist()

    df_gscore_iter = df_gscore[df_gscore['classification_product'] == iter]

    # shorten df to only return limited rows per function call
    if limititer > 0 and limititer < len(df_gscore_iter):
        df_gscore_iter = df_gscore_iter.iloc[0:limititer]

    tqdm.pandas(desc=f"Googel search related keyword classification for {iter}")
    # replace progress_apply with apply if you run this on google colab with T4 or on a GPU
    df_gscore_iter['predicted_query_label'], df_gscore_iter['predicted_probability'] = zip(*df_gscore_iter['text_to_classify'].apply(lambda x: get_predictions_score(classifier(x, candidate_labels))))

    return df_gscore_iter

###################################
# 1. Load data for labeling process
###################################

file_path_features = '../data/data_features.csv'
file_path_labels = '../data/related_queries.csv'

df = pd.read_csv(file_path_features)
df_labels = pd.read_csv(file_path_labels)

###################################
# 2. Enrich data with google score
###################################


# prepare dataset by creating a concatenated column to be classified which considers the abstract, the meta title and the meta description
relevant_columns = ['page_id', 'classification_product', 'abstract', 'meta_description', 'meta_title' ]
df_gscore = df[relevant_columns].copy()
df_gscore['text_to_classify'] = df_gscore['abstract'].fillna('') + ' ' + df_gscore['meta_description'].fillna('') + ' ' + df_gscore['meta_title'].fillna('')
class_product = df.classification_product.unique().tolist()

# Initialisation of the pipeline
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=0)

# Classify (can take long)
iterations = 0  # set to small number for testrun, e.g. 1 or 2 (0 means unlimited).

# create empty instance of target dataframe
df_gscore_out = pd.DataFrame(columns=relevant_columns + ['text_to_classify', 'predicted_query_label', 'predicted_probability'])
for cp in tqdm(class_product):
    print(f"Googel search related keyword classification for {cp}")
    df_gscore_classified = trends_classify(cp, df_labels, df_gscore, classifier, limititer=iterations)
    df_gscore_out = pd.concat([df_gscore_out, df_gscore_classified], axis=0).reset_index(drop=True)

###################################
# 3 Join score to dataset and export
###################################

# df_labels['predicted_query_label'] equals to df_labels['query']
df_labels = df_labels.rename(columns={'query': 'predicted_query_label', 'value': 'query_score'})

# join score in one step
df_gscore_new = df_gscore_out.merge(df_labels, on=['classification_product', 'predicted_query_label'], how='left')
df_gscore_new = df_gscore_new.drop_duplicates(keep='first')

df_gscore_new.to_csv('../data/google_trends/data_trends_classified.csv', encoding='utf-8', index=False)