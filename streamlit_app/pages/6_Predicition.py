import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
from sklearn.preprocessing import PowerTransformer
import pickle
import pycaret.regression as pyreg

from typing import Dict, List



# Initialize PowerTransformer
file_path_pt = "./pages/power_transformer_ext_impr.pkl"
with open(file_path_pt, "rb") as file:
    loaded_pt = pickle.load(file)

# Initialize Light Gradient Boost Machine model
model_loaded = pyreg.load_model('../models/lightgbm_tuned_ext_imp_pt')

# # Initialize Light Gradient Boost Machine model
# file_path_model = "../streamlit_app/pages/lightgbm_tuned_ext_imp_pt.pkl"
# with open(file_path_model, "rb") as file:
#     loaded_model = pickle.load(file)

def inverse_transform(predicted_value, pt):
    if isinstance(predicted_value, float):
        # Reshape the predicted value for inverse transformation
        predicted_value_transformed = [[predicted_value]]
    else:
        # Reshape the predicted value for inverse transformation
        predicted_value_transformed = predicted_value.reshape(-1, 1)

    # Inverse transform the predicted value
    return pt.inverse_transform(predicted_value_transformed)


def predict_ext_imp(x, model_predict, model_transform): 
    # predict
    y_trans = model_predict.predict(x)
    # inverse transform
    y = inverse_transform(y_trans, model_transform)
    return y

def load_sample_set(row_index, column_list):
    return dict

def load_samples(i=0, n_rows=100):
    df = pd.read_csv('../data/data_nlp_A.csv', parse_dates=['last_publish_date', 'date_min'], nrows=n_rows)

    selected_columns = [
        'h1',
        'abstract',
        'meta_title',
        'meta_description',
        #'article', # --> word_count
        'google_trend_label',
        'google_trend_score',
        'author_list',
        'urls_per_days',
        'classification_product', # not needed for most recent model
        'classification_type'
    ]
    df_sample = df[selected_columns].iloc[i]
    
    return df_sample
    #df_sample.h1

def insert_samples(df):
    st.session_state.session_title_text = df.h1

#######################
#### Streamlit app ####
#######################

# Set page title and favicon
st.set_page_config(
    page_title="Prediction - Article Impressions",
    page_icon="🟦",
    layout="wide",
)

st.image("DATA-DRIVEN SEARCH FOR TRAFFIC DRIVERS.png", use_column_width=True)
st.title("Prediciton of Article Impressions")

# Initialize the text_input box with a default value if not already initialized
# If-clause prevents to load defaults whenever session refreshes
if "h1" not in st.session_state:
    st.session_state.h1 = "Your Title Here"
if "abstract" not in st.session_state:
    st.session_state.abstract = "Your Abstract Here"
if "meta_description" not in st.session_state:   
    st.session_state.meta_description = "Your Meta Description Here"
if "classification_product" not in st.session_state:
    st.session_state.classification_product = "E-Auto"  # not needed for most recent model
if "classification_type" not in st.session_state:
    st.session_state.classification_type = "Ratgeber"
if "google_trend_label" not in st.session_state:
    st.session_state.google_trend_label = "Your Label Here, e.g. 'elektroauto'"
if "google_trend_score" not in st.session_state:
    st.session_state.google_trend_score = 33
if "urls_per_days" not in st.session_state:
    st.session_state.urls_per_days = 0.5
if "author_list" not in st.session_state:
    st.session_state.author_list = "Eva Goldschald"

# overwrite default values with sample data
if st.button("Load Samples"):
    if 'df_sample' not in globals():
        df_sample = load_samples(2) 
        st.session_state.h1 = df_sample.h1
        st.session_state.abstract = df_sample.abstract
        st.session_state.meta_description = df_sample.meta_description
        st.session_state.classification_product = df_sample.classification_product
        st.session_state.classification_type = df_sample.classification_type
        st.session_state.google_trend_label = df_sample.google_trend_label
        st.session_state.google_trend_score = df_sample.google_trend_score
        st.session_state.urls_per_days = df_sample.urls_per_days
        st.session_state.author_list = df_sample.author_list    # to be checked: streamlit control with addable items


# Text areas for Title, Abstract, and Article
# prepopulation with session_state and key is needed for loading sample data
title_text = st.text_input("Title", st.session_state.h1, key='h1')
abstract_text = st.text_area("Abstract", st.session_state.abstract, key='abstract')
meta_description_text = st.text_area("Meta Description", st.session_state.meta_description, key='meta_description')
article_text = st.text_area("Article", "Your Article Here")


# Further Input features

# Section for classification (product and type)
cols = st.columns(2)
with cols[0]:
    classification_product = st.selectbox(
        "Classification Product",
        [
            "E-Auto",
            "Auto",
            "Zubehör",
            "Motorrad",
            "Energie",
            "Verkehr",
            "Wallbox/Laden",
            "Solaranlagen",
            "E-Bike",
            "Fahrrad",
            "E-Scooter",
            "Solarspeicher",
            "Balkonkraftwerk",
            "Solargenerator",
            "THG",
            "Wärmepumpe",
            "Versicherung",
        ], key='classification_product')
with cols[1]:
    classification_type = st.selectbox(
        "Classification Type",
        ["Ratgeber", "News", "Kaufberatung", "Deal", "Test", "Erfahrungsbericht", "Video"], key='classification_type')

# Section for Authors and URLs per days
cols = st.columns(2)
with cols[0]:
    author_list = st.selectbox(
        "Author(s)",
        ["Eva Goldschald", "Moritz Diethelm", "Marius Eichfelder", "Christian Lutz", "Lisa Brack", "Christian Lutz", "Sebastian Barsch;Christian Lutz", "Carina Dietze", "Vanessa Finkler"], key='author_list')
with cols[1]:
    urls_per_days = st.slider(
        "Publishing Frequency", min_value=0.0, max_value=2.0, step=0.01, key='urls_per_days')

# Section for media types
cols = st.columns(2)    
with cols[0]:
    # video_standard_and_widget = st.selectbox("Video Standard and Widget", ["True", "False"])
    media_type = st.radio("Media Type", ["img", "video"])
with cols[1]:
    if media_type == "video":
        video_widget = st.radio("Video type", ["1 - Standard", "2 - Standard and Widget", "3 - Widget"])
    else:
        video_widget = "False"
# n_days = st.slider("Expected time the article is online",0,20)
n_days = 15  # mean of n_days

cols = st.columns(2)

with cols[0]:
    google_trend_label = st.text_input("Google Trends Label", st.session_state.google_trend_label, key='google_trend_label')
with cols[1]:
    google_trend_score = st.number_input(label='Google Trends Score', step=1., format='%.0f', placeholder=33, key='google_trend_score')

# Count characters in Title, Abstract, and Article
h1_len = len(title_text)
meta_title_len = len(title_text)
abstract_len = len(abstract_text)
meta_desc_len = abstract_len - 100
word_count = abstract_len + len(article_text)

# Prepare instance dictionary (input for model prediction)
instance_dict = {
    "word_count": word_count, # e.g. 390
    "classification_product": classification_product, # e.g. 'Solaranlagen',
    "classification_type": classification_type, # e.g. 'Kaufberatung',
    "urls_per_days": urls_per_days, # e.g. 0.1,
    "meta_title_len": meta_title_len, # e.g. 75,
    "meta_desc_len": meta_desc_len, # e.g. 145,
    "h1_len": h1_len, # e.g. 66,
    "abstract_len": abstract_len, # e.g. 250,
    "google_trend_label": google_trend_label, # e.g. 'elektroauto',
    "google_trend_score": google_trend_score, # e.g. 33,
    "video_player_types": video_widget, # e.g. '2 - Standard and Widget',
    "media_type": media_type, # e.g. 'video',
    "author_list": author_list, # e.g. 'Lisa Brack',
}

df_pred = pd.DataFrame([instance_dict])

# Button to trigger prediction
if st.button("Predict"):
    prediction = predict_ext_imp(df_pred, model_loaded, loaded_pt)[0][0]
    if (prediction is not None):
        ui.metric_card(title="Predicted Impressions:", 
                    content=f"{prediction:,.2f}", 
                    description="", 
                    key="card1")
    else:
        st.write("No valid predictions received.")

t1 = 'Prediction based on these independent variables (features):'

instance_dict
