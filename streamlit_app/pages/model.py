import streamlit as st
from typing import Dict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from sklearn.preprocessing import PowerTransformer
import pickle

# Initialize PowerTransformer
#pt = PowerTransformer()
file_path_specific = '/Users/clara/Desktop/neuefische/d-drivers/notebooks/power_transformer_ext_impr.pkl'
with open(file_path_specific, 'rb') as file:
    loaded_pt = pickle.load(file)

# Function to reverse power transformation
def reverse_power_transformation(predicted_value, pt):
    # Reshape the predicted value for inverse transformation
    predicted_value_transformed = predicted_value.reshape(-1, 1)
    
    # Inverse transform the predicted value
    return loaded_pt.inverse_transform(predicted_value_transformed)


def predict_tabular_regression_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    
    # Initialize client that will be used to create and send requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    
    # Parse the instance dictionary
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    
    # Prepare empty parameters
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    
    # Construct the endpoint path
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    
    try:
        # Send the prediction request
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        
        # Display the response
        st.write("Response:")
        st.write(f"Deployed Model ID: {response.deployed_model_id}")
        
        # Print the predictions
        predictions = response.predictions
        for prediction in predictions:
            st.write("Prediction:", dict(prediction))
            
    except Exception as e:
        st.write(f"Error during prediction: {e}")


# Streamlit app
st.title("Vertex AI Prediction Demo")

# Input features
word_count = st.text_input("Word Count", "443.0")
classification_product = st.selectbox("Classification Product", ["E-Auto"])
classification_type = st.selectbox("Classification Type", ["News"])
urls_per_days = st.text_input("URLs per Days", "0.125")
meta_title_len = st.text_input("Meta Title Length", "72.0")
meta_desc_len = st.text_input("Meta Description Length", "153.0")
h1_len = st.text_input("H1 Length", "72.0")
abstract_len = st.text_input("Abstract Length", "260.0")
google_trend_label = st.text_input("Google Trend Label", "elektroauto")
google_trend_score = st.text_input("Google Trend Score", "31.0")
sentiment_abstract_neutral = st.selectbox("Sentiment Abstract Neutral", ["True", "False"])
sentiment_abstract_positive = st.selectbox("Sentiment Abstract Positive", ["True", "False"])
sentiment_meta_title_neutral = st.selectbox("Sentiment Meta Title Neutral", ["True", "False"])
sentiment_meta_title_positive = st.selectbox("Sentiment Meta Title Positive", ["True", "False"])
video_standard_and_widget = st.selectbox("Video Standard and Widget", ["True", "False"])
video_widget = st.selectbox("Video Widget", ["True", "False"])
title_has_colon_True = st.selectbox("Title has Colon True", ["True", "False"])
media_type_other = st.selectbox("Media Type Other", ["True", "False"])
media_type_video = st.selectbox("Media Type Video", ["True", "False"])
authors = st.text_input("Authors", "lemur")
YOUR_N_DAYS_VALUE = st.number_input("Expected time the article is online", 100)

# Prepare instance dictionary
instance_dict = {
    "word_count": word_count,
    "classification_product": classification_product,
    "classification_type": classification_type,
    "urls_per_days": urls_per_days,
    "meta_title_len": meta_title_len,
    "meta_desc_len": meta_desc_len,
    "h1_len": h1_len,
    "abstract_len": abstract_len,
    "google_trend_label": google_trend_label,
    "google_trend_score": google_trend_score,
    "sentiment_abstract_neutral": sentiment_abstract_neutral,
    "sentiment_abstract_positive": sentiment_abstract_positive,
    "sentiment_meta_title_neutral": sentiment_meta_title_neutral,
    "sentiment_meta_title_positive": sentiment_meta_title_positive,
    "video_standard_and_widget": video_standard_and_widget,
    "video_widget": video_widget,
    "title_has_colon_True": title_has_colon_True,
    "media_type_other": media_type_other,
    "media_type_video": media_type_video,
    "Authors": authors
}

# Button to trigger prediction
if st.button("Predict"):
    response = predict_tabular_regression_sample(
        project="428691118973",
        endpoint_id="1008014668158992384",
        instance_dict=instance_dict,
        location="europe-north1",
        api_endpoint="europe-north1-aiplatform.googleapis.com"
    )
    
    # Extract prediction value from the respoanse
#    predicted_value = response.predictions[0]['value']
    predicted_value = response['value']

    # Reverse Power Transformation
    reversed_value_transformed = reverse_power_transformation(predicted_value, loaded_pt)
    
    # Reverse Normalization by multiplying through n_days
    n_days = YOUR_N_DAYS_VALUE 
    reversed_value = reversed_value_transformed * n_days
    
    # Display the reversed value
    st.write(f"Predicted Value (Original Scale): {reversed_value}")