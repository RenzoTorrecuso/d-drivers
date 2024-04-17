import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Read the CSV file
df = pd.read_csv('data_nlp_A.csv')

# Set page title and favicon
st.set_page_config(
    page_title="Histogram App",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and headline picture
st.title('D-driver')
st.header('Data-driven search for traffic drivers')
st.image('title-d-driver.jpg', caption='Your Caption Here', use_column_width=True)
st.dataframe(df)

# Dropdown to select a column 
selected_column = st.selectbox('Select a column', df.columns)

# Display histogram for the selected column
st.bar_chart(df[selected_column])

# Analyze based on deployed model
st.header('Get an estimate on your article')
title = st.text_input('Select a title')
product = st.selectbox('Select a classification type', df.classification_type)
product = st.selectbox('Select a classification product', df.classification_product)
st.image('placeholder_eda.png', caption='Prediction of based on the input values', use_column_width=True)

# Backend of the presentation
st.header('EDA & Backend topics')
st.markdown("WIP: Add here additional backend things like drill down of the feature groups ")


# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Team D-Driver")
