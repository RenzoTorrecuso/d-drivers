import streamlit as st
import pandas as pd
import altair as alt

# Set page title and favicon
st.set_page_config(
    page_title="Start - Data Overview",
    page_icon="🟦",
    layout="wide",
)

# # Set background color
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #FFFFFF;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Load DataFrame
df = pd.read_csv('data_nlp_A.csv')

# Page title and image
st.image("/Users/clara/Desktop/neuefische/d-drivers/streamlit_app/DATA-DRIVEN SEARCH FOR TRAFFIC DRIVERS (1).png", use_column_width=True)
st.title("Article Performance Exploration and Prediction for D-DRIVERS")

# Sidebar menu
option = st.selectbox(
    'Select Target Feature',
    ('external_impressions', 'external_clicks', 'ctr')
)

# Filter dataframe based on selected feature
#df_filtered = df[option]
df_filtered = df['external_impressions']

# Line chart
chart = alt.Chart(df).mark_line().encode(
    x='last_publish_date',
    y=alt.Y(option, title=option.capitalize()),
    tooltip=['last_publish_date', 'external_impressions']
).properties(
    width=800,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Display the chart
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.write(chart)

# Add a button to redirect to the Prediction page
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Go to Prediction"):
        st.query_params(page="Prediction")