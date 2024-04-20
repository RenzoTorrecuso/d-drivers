from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st
import altair as alt

 
# Set page title and favicon
st.set_page_config(
    page_title="Data Overview",
    page_icon="🟦",
    layout="wide",
)
 
# Load DataFrame
df = pd.read_csv('../data/preprocessing_nlp_v4.csv')

# Page title and image
st.image("/Users/clara/Desktop/neuefische/d-drivers/streamlit_app/DATA-DRIVEN SEARCH FOR TRAFFIC DRIVERS (1).png", use_column_width=True)
st.title("Composition of dataset")

#import streamlit as st
import altair as alt
import pandas as pd

# Sidebar menu
feature = st.selectbox('Select Feature:', ['word_count', 'classification_product', 'classification_type',
                                            'meta_title_len', 'meta_desc_len', 'h1_len', 'abstract_len',
                                            'sentiment_abstract_neutral', 'sentiment_abstract_positive',
                                            'video_standard_and_widget', 'video_widget', 'not_clickbait',
                                            'title_has_colon_True', 'media_type_video', 'Authors'],
                                            index=1)

option = st.selectbox('Select Target Feature', ['external_impressions', 'external_clicks', 'ctr'])

# Filter dataframe based on selected feature
df_filtered = df[[feature]]

# Calculate frequency of each category
freq_df = df_filtered[feature].value_counts().reset_index()
freq_df.columns = [feature, 'count']

# Create bar chart for count
bar_count = alt.Chart(freq_df).mark_bar().encode(
    x=alt.X(f"{feature}:N", title=feature.capitalize()),
    y=alt.Y('count:Q', title='Count'),
    tooltip=['count:Q']
).properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Filter dataframe based on selected option
df_filtered_option = df[[feature, option]]

# Create bar chart for performance
bar_performance = alt.Chart(df_filtered_option).mark_bar().encode(
    x=alt.X(f"{feature}:N", title=feature.capitalize()),
    y=alt.Y(f"{option}:Q", title=option.capitalize())
).properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Display the charts side by side
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown(f"<h2>Distribution of features</h2>", unsafe_allow_html=True)
    st.write(bar_count)

with col2:
    st.markdown(f"<h2>Performance of features</h2>", unsafe_allow_html=True)
    st.write(bar_performance)
