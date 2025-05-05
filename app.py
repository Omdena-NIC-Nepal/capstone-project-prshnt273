import streamlit as st
from utils.data_loader import load_data

# Configure page
st.set_page_config(
    page_title="Climate Change Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data (cached)
@st.cache_data
def load_climate_data():
    return load_data()

df = load_climate_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Exploratory Data Analysis",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Prediction",
    "Climate Text Analysis"
])

# Display selected page
if page == "Home":
    from pages import Home
    Home.show(df)
elif page == "Exploratory Data Analysis":
    from pages import EDA
    EDA.show(df)
elif page == "Feature Engineering":
    from pages import Feature_Engineering
    Feature_Engineering.show(df)
elif page == "Model Training":
    from pages import Model_Training
    Model_Training.show(df)
elif page == "Model Evaluation":
    from pages import Model_Evaluation
    Model_Evaluation.show(df)
elif page == "Prediction":
    from pages import Prediction
    Prediction.show(df)
elif page == "Climate Text Analysis":
    from pages import Text_Analysis
    Text_Analysis.show()