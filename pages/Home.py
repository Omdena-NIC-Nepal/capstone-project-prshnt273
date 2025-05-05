import streamlit as st
import pandas as pd
from utils.visualization import plot_temperature_trends

def show(df):
    st.title("🌍 Climate Change Prediction Dashboard")
    st.markdown("""
    Welcome to the Climate Change Prediction project. This interactive dashboard allows you to:
    - Explore climate data trends
    - Analyze text about climate change
    - Train and evaluate prediction models
    - Make future climate predictions
    """)
    
    st.header("Dataset Overview")
    st.dataframe(df.head())
    
    st.header("Key Climate Trends")
    fig = plot_temperature_trends(df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Project Features
    - **Exploratory Data Analysis**: Visualize climate patterns
    - **Feature Engineering**: Prepare data for modeling
    - **Model Training**: Build predictive models
    - **Prediction**: Forecast future climate metrics
    - **Text Analysis**: NLP insights from climate reports
    """)