import streamlit as st
import pandas as pd
import plotly.express as px
from utils.visualization import plot_correlation_matrix

def show(df):
    st.title(" Exploratory Data Analysis")
    st.write("Explore patterns and relationships in the climate data.")
    
    # Year range selector
    year_range = st.slider(
        "Select Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )
    
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Variable selector
    variables = st.multiselect(
        "Select Variables to Analyze",
        options=df.columns[1:],
        default=['avg_temp', 'precipitation', 'co2_transport_mt']
    )
    
    if variables:
        # Time series plot
        st.subheader("Time Series Analysis")
        fig = px.line(filtered_df, x='year', y=variables,
                     title='Climate Variables Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Analysis")
        corr_fig = plot_correlation_matrix(filtered_df, variables)
        st.pyplot(corr_fig)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(filtered_df[variables].describe())