import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

def show(df):
    st.title(" Feature Engineering")
    st.write("Transform and select features for better model performance.")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Section 1: Feature Transformation
    st.header("1. Feature Transformation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scaling Methods")
        scaling_method = st.radio(
            "Select scaling method:",
            ["None", "Standard Scaler (mean=0, std=1)", "MinMax Scaler (0-1)"],
            index=0
        )
        
    with col2:
        st.subheader("Features to Scale")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_features = st.multiselect(
            "Select features to scale:",
            options=numeric_cols,
            default=numeric_cols[:3]
        )
    
    # Apply scaling if selected
    if scaling_method != "None" and scale_features:
        if scaling_method == "Standard Scaler (mean=0, std=1)":
            scaler = StandardScaler()
            df_processed[scale_features] = scaler.fit_transform(df[scale_features])