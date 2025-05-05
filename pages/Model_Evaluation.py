import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (mean_absolute_error, 
                           mean_squared_error, 
                           r2_score)

def load_model(target_var):
    model_path = os.path.join("models", f"{target_var}_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # Get feature names if available (for scikit-learn >= 1.0)
            if hasattr(model, 'feature_names_in_'):
                model.expected_features = model.feature_names_in_
            return model
    return None

def show(df):
    st.title("Model Evaluation")
    st.write("Evaluate the performance of trained climate prediction models.")
    
    # Model selection
    target_var = st.selectbox(
        "Select Target Variable to Evaluate",
        options=['avg_temp', 'max_temp', 'min_temp', 'precipitation'],
        index=0
    )
    
    model = load_model(target_var)
    
    if model is None:
        st.warning(f"No trained model found for {target_var}. Please train a model first.")
        return
    
    # Get the expected features
    if hasattr(model, 'expected_features'):
        required_features = model.expected_features
        st.info(f"This model requires these exact features: {', '.join(required_features)}")
        
        # Check if all required features exist in the dataframe
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required features in your data: {', '.join(missing_features)}")
            return
            
        # Use only the required features in correct order
        X = df[required_features]
    else:
        st.warning("""
        Warning: This model doesn't store feature information. 
        Evaluation may fail if features don't match training configuration.
        """)
        # Fallback - use all features except year and target
        feature_options = [col for col in df.columns if col not in ['year', target_var]]
        selected_features = st.multiselect(
            "Select Features (must match training features)",
            options=feature_options,
            default=feature_options
        )
        
        if not selected_features:
            st.error("Please select at least one feature.")
            return
            
        X = df[selected_features]
    
    y = df[target_var]
    
    try:
        # Make predictions
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("""
        This usually means:
        1. You're using different features than were used in training
        2. The feature order is different from training
        3. Feature data types are incorrect
        """)
        return
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")
    
    # Visualization code remains the same...
    # ... [keep your existing visualization code]