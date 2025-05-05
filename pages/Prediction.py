import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(target_var):
    model_path = os.path.join("models", f"{target_var}_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # Store the expected features if available
            if hasattr(model, 'feature_names_in_'):
                model.expected_features = model.feature_names_in_
            return model
    return None

def show(df):
    st.title("Climate Prediction")
    st.write("Make future climate predictions using trained models.")
    
    # Model selection
    target_var = st.selectbox(
        "Select Target Variable to Predict",
        options=['avg_temp', 'max_temp', 'min_temp', 'precipitation'],
        index=0
    )
    
    model = load_model(target_var)
    
    if model is None:
        st.warning(f"No trained model found for {target_var}. Please train a model first.")
        return
    
    # Get expected features from the model
    if hasattr(model, 'expected_features'):
        feature_options = model.expected_features
        st.info(f"Model requires these features: {', '.join(feature_options)}")
    else:
        st.warning("Model doesn't have feature names information. Using all available features except target and year.")
        feature_options = [col for col in df.columns if col not in ['year', target_var]]
    
    st.subheader("Input Features for Prediction")
    
    # Create input form for features
    input_features = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(feature_options):
        col = cols[i % 2]
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].median())
        
        input_features[feature] = col.number_input(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val)/100
        )
    
    # Year selection
    prediction_year = st.number_input(
        "Prediction Year",
        min_value=2023,
        max_value=2100,
        value=2030,
        step=1
    )
    
    if st.button("Make Prediction"):
        # Prepare input data with correct feature order
        try:
            input_data = pd.DataFrame([input_features])[feature_options]  # Ensure correct order
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.success(f"Predicted {target_var} for {prediction_year}: {prediction:.4f}")
            
            # Show historical trend
            st.subheader("Historical Trend vs Prediction")
            
            # Filter data for the last 30 years
            current_year = datetime.now().year
            historical_data = df[df['year'] >= (current_year - 30)]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=historical_data, x='year', y=target_var, ax=ax, label='Historical')
            
            # Add prediction point
            plt.scatter(prediction_year, prediction, color='red', s=100, label='Prediction')
            
            ax.set_xlabel("Year")
            ax.set_ylabel(target_var)
            ax.set_title(f"Historical Trend and Prediction for {target_var}")
            ax.legend()
            
            st.pyplot(fig)
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Impact on Prediction")
                
                importance = pd.DataFrame({
                    'Feature': feature_options,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance, ax=ax2)
                ax2.set_title("Feature Importance")
                st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please ensure you're providing all required features in the correct format.")