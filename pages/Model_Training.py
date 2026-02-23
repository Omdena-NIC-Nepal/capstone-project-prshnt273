import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os

def show(df):
    st.title(" Model Training")
    st.write("Train machine learning models to predict climate variables.")
    
    # Model configuration
    st.sidebar.header("Model Configuration")
    target_var = st.sidebar.selectbox(
        "Select Target Variable",
        options=['avg_temp', 'max_temp', 'min_temp', 'precipitation']
    )
    
    feature_options = [col for col in df.columns if col not in ['year', target_var]]
    selected_features = st.sidebar.multiselect(
        "Select Features",
        options=feature_options,
        default=['co2_transport_mt', 'fert_consumption_kg_ha', 'humidity']
    )
    
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    
    if st.sidebar.button("Train Model"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            # Prepare data
            X = df[selected_features]
            y = df[target_var]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # Save model
            model_path = os.path.join("models", f"{target_var}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Display results
            st.success(f"Model trained successfully for {target_var}!")
            st.metric("Mean Squared Error", round(mse, 4))
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance.set_index('Feature'))