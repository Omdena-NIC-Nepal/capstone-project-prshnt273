import pandas as pd
import os

def load_data():
    """Load processed climate data"""
    data_path = os.path.join("data", "processed", "climate_processed_data.csv")
    return pd.read_csv(data_path)