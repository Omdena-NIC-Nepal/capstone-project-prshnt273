import pandas as pd
import os
from sklearn.impute import SimpleImputer
import numpy as np

def load_and_process_rainfall(filepath):
    """Load and process rainfall data"""
    rainfall = pd.read_csv(filepath, comment='#')
    rainfall['date'] = pd.to_datetime(rainfall['date'])
    rainfall['year'] = rainfall['date'].dt.year
    
    rainfall_annual = rainfall.groupby('year').agg({
        'rfh': 'mean',
        'rfh_avg': 'mean',
        'r1h': 'mean',
        'r1h_avg': 'mean',
        'r3h': 'mean',
        'r3h_avg': 'mean',
        'rfq': 'mean',
        'r1q': 'mean',
        'r3q': 'mean'
    }).reset_index()
    
    return rainfall_annual

def load_and_clean(filepath, col_name):
    """Helper function to load and clean other datasets"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace('\ufeff', '')
    df = df.rename(columns={df.columns[0]: 'year', df.columns[1]: col_name})
    return df

def process_eco_socio_env(filepath):
    """Process the economic/socio-environmental data from long to wide format"""
    df = pd.read_csv(filepath)
    
    # Filter for specific indicators we want
    indicators = [
        "Fertilizer consumption (kilograms per hectare of arable land)",
        "Agricultural land (sq. km)",
        "Carbon dioxide (CO2) emissions from Transport (Energy) (Mt CO2e)"
    ]
    
    df = df[df['Indicator Name'].isin(indicators)]
    
    # Create short column names
    short_names = {
        "Fertilizer consumption (kilograms per hectare of arable land)": "fert_consumption_kg_ha",
        "Agricultural land (sq. km)": "agri_land_sqkm",
        "Carbon dioxide (CO2) emissions from Transport (Energy) (Mt CO2e)": "co2_transport_mt"
    }
    
    # Pivot to wide format and drop country columns
    df_wide = df.pivot_table(
        index=['Year'],
        columns='Indicator Name',
        values='Value'
    ).reset_index()
    
    # Rename columns using short names
    df_wide = df_wide.rename(columns=short_names)
    df_wide = df_wide.rename(columns={'Year': 'year'})
    
    return df_wide

def impute_missing_data(df):
    """Impute missing values with appropriate strategies"""
    df_imputed = df.copy()
    
    strategies = {
        # Temperature and humidity
        'avg_temp': 'linear',
        'max_temp': 'linear',
        'min_temp': 'linear',
        'humidity': 'linear',
        
        # Precipitation
        'precipitation': 'median',
        
        # Rainfall metrics
        'rfh': 'ffill',
        'rfh_avg': 'ffill',
        'r1h': 'ffill',
        'r1h_avg': 'ffill',
        'r3h': 'ffill',
        'r3h_avg': 'ffill',
        'rfq': 'ffill',
        'r1q': 'ffill',
        'r3q': 'ffill',
        
        # Economic/socio-environmental
        'fert_consumption_kg_ha': 'linear',
        'agri_land_sqkm': 'linear',
        'co2_transport_mt': 'linear'
    }
    
    for col, strategy in strategies.items():
        if col in df_imputed.columns:
            if strategy == 'linear':
                df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
            elif strategy == 'median':
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
            elif strategy == 'ffill':
                df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_imputed

def combine_all_data(climate_dir='data/raw_data/climate', 
                    socio_dir='data/raw_data/socio_economic',
                    output_dir='data/processed'):
    """Main function to combine and impute all datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct file paths - climate data
    rainfall_path = os.path.join(climate_dir, 'npl-rainfall-adm2-full.csv')
    temp_avg_path = os.path.join(climate_dir, 'observed-annual-average_temp.csv')
    temp_max_path = os.path.join(climate_dir, 'observed-annual-average-max-temp.csv')
    temp_min_path = os.path.join(climate_dir, 'observed-annual-average-min-temp.csv')
    humidity_path = os.path.join(climate_dir, 'observed-annual-relative-humidity.csv')
    precipitation_path = os.path.join(climate_dir, 'precipitation data.csv')
    
    # Construct file path - socio-economic data
    eco_socio_path = os.path.join(socio_dir, 'eco-socio-env-health-edu-dev-energy_npl.csv')
    
    # Load and process data
    rainfall_data = load_and_process_rainfall(rainfall_path)
    temp_avg = load_and_clean(temp_avg_path, 'avg_temp')
    temp_max = load_and_clean(temp_max_path, 'max_temp')
    temp_min = load_and_clean(temp_min_path, 'min_temp')
    humidity = load_and_clean(humidity_path, 'humidity')
    precipitation = load_and_clean(precipitation_path, 'precipitation')
    eco_socio_data = process_eco_socio_env(eco_socio_path)
    
    # Merge datasets step by step
    combined = temp_avg.merge(temp_max, on='year', how='outer')
    combined = combined.merge(temp_min, on='year', how='outer')
    combined = combined.merge(humidity, on='year', how='outer')
    combined = combined.merge(precipitation, on='year', how='outer')
    combined = combined.merge(rainfall_data, on='year', how='outer')
    combined = combined.merge(eco_socio_data, on='year', how='outer')
    
    # Sort by year
    final_df = combined.sort_values('year').reset_index(drop=True)
    
    # Impute missing values
    final_df_imputed = impute_missing_data(final_df)
    
    # Save  imputed data
    output_path_imputed = os.path.join(output_dir, 'climate_processed_data.csv')
    
    final_df_imputed.to_csv(output_path_imputed, index=False)
    
    return final_df_imputed, output_path_imputed

# Execute the processing pipeline
processed_data, output_file = combine_all_data()

print("\nData processing and imputation complete!")
print(f"Imputed data saved to: {output_file}")
print(f"Final dataframe shape: {processed_data.shape}")
print("\nMissing values after imputation:")
print(processed_data.isna().sum())
print("\nFirst 5 years of imputed data:")
print(processed_data.head())