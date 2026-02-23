import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature_trends(df):
    """Plot temperature trends over years"""
    fig = px.line(df, x='year', y=['avg_temp', 'max_temp', 'min_temp'],
                 title='Temperature Trends Over Years',
                 labels={'value': 'Temperature (°C)', 'year': 'Year'})
    return fig

def plot_correlation_matrix(df, features):
    """Plot correlation matrix for selected features"""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature Correlation Matrix')
    return fig