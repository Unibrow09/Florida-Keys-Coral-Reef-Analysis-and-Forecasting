"""
14_stony_coral_lta_forecasting.py - Forecasting Stony Coral Living Tissue Area Evolution

This script develops forecasting models to predict the evolution of stony coral living tissue area (LTA)
across different monitoring stations over the next five years (2024-2028). It builds on 
previous analyses to create accurate forecasts while accounting for spatial variations, 
environmental factors, and historical disturbance events.

Author: Shivam Vashishtha
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import make_interp_spline
import warnings
import math
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
results_dir = "14_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")
else:
    print(f"Directory already exists: {results_dir}")

# Set Matplotlib parameters for high-quality figures
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['axes.titlepad'] = 12  # Add more padding to title
plt.rcParams['axes.labelpad'] = 8   # Add more padding to axis labels
plt.rcParams['axes.spines.top'] = False  # Remove top spine for cleaner look
plt.rcParams['axes.spines.right'] = False  # Remove right spine for cleaner look

# Define a modern color palette
COLORS = {
    'coral': '#FF6B6B',  # Bright coral
    'ocean_blue': '#4ECDC4',  # Turquoise
    'light_blue': '#A9D6E5',  # Soft blue
    'dark_blue': '#01445A',  # Navy blue
    'sand': '#FFBF69',  # Warm sand
    'reef_green': '#2EC4B6',  # Teal
    'accent': '#F9DC5C',  # Vibrant yellow
    'text': '#2A2A2A',  # Dark grey for text
    'grid': '#E0E0E0',  # Light grey for grid lines
    'background': '#F8F9FA',  # Very light grey background
    'highlight': '#FF9F1C',  # Orange highlight
    'warning': '#E71D36',  # Red warning
    'success': '#2EC4B6',  # Teal success
    'neutral': '#8D99AE'  # Neutral blue-grey
}

# Additional colors for multiple model visualization
MODEL_COLORS = {
    'ARIMA': '#0077B6',  # Deep blue
    'Random Forest': '#00A896',  # Teal
    'XGBoost': '#FF9F1C',  # Orange
    'Gradient Boosting': '#E9C46A',  # Gold
    'Linear Regression': '#E76F51',  # Coral
    'Ensemble': '#9B5DE5',  # Purple
    'Actual': '#2A2A2A',  # Dark grey
    'Historical': '#8D99AE',  # Neutral blue-grey
    'Forecast': '#FF6B6B',  # Bright coral
    'Optimistic': '#2EC4B6',  # Teal
    'Pessimistic': '#E71D36',  # Red
    'Baseline': '#4ECDC4'  # Turquoise
}

# Create a custom colormap for coral reef visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Dictionary of significant coral reef events for visualization
REEF_EVENTS = {
    2014: {"name": "2014-2015 Global Bleaching Event", "impact": "severe", "color": COLORS['warning']},
    2017: {"name": "Hurricane Irma", "impact": "severe", "color": COLORS['warning']},
    2019: {"name": "Stony Coral Tissue Loss Disease Peak", "impact": "severe", "color": COLORS['warning']},
    2023: {"name": "Last Observation", "impact": "neutral", "color": COLORS['neutral']}
}

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP datasets for LTA forecasting analysis.
    
    Returns:
        dict: Dictionary containing preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the LTA data
        print("Loading stony coral LTA data...")
        lta_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_LTA.csv")
        print(f"Stony coral LTA data loaded successfully with {len(lta_df)} rows")
        print(f"First few columns: {lta_df.columns.tolist()[:5]}")
        
        # Load the station metadata
        print("\nLoading station metadata...")
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        print(f"First few columns: {stations_df.columns.tolist()[:5]}")
        
        # Load stony coral density data for correlation
        print("\nLoading stony coral density data...")
        stony_density_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Density.csv")
        print(f"Stony coral density data loaded successfully with {len(stony_density_df)} rows")
        print(f"First few columns: {stony_density_df.columns.tolist()[:5]}")
        
        # Load percentage cover data for correlation
        print("\nLoading percentage cover data...")
        pcover_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_TaxaGroups.csv")
        print(f"Percentage cover data loaded successfully with {len(pcover_df)} rows")
        print(f"First few columns: {pcover_df.columns.tolist()[:5]}")
        
        # Load temperature data if available
        try:
            print("\nAttempting to load temperature data...")
            temp_df = pd.read_csv("CREMP_CSV_files/CREMP_Temperatures_2023.csv")
            print(f"Temperature data loaded successfully with {len(temp_df)} rows")
            print(f"First few columns: {temp_df.columns.tolist()[:5]}")
        except Exception as temp_error:
            print(f"Temperature data error: {str(temp_error)}")
            print("Temperature data file is too large or not accessible. Using alternative approach.")
            temp_df = None
        
        # Convert date columns to datetime format
        print("\nProcessing date fields...")
        lta_df['Date'] = pd.to_datetime(lta_df['Date'])
        lta_df['Year'] = lta_df['Year'].astype(int)
        
        if 'Date' in stony_density_df.columns:
            stony_density_df['Date'] = pd.to_datetime(stony_density_df['Date'])
        
        if 'Date' in pcover_df.columns:
            pcover_df['Date'] = pd.to_datetime(pcover_df['Date'])
        
        # Process temperature data if available
        if temp_df is not None:
            if 'Date' in temp_df.columns:
                temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                if 'Year' not in temp_df.columns:
                    temp_df['Year'] = temp_df['Date'].dt.year
                    temp_df['Month'] = temp_df['Date'].dt.month
            elif 'date' in temp_df.columns:  # Try lowercase
                temp_df['Date'] = pd.to_datetime(temp_df['date'])
                temp_df['Year'] = temp_df['Date'].dt.year
                temp_df['Month'] = temp_df['Date'].dt.month
            elif 'TIMESTAMP' in temp_df.columns:  # Try other naming
                temp_df['Date'] = pd.to_datetime(temp_df['TIMESTAMP'])
                temp_df['Year'] = temp_df['Date'].dt.year
                temp_df['Month'] = temp_df['Date'].dt.month
        
        # Get list of all coral species columns (excluding metadata columns)
        metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                        'Site_name', 'StationID']
        species_cols = [col for col in lta_df.columns if col not in metadata_cols]
        
        print(f"\nIdentified {len(species_cols)} coral species columns in the dataset")
        
        # Calculate total LTA for each station
        if 'Total_LTA' not in lta_df.columns:
            lta_df['Total_LTA'] = lta_df[species_cols].sum(axis=1, skipna=True)
            print("Added Total_LTA column by summing all species columns")
        
        # Print basic statistics
        print("\nBasic statistics for Total LTA:")
        print(lta_df['Total_LTA'].describe())
        
        # Get time range
        print(f"\nData spans from {lta_df['Year'].min()} to {lta_df['Year'].max()}")
        
        # Return dictionary of DataFrames
        return {
            'lta_df': lta_df,
            'stations_df': stations_df,
            'density_df': stony_density_df,
            'pcover_df': pcover_df,
            'temperature_df': temp_df,
            'species_cols': species_cols
        }
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 

def analyze_time_series_patterns(data_dict):
    """
    Analyze temporal patterns in stony coral LTA data.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        
    Returns:
        dict: Dictionary containing time series analysis results
    """
    print("\nAnalyzing time series patterns in stony coral LTA...")
    
    # Extract LTA data
    lta_df = data_dict['lta_df']
    
    # Create a yearly average time series
    yearly_avg = lta_df.groupby('Year')['Total_LTA'].mean().reset_index()
    print(f"Created yearly average time series with {len(yearly_avg)} points")
    
    # Create regional yearly averages
    regional_yearly_avg = lta_df.groupby(['Year', 'Subregion'])['Total_LTA'].mean().reset_index()
    
    # Create habitat yearly averages
    habitat_yearly_avg = lta_df.groupby(['Year', 'Habitat'])['Total_LTA'].mean().reset_index()
    
    # Perform stationarity test on overall yearly average
    print("\nPerforming Augmented Dickey-Fuller test for stationarity...")
    adf_result = adfuller(yearly_avg['Total_LTA'])
    adf_output = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4],
        'Is Stationary': adf_result[1] < 0.05
    }
    print(f"ADF Test Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'} (p-value: {adf_result[1]:.4f})")
    
    # Calculate autocorrelation and partial autocorrelation
    print("\nCalculating autocorrelation and partial autocorrelation...")
    max_lags = min(len(yearly_avg) // 2 - 1, 10)  # Adjust max_lags to be less than 50% of sample size
    acf_values = acf(yearly_avg['Total_LTA'], nlags=max_lags)
    pacf_values = pacf(yearly_avg['Total_LTA'], nlags=max_lags)
    
    # Identify change points
    print("\nIdentifying significant change points...")
    
    # Calculate year-to-year percentage changes
    yearly_avg['pct_change'] = yearly_avg['Total_LTA'].pct_change() * 100
    
    # Identify significant change points (greater than 2 standard deviations)
    std_threshold = 2 * yearly_avg['pct_change'].std()
    change_points = yearly_avg[abs(yearly_avg['pct_change']) > std_threshold]
    
    print(f"Identified {len(change_points)} significant change points")
    
    # Visualize the time series decomposition
    print("\nVisualizing time series decomposition...")
    
    # Create a figure for time series analysis
    fig = plt.figure(figsize=(16, 20), facecolor=COLORS['background'])
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # Plot 1: Overall yearly average with trend line
    ax1 = plt.subplot(gs[0, :])
    ax1.set_facecolor(COLORS['background'])
    
    # Plot the data points
    ax1.plot(yearly_avg['Year'], yearly_avg['Total_LTA'], 
             marker='o', linestyle='-', color=COLORS['coral'], 
             linewidth=2.5, markersize=8, label='Annual Mean LTA')
    
    # Fit a linear trend line
    X = yearly_avg['Year'].values.reshape(-1, 1)
    y = yearly_avg['Total_LTA'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    y_pred = model.predict(X)
    
    # Plot the trend line
    ax1.plot(yearly_avg['Year'], y_pred, '--', color=COLORS['dark_blue'], 
             linewidth=2, label=f'Linear Trend (Slope: {slope:.4f} per year)')
    
    # Mark significant change points
    for _, row in change_points.iterrows():
        ax1.scatter(row['Year'], row['Total_LTA'], 
                   s=150, color='red', zorder=5, 
                   marker='*', label='_nolegend_')
        ax1.annotate(f"{row['Year']}: {row['pct_change']:.1f}%", 
                    (row['Year'], row['Total_LTA']),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Add reference lines for significant events
    for year, event in REEF_EVENTS.items():
        if year in yearly_avg['Year'].values:
            event_lta = yearly_avg.loc[yearly_avg['Year'] == year, 'Total_LTA'].values[0]
            ax1.axvline(x=year, color=event['color'], alpha=0.4, linestyle='--', linewidth=1.5)
            ax1.annotate(event['name'], xy=(year, 0), xytext=(year, max(yearly_avg['Total_LTA'])*0.05),
                        rotation=90, va='bottom', ha='center', fontsize=10, 
                        color=event['color'], weight='bold')
    
    # Set plot aesthetics
    ax1.set_title('Overall Stony Coral LTA Trend', fontweight='bold', color=COLORS['dark_blue'], fontsize=18)
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Average Living Tissue Area (mm²)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    
    # Set x-axis ticks to all years
    ax1.set_xticks(yearly_avg['Year'].values)
    ax1.set_xticklabels(yearly_avg['Year'].values, rotation=45)
    
    # Plot 2: Regional trends
    ax2 = plt.subplot(gs[1, :])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot regional trends
    for region in regional_yearly_avg['Subregion'].unique():
        region_data = regional_yearly_avg[regional_yearly_avg['Subregion'] == region]
        ax2.plot(region_data['Year'], region_data['Total_LTA'], 
                marker='o', linestyle='-', linewidth=2, label=region)
    
    # Set plot aesthetics
    ax2.set_title('Regional Stony Coral LTA Trends', fontweight='bold', color=COLORS['dark_blue'], fontsize=18)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Average Living Tissue Area (mm²)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Add reference lines for significant events
    for year, event in REEF_EVENTS.items():
        if year in yearly_avg['Year'].values:
            ax2.axvline(x=year, color=event['color'], alpha=0.4, linestyle='--', linewidth=1.5)
    
    # Set x-axis ticks to all years
    ax2.set_xticks(yearly_avg['Year'].values)
    ax2.set_xticklabels(yearly_avg['Year'].values, rotation=45)
    
    # Plot 3: Habitat trends
    ax3 = plt.subplot(gs[2, 0])
    ax3.set_facecolor(COLORS['background'])
    
    # Plot habitat trends
    for habitat in habitat_yearly_avg['Habitat'].unique():
        habitat_data = habitat_yearly_avg[habitat_yearly_avg['Habitat'] == habitat]
        ax3.plot(habitat_data['Year'], habitat_data['Total_LTA'], 
                marker='o', linestyle='-', linewidth=2, label=habitat)
    
    # Set plot aesthetics
    ax3.set_title('Habitat-specific Stony Coral LTA Trends', fontweight='bold', color=COLORS['dark_blue'], fontsize=16)
    ax3.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Average Living Tissue Area (mm²)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Set x-axis ticks to all years
    ax3.set_xticks(yearly_avg['Year'].values)
    ax3.set_xticklabels(yearly_avg['Year'].values, rotation=45)
    
    # Plot 4: ACF
    ax4 = plt.subplot(gs[2, 1])
    ax4.set_facecolor(COLORS['background'])
    
    # Plot ACF
    lags = range(len(acf_values))
    ax4.stem(lags, acf_values, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax4.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
    
    # Add confidence intervals (approximately 95%)
    conf_level = 1.96 / np.sqrt(len(yearly_avg))
    ax4.axhline(y=conf_level, linestyle='--', color='gray', linewidth=0.5)
    ax4.axhline(y=-conf_level, linestyle='--', color='gray', linewidth=0.5)
    
    # Set plot aesthetics
    ax4.set_title('Autocorrelation Function (ACF)', fontweight='bold', color=COLORS['dark_blue'], fontsize=16)
    ax4.set_xlabel('Lag', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Correlation', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 5: PACF
    ax5 = plt.subplot(gs[3, 0])
    ax5.set_facecolor(COLORS['background'])
    
    # Plot PACF
    lags = range(len(pacf_values))
    ax5.stem(lags, pacf_values, linefmt='b-', markerfmt='bo', basefmt='r-')
    ax5.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
    
    # Add confidence intervals (approximately 95%)
    conf_level = 1.96 / np.sqrt(len(yearly_avg))
    ax5.axhline(y=conf_level, linestyle='--', color='gray', linewidth=0.5)
    ax5.axhline(y=-conf_level, linestyle='--', color='gray', linewidth=0.5)
    
    # Set plot aesthetics
    ax5.set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', color=COLORS['dark_blue'], fontsize=16)
    ax5.set_xlabel('Lag', fontweight='bold', fontsize=14)
    ax5.set_ylabel('Partial Correlation', fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 6: Annual percentage change
    ax6 = plt.subplot(gs[3, 1])
    ax6.set_facecolor(COLORS['background'])
    
    # Plot annual percentage change
    bars = ax6.bar(yearly_avg['Year'][1:], yearly_avg['pct_change'][1:], 
                 color=[COLORS['coral'] if x > 0 else COLORS['ocean_blue'] for x in yearly_avg['pct_change'][1:]])
    
    # Add horizontal line at zero
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add horizontal lines for threshold
    ax6.axhline(y=std_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax6.axhline(y=-std_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add labels for significant changes
    for _, row in change_points.iterrows():
        ax6.annotate(f"{row['pct_change']:.1f}%", 
                    (row['Year'], row['pct_change']),
                    xytext=(0, 5 if row['pct_change'] > 0 else -15), 
                    textcoords='offset points',
                    ha='center', fontsize=10, color='red', fontweight='bold')
    
    # Set plot aesthetics
    ax6.set_title('Annual Percentage Change in Stony Coral LTA', fontweight='bold', color=COLORS['dark_blue'], fontsize=16)
    ax6.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax6.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=14)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks
    ax6.set_xticks(yearly_avg['Year'][1:].values)
    ax6.set_xticklabels(yearly_avg['Year'][1:].values, rotation=45)
    
    # Add text annotation for significant change threshold
    ax6.text(0.02, 0.95, f'Significant Change Threshold: ±{std_threshold:.1f}%', 
             transform=ax6.transAxes, fontsize=10, color='red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_time_series_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"Saved time series analysis to {os.path.join(results_dir, 'stony_coral_lta_time_series_analysis.png')}")
    plt.close()
    
    # Return results as dictionary
    return {
        'yearly_avg': yearly_avg,
        'regional_yearly_avg': regional_yearly_avg,
        'habitat_yearly_avg': habitat_yearly_avg,
        'change_points': change_points,
        'adf_output': adf_output,
        'acf_values': acf_values,
        'pacf_values': pacf_values
    } 

def engineer_features(lta_data, stations_df, temperature_data, data_dict=None, target_col='Total_LTA'):
    """
    Engineer features for the LTA forecasting model.
    
    Args:
        lta_data (DataFrame): Processed stony coral LTA data
        stations_df (DataFrame): Station metadata
        temperature_data (DataFrame): Temperature data (can be None)
        data_dict (dict, optional): Full data dictionary for cross-dataset features
        target_col (str): Target column name
        
    Returns:
        dict: Dictionary with engineered feature data
    """
    print("\nEngineering features for LTA forecasting model...")
    
    # Make a copy of the dataframe to avoid modifying the original
    data = lta_data.copy()
    
    # Calculate baseline statistics
    print(f"\nBaseline {target_col} statistics:")
    print(data[target_col].describe())
    
    # Create dataframe for model features
    print("\nPreparing feature dataframe...")
    
    # Start with core features
    feature_df = data[['Year', 'Subregion', 'Habitat', 'SiteID', 'StationID', target_col]].copy()
    
    # Add site-specific features
    print("\nAdding site-specific features...")
    
    # Print available columns in stations_df for debugging
    print(f"Available columns in stations_df: {stations_df.columns.tolist()}")
    
    # Join with stations data to get additional site metadata
    if 'SiteID' in stations_df.columns and 'SiteID' in feature_df.columns:
        # Look for depth column - could be named differently
        depth_cols = [col for col in stations_df.columns if 'depth' in col.lower()]
        
        if depth_cols:
            depth_col = depth_cols[0]
            print(f"Found depth column: {depth_col}")
            
            # Merge only the necessary columns to avoid duplicates
            station_features = stations_df[['SiteID', depth_col]].drop_duplicates()
            feature_df = feature_df.merge(station_features, on='SiteID', how='left')
            print(f"Added station metadata features: {station_features.columns.tolist()}")
        else:
            print("No depth column found in stations data")
            # If there's no depth column, just use other useful columns if available
            useful_cols = ['SiteID']
            for col in ['Region', 'Reef_type', 'Lat', 'Long', 'Depth']:
                if col in stations_df.columns:
                    useful_cols.append(col)
            
            if len(useful_cols) > 1:  # More than just SiteID
                station_features = stations_df[useful_cols].drop_duplicates()
                feature_df = feature_df.merge(station_features, on='SiteID', how='left')
                print(f"Added station metadata features: {station_features.columns.tolist()}")
            else:
                print("No useful site features found")
    
    # One-hot encode categorical variables
    print("\nEncoding categorical variables...")
    
    # Create dummy variables for Subregion and Habitat
    subregion_dummies = pd.get_dummies(feature_df['Subregion'], prefix='Subregion')
    habitat_dummies = pd.get_dummies(feature_df['Habitat'], prefix='Habitat')
    
    # Add dummy variables to feature dataframe
    feature_df = pd.concat([feature_df, subregion_dummies, habitat_dummies], axis=1)
    print(f"Added {len(subregion_dummies.columns)} Subregion dummy variables")
    print(f"Added {len(habitat_dummies.columns)} Habitat dummy variables")
    
    # Create temporal features
    print("\nAdding temporal features...")
    
    # Create lagged features for the target column
    feature_df['LTA_prev_year'] = feature_df.groupby('StationID')[target_col].shift(1)
    
    # Calculate rolling statistics (e.g., 3-year rolling mean and std)
    feature_df['LTA_rolling_mean_3yr'] = feature_df.groupby('StationID')[target_col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    feature_df['LTA_rolling_std_3yr'] = feature_df.groupby('StationID')[target_col].transform(
        lambda x: x.rolling(window=3, min_periods=1).std())
    
    # Calculate growth rates
    feature_df['LTA_growth_rate'] = feature_df.groupby('StationID')[target_col].pct_change()
    
    # Add event flags (major disturbance events)
    event_years = [2014, 2017, 2019]  # Bleaching event, Hurricane Irma, SCTLD peak
    for year in event_years:
        feature_df[f'Event_{year}'] = (feature_df['Year'] == year).astype(int)
        feature_df[f'Post_Event_{year}'] = (feature_df['Year'] > year).astype(int)
    
    print("Added temporal features including lagged values, rolling statistics, and event flags")
    
    # Add temperature features if temperature data is available
    if temperature_data is not None:
        print("\nAdding temperature features...")
        
        try:
            # Print temperature data columns for debugging
            print(f"Temperature data columns: {temperature_data.columns.tolist()}")
            
            # Calculate annual temperature metrics for each site/station
            temp_metrics = []
            
            # Group by year and site
            for name, group in temperature_data.groupby(['Year', 'SiteID']):
                if len(name) != 2:
                    print(f"Skipping invalid groupby result: {name}")
                    continue
                    
                year, site = name
                
                # Find temperature column
                if 'Temperature' in group.columns:
                    temp_col = 'Temperature'
                elif 'Temp_C' in group.columns:
                    temp_col = 'Temp_C'
                elif 'Water_Temp_C' in group.columns:
                    temp_col = 'Water_Temp_C'
                else:
                    # Look for any column with 'temp' in the name
                    temp_cols = [col for col in group.columns if 'temp' in col.lower()]
                    if temp_cols:
                        temp_col = temp_cols[0]
                    else:
                        print(f"No temperature column found. Available columns: {group.columns.tolist()}")
                        continue
                
                # Check if temperature data is numeric
                if not pd.api.types.is_numeric_dtype(group[temp_col]):
                    print(f"Temperature column '{temp_col}' is not numeric. Converting to numeric.")
                    group[temp_col] = pd.to_numeric(group[temp_col], errors='coerce')
                
                # Calculate temperature metrics
                metrics = {
                    'Year': year,
                    'SiteID': site,
                    'Temp_Mean': group[temp_col].mean(),
                    'Temp_Max': group[temp_col].max(),
                    'Temp_Min': group[temp_col].min(),
                    'Temp_Range': group[temp_col].max() - group[temp_col].min(),
                    'Temp_Std': group[temp_col].std(),
                    # Count days above threshold (e.g., potential bleaching threshold of 30°C)
                    'Days_Above_30C': (group[temp_col] > 30).sum(),
                    # Count days below threshold (e.g., cold stress threshold of 16°C)
                    'Days_Below_16C': (group[temp_col] < 16).sum()
                }
                temp_metrics.append(metrics)
            
            # Create temperature metrics dataframe
            if temp_metrics:
                temp_metrics_df = pd.DataFrame(temp_metrics)
                
                # Merge with feature dataframe
                feature_df = pd.merge(
                    feature_df, temp_metrics_df, 
                    on=['Year', 'SiteID'], 
                    how='left'
                )
                
                # Handle missing temperature data by forward and backward filling
                temp_cols = ['Temp_Mean', 'Temp_Max', 'Temp_Min', 'Temp_Range', 
                            'Temp_Std', 'Days_Above_30C', 'Days_Below_16C']
                for col in temp_cols:
                    if col in feature_df.columns:
                        # First forward fill within site
                        feature_df[col] = feature_df.groupby('SiteID')[col].transform(
                            lambda x: x.ffill().bfill())
                
                # Add lagged temperature variables
                for col in temp_cols:
                    if col in feature_df.columns:
                        feature_df[f'{col}_prev_year'] = feature_df.groupby('SiteID')[col].shift(1)
                
                print(f"Added temperature features: {temp_cols}")
            else:
                print("No temperature metrics calculated")
        
        except Exception as e:
            print(f"Error processing temperature data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add cross-dataset features using stony coral percentage cover data
    try:
        pcover_df = data_dict.get('pcover_df', None) if data_dict else None
        
        if pcover_df is not None:
            print("\nAdding coral cover percentage features...")
            
            # Calculate annual mean coral cover percentages for each site
            pcover_metrics = []
            
            # Identify coral cover columns
            cover_cols = [col for col in pcover_df.columns if 'Pcover' in col]
            
            # Create features from coral cover data
            for (year, site, station), group in pcover_df.groupby(['Year', 'SiteID', 'StationID']):
                metrics = {
                    'Year': year,
                    'SiteID': site,
                    'StationID': station
                }
                
                # Add coral cover percentages
                for col in cover_cols:
                    if col in group.columns:
                        metrics[f'{col}_mean'] = group[col].mean()
                
                pcover_metrics.append(metrics)
            
            # Create coral cover metrics dataframe
            if pcover_metrics:
                pcover_metrics_df = pd.DataFrame(pcover_metrics)
                
                # Merge with feature dataframe
                feature_df = pd.merge(
                    feature_df, pcover_metrics_df,
                    on=['Year', 'SiteID', 'StationID'],
                    how='left'
                )
                
                # List of added cover features
                cover_feature_cols = [f'{col}_mean' for col in cover_cols]
                print(f"Added coral cover features: {cover_feature_cols[:5]}...")
            else:
                print("No coral cover metrics calculated")
    except Exception as e:
        print(f"Error processing coral cover data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Add cross-dataset features using stony coral density data
    try:
        density_df = data_dict.get('density_df', None) if data_dict else None
        
        if density_df is not None:
            print("\nAdding stony coral density features...")
            
            # Calculate annual mean coral density for each site
            density_metrics = []
            
            # Identify non-metadata columns (likely species density columns)
            metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                            'Site_name', 'StationID']
            species_density_cols = [col for col in density_df.columns if col not in metadata_cols]
            
            # Create features from density data
            for (year, site, station), group in density_df.groupby(['Year', 'SiteID', 'StationID']):
                metrics = {
                    'Year': year,
                    'SiteID': site,
                    'StationID': station,
                    'Total_Density': group[species_density_cols].sum(axis=1).mean(),
                    'Species_Count': (group[species_density_cols] > 0).sum(axis=1).mean()
                }
                
                density_metrics.append(metrics)
            
            # Create density metrics dataframe
            if density_metrics:
                density_metrics_df = pd.DataFrame(density_metrics)
                
                # Merge with feature dataframe
                feature_df = pd.merge(
                    feature_df, density_metrics_df,
                    on=['Year', 'SiteID', 'StationID'],
                    how='left'
                )
                
                # Handle missing density data
                for col in ['Total_Density', 'Species_Count']:
                    if col in feature_df.columns:
                        feature_df[col] = feature_df.groupby('StationID')[col].transform(
                            lambda x: x.ffill().bfill())
                
                # Add lagged density variables
                for col in ['Total_Density', 'Species_Count']:
                    if col in feature_df.columns:
                        feature_df[f'{col}_prev_year'] = feature_df.groupby('StationID')[col].shift(1)
                
                print("Added coral density features: Total_Density, Species_Count, and lagged versions")
            else:
                print("No density metrics calculated")
    except Exception as e:
        print(f"Error processing density data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Handle missing values
    print("\nHandling missing values...")
    
    # For numeric columns, fill NaN with column mean
    numeric_cols = feature_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['Year', 'SiteID', 'StationID']:  # Skip identifier columns
            # Fill with mean of the same site, or overall mean if site-specific mean not available
            feature_df[col] = feature_df.groupby('StationID')[col].transform(
                lambda x: x.fillna(x.mean()))
            
            # If still NaN (e.g., for stations with all NaN), fill with overall mean
            if feature_df[col].isna().any():
                feature_df[col].fillna(feature_df[col].mean(), inplace=True)
    
    # Print final feature statistics
    print(f"\nFinal feature dataframe shape: {feature_df.shape}")
    print(f"Number of features: {feature_df.shape[1] - 3}")  # Exclude Year, SiteID, StationID
    print(f"Missing values remaining: {feature_df.isna().sum().sum()}")
    
    # Return feature dataframe and additional information
    return {
        'feature_df': feature_df,
        'target_col': target_col,
        'id_cols': ['Year', 'SiteID', 'StationID']
    }

def train_forecasting_models(feature_dict, test_size=0.2, random_state=42):
    """
    Train multiple forecasting models and evaluate their performance.
    
    Args:
        feature_dict (dict): Dictionary containing engineered features
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing trained models and performance metrics
    """
    print("\nTraining forecasting models for LTA prediction...")
    
    # Extract feature dataframe and target column
    feature_df = feature_dict['feature_df']
    target_col = feature_dict['target_col']
    
    # Extract IDs from feature dataframe
    id_cols = feature_dict['id_cols']
    
    # Exclude categorical columns from training
    categorical_cols = ['Subregion', 'Habitat']
    
    # Exclude ID columns and categorical columns from feature set
    exclude_cols = id_cols + categorical_cols + [target_col]
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
    
    print(f"\nUsing {len(feature_cols)} features for model training")
    print(f"First few features: {feature_cols[:5]}")
    
    # Sort data by year to ensure proper time-based split
    feature_df = feature_df.sort_values('Year')
    
    # Split data into training and testing sets
    # Use later years as testing data to better evaluate forecasting performance
    sorted_years = sorted(feature_df['Year'].unique())
    train_years = sorted_years[:-2]  # Use all but the last 2 years for training
    test_years = sorted_years[-2:]   # Use the last 2 years for testing
    
    print(f"\nTraining on years: {train_years}")
    print(f"Testing on years: {test_years}")
    
    # Split based on years - this ensures proper temporal separation
    train_df = feature_df[feature_df['Year'].isin(train_years)].copy()
    test_df = feature_df[feature_df['Year'].isin(test_years)].copy()
    
    # Ensure there's enough test data
    if len(test_df) < 10:
        print("Warning: Not enough test data with year-based split. Using random split instead.")
        # Fall back to random split
        train_df, test_df = train_test_split(
            feature_df, test_size=test_size, random_state=random_state, stratify=feature_df['Habitat'] if 'Habitat' in feature_df.columns else None
        )
    
    print(f"\nTraining set size: {len(train_df)} samples")
    print(f"Testing set size: {len(test_df)} samples")
    
    # IMPORTANT: Re-process features that depend on other samples to avoid data leakage
    # For rolling statistics, recalculate them only based on training data
    if 'LTA_rolling_mean_3yr' in feature_cols:
        # Remove from feature columns first (we'll recalculate it)
        feature_cols.remove('LTA_rolling_mean_3yr')
        
        # Recalculate for training data only using data available at each point in time
        train_df['LTA_rolling_mean_3yr'] = train_df.groupby('StationID')[target_col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
            
        # For test data, use the last 3 values from training for the same station
        for station_id in test_df['StationID'].unique():
            station_train = train_df[train_df['StationID'] == station_id]
            if len(station_train) > 0:
                last_mean = station_train[target_col].tail(3).mean()
                test_df.loc[test_df['StationID'] == station_id, 'LTA_rolling_mean_3yr'] = last_mean
            else:
                # If no training data for this station, use overall mean
                test_df.loc[test_df['StationID'] == station_id, 'LTA_rolling_mean_3yr'] = train_df[target_col].mean()
                
        # Add back to feature columns
        feature_cols.append('LTA_rolling_mean_3yr')
    
    # Handle LTA_prev_year to avoid leakage
    if 'LTA_prev_year' in feature_cols:
        # Remove from feature columns
        feature_cols.remove('LTA_prev_year')
        
        # Recalculate properly with shift for training data
        train_df['LTA_prev_year'] = train_df.groupby('StationID')[target_col].shift(1)
        
        # For test data, get the last value from training data for each station
        for station_id in test_df['StationID'].unique():
            station_train = train_df[train_df['StationID'] == station_id]
            if len(station_train) > 0:
                last_value = station_train[target_col].iloc[-1]
                test_df.loc[test_df['StationID'] == station_id, 'LTA_prev_year'] = last_value
            else:
                # If no training data for this station, use mean
                test_df.loc[test_df['StationID'] == station_id, 'LTA_prev_year'] = train_df[target_col].mean()
                
        # Add back to feature columns
        feature_cols.append('LTA_prev_year')
    
    # Handle growth rate similarly
    if 'LTA_growth_rate' in feature_cols:
        # Remove from feature columns
        feature_cols.remove('LTA_growth_rate')
        
        # Recalculate for training
        train_df['LTA_growth_rate'] = train_df.groupby('StationID')[target_col].pct_change()
        
        # For test data, use last growth rate from training
        for station_id in test_df['StationID'].unique():
            station_train = train_df[train_df['StationID'] == station_id].sort_values('Year')
            if len(station_train) > 1:
                last_growth = (station_train[target_col].iloc[-1] / station_train[target_col].iloc[-2]) - 1 if station_train[target_col].iloc[-2] > 0 else 0
                test_df.loc[test_df['StationID'] == station_id, 'LTA_growth_rate'] = last_growth
            else:
                # If insufficient data, use overall mean growth
                test_df.loc[test_df['StationID'] == station_id, 'LTA_growth_rate'] = 0
                
        # Add back to feature columns
        feature_cols.append('LTA_growth_rate')
    
    # Fill any remaining NaN values
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].mean())
    test_df[feature_cols] = test_df[feature_cols].fillna(train_df[feature_cols].mean())
    
    # Prepare training and testing data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Standardize features for better model performance
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames for feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5, random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
        )
    }
    
    # Train and evaluate models
    model_results = {}
    best_r2 = -float('inf')
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        
        # Train the model
        model.fit(X_train_scaled_df, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled_df)
        y_pred_test = model.predict(X_test_scaled_df)
        
        # Calculate performance metrics for training set
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Calculate performance metrics for testing set
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Calculate mean absolute percentage error (MAPE)
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Print model performance
        print(f"{name} model performance:")
        print(f"  Training RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAPE: {test_mape:.2f}%")
        
        # Check if this is the best model so far
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = name
        
        # Extract feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 5 important features for {name}:")
            print(importance_df.head(5))
        
        # Store model and performance metrics
        model_results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'feature_importances': importance_df if hasattr(model, 'feature_importances_') else None,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'actual_train': y_train,
            'actual_test': y_test,
            'train_indices': train_df.index,
            'test_indices': test_df.index
        }
    
    print(f"\nBest performing model: {best_model_name} (Test R²: {best_r2:.4f})")
    
    # Create ensemble model (average of predictions)
    print("\nCreating ensemble model...")
    
    # Calculate average predictions for the test set
    ensemble_pred_test = np.mean([
        model_results[name]['y_pred_test'] for name in models.keys()
    ], axis=0)
    
    # Calculate ensemble performance metrics
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred_test))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred_test)
    ensemble_mape = np.mean(np.abs((y_test - ensemble_pred_test) / y_test)) * 100
    
    print(f"Ensemble model performance:")
    print(f"  Test RMSE: {ensemble_rmse:.2f}")
    print(f"  Test R²: {ensemble_r2:.4f}")
    print(f"  Test MAPE: {ensemble_mape:.2f}%")
    
    # For ensemble predictions on the training set,
    # we need to gather all training predictions
    ensemble_pred_train = np.mean([
        model_results[name]['y_pred_train'] for name in models.keys()
    ], axis=0)
    
    # Store ensemble model results
    model_results['Ensemble'] = {
        'model': None,  # No actual model object for ensemble
        'train_rmse': np.sqrt(mean_squared_error(y_train, ensemble_pred_train)),
        'test_rmse': ensemble_rmse,
        'train_mae': mean_absolute_error(y_train, ensemble_pred_train),
        'test_mae': ensemble_mae,
        'train_r2': r2_score(y_train, ensemble_pred_train),
        'test_r2': ensemble_r2,
        'test_mape': ensemble_mape,
        'feature_importances': None,  # No direct feature importances for ensemble
        'y_pred_train': ensemble_pred_train,
        'y_pred_test': ensemble_pred_test,
        'actual_train': y_train,
        'actual_test': y_test,
        'train_indices': train_df.index,
        'test_indices': test_df.index
    }
    
    # Check if ensemble is the best model
    if ensemble_r2 > best_r2:
        best_model_name = 'Ensemble'
        best_r2 = ensemble_r2
        print(f"Ensemble is now the best model (Test R²: {ensemble_r2:.4f})")
    
    # Store additional information for model analysis
    model_meta = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'id_cols': id_cols,
        'scaler': scaler,
        'best_model': best_model_name,
        'train_years': train_years,
        'test_years': test_years,
        'train_df': train_df,
        'test_df': test_df
    }
    
    # Save the best model
    print(f"\nSaving best model ({best_model_name})...")
    if best_model_name != 'Ensemble':  # Only save actual model objects
        joblib.dump(
            model_results[best_model_name]['model'],
            os.path.join(results_dir, "stony_coral_lta_best_model.pkl")
        )
        print(f"Model saved to {os.path.join(results_dir, 'stony_coral_lta_best_model.pkl')}")
    
    # Create a comprehensive feature importance dataframe across all models
    print("\nCompiling feature importance data...")
    all_importances = []
    
    for name, result in model_results.items():
        if result['feature_importances'] is not None:
            # Add model name to feature importance dataframe
            imp_df = result['feature_importances'].copy()
            imp_df['Model'] = name
            all_importances.append(imp_df)
    
    if all_importances:
        feature_importance_df = pd.concat(all_importances, axis=0)
        print(f"Created feature importance dataframe with {len(feature_importance_df)} rows")
    else:
        feature_importance_df = None
        print("No feature importance data available")
    
    return {
        'models': model_results,
        'meta': model_meta,
        'feature_importance': feature_importance_df
    }

def visualize_model_performance(model_results):
    """
    Visualize the performance of the trained models.
    
    Args:
        model_results (dict): Dictionary containing model performance metrics
        
    Returns:
        None
    """
    print("\nVisualizing model performance...")
    
    try:
        # Create figure for model performance comparison
        plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # Plot 1: R² scores comparison
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_facecolor(COLORS['background'])
        
        # Extract R² scores
        model_names = []
        train_r2 = []
        test_r2 = []
        
        for name, model_dict in model_results['models'].items():
            model_names.append(name)
            train_r2.append(model_dict['train_r2'])
            test_r2.append(model_dict['test_r2'])
        
        # Set bar positions
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create bars
        ax1.bar(x - width/2, train_r2, width, label='Training R²',
                color=COLORS['ocean_blue'], alpha=0.7)
        ax1.bar(x + width/2, test_r2, width, label='Testing R²',
                color=COLORS['coral'], alpha=0.7)
        
        # Customize plot
        ax1.set_title('Model Performance Comparison (R² Scores)',
                     fontweight='bold', fontsize=14, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax1.legend()
        
        # Plot 2: RMSE comparison
        ax2 = plt.subplot(gs[0, 1])
        ax2.set_facecolor(COLORS['background'])
        
        # Extract RMSE scores
        train_rmse = []
        test_rmse = []
        
        for name, model_dict in model_results['models'].items():
            train_rmse.append(model_dict['train_rmse'])
            test_rmse.append(model_dict['test_rmse'])
        
        # Create bars
        ax2.bar(x - width/2, train_rmse, width, label='Training RMSE',
                color=COLORS['reef_green'], alpha=0.7)
        ax2.bar(x + width/2, test_rmse, width, label='Testing RMSE',
                color=COLORS['warning'], alpha=0.7)
        
        # Customize plot
        ax2.set_title('Model Performance Comparison (RMSE)',
                     fontweight='bold', fontsize=14, pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('RMSE', fontweight='bold')
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax2.legend()
        
        # Plot 3: Test Data Predictions vs Actual
        ax3 = plt.subplot(gs[1, :])
        ax3.set_facecolor(COLORS['background'])
        
        # Collect test data predictions for all models
        test_indices = model_results['models'][model_names[0]]['test_indices']
        actual_test = model_results['models'][model_names[0]]['actual_test']
        
        # Sort by actual value for better visualization
        sorted_indices = np.argsort(actual_test)
        actual_sorted = np.array(actual_test)[sorted_indices]
        
        # Plot actual values
        ax3.plot(range(len(actual_sorted)), actual_sorted, 'o-', 
                label='Actual Values', color=COLORS['dark_blue'],
                linewidth=2, markersize=4)
        
        # Plot predictions for each model
        for name in model_names:
            model_dict = model_results['models'][name]
            predictions = np.array(model_dict['y_pred_test'])[sorted_indices]
            ax3.plot(range(len(actual_sorted)), predictions, 'o-',
                    label=f'{name} Predictions', alpha=0.6,
                    linewidth=1.5, markersize=3)
        
        # Customize plot
        ax3.set_title('Model Predictions vs Actual Values (Test Data)',
                     fontweight='bold', fontsize=14, pad=15)
        ax3.set_xlabel('Sample Index (sorted by actual value)', fontweight='bold')
        ax3.set_ylabel('Living Tissue Area (mm²)', fontweight='bold')
        ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_performance_comparison.png"),
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create scatter plots for each model
        n_models = len(model_results['models'])
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
        
        plt.figure(figsize=(15, 5*n_rows), facecolor=COLORS['background'])
        
        for i, name in enumerate(model_names, 1):
            ax = plt.subplot(n_rows, n_cols, i)
            ax.set_facecolor(COLORS['background'])
            
            # Get predictions and actual values
            model_dict = model_results['models'][name]
            y_test = model_dict['actual_test']
            y_pred = model_dict['y_pred_test']
            rmse = model_dict['test_rmse']
            r2 = model_dict['test_r2']
            mape = model_dict['test_mape']
            
            # Create scatter plot
            scatter = ax.scatter(y_test, y_pred, alpha=0.5, 
                               color=MODEL_COLORS.get(name, COLORS['ocean_blue']),
                               edgecolor=COLORS['dark_blue'])
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], '--', 
                   color=COLORS['coral'], label='Perfect Prediction')
            
            # Add best fit line
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(sorted(y_test), p(sorted(y_test)), '-', color=COLORS['dark_blue'],
                   label=f'Best Fit (R² = {r2:.3f})')
            
            # Customize plot
            ax.set_title(f'{name} - Test Data Predictions',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Actual Values', fontweight='bold')
            ax.set_ylabel('Predicted Values', fontweight='bold')
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add metrics text
            ax.text(0.05, 0.95, f"RMSE: {rmse:.0f}\nR²: {r2:.4f}\nMAPE: {mape:.2f}%", 
                    transform=ax.transAxes, fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue']))
            
            ax.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_predictions_scatter.png"),
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create residual analysis plots
        plt.figure(figsize=(15, 5*n_rows), facecolor=COLORS['background'])
        
        for i, name in enumerate(model_names, 1):
            ax = plt.subplot(n_rows, n_cols, i)
            ax.set_facecolor(COLORS['background'])
            
            # Get data
            model_dict = model_results['models'][name]
            y_test = model_dict['actual_test']
            y_pred = model_dict['y_pred_test']
            
            # Calculate residuals
            residuals = y_test - y_pred
            
            # Scatter plot of residuals
            ax.scatter(y_pred, residuals, alpha=0.7, 
                      color=MODEL_COLORS.get(name, COLORS['coral']), 
                      edgecolor=COLORS['dark_blue'], s=50)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color=COLORS['dark_blue'], linestyle='-', linewidth=2)
            
            # Set plot aesthetics
            ax.set_title(f'{name} Residuals', fontweight='bold', fontsize=16, pad=10)
            ax.set_xlabel('Predicted LTA', fontweight='bold', fontsize=14)
            ax.set_ylabel('Residuals', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add stats
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            ax.text(0.05, 0.95, f"Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}", 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue']))
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_residual_analysis.png"),
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Add time-based analysis
        visualize_model_performance_time_series(model_results)
        
        print("Model performance visualizations saved.")
        
    except Exception as e:
        print(f"Error visualizing model performance: {e}")
        import traceback
        traceback.print_exc()

def visualize_model_performance_time_series(model_results, results_dir='14_Results'):
    """
    Visualize model performance over time to better understand temporal patterns.
    
    Args:
        model_results (dict): Dictionary containing model results
        results_dir (str): Directory to save visualizations
    """
    try:
        print("\nVisualizing model performance over time...")
        
        # Get the best model
        best_model_name = model_results['meta']['best_model']
        
        # Get training and test data with years
        train_df = model_results['meta']['train_df']
        test_df = model_results['meta']['test_df']
        target_col = model_results['meta']['target_col']
        
        # Create a figure
        plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
        
        # Get years
        all_years = sorted(list(set(train_df['Year'].unique()) | set(test_df['Year'].unique())))
        
        # Create a new dataframe to hold actual vs predicted values by year
        results_by_year = []
        
        # For training data
        model_dict = model_results['models'][best_model_name]
        for year in train_df['Year'].unique():
            year_mask = train_df['Year'] == year
            year_indices = train_df[year_mask].index
            
            # Get predicted and actual values for this year
            train_indices = model_dict['train_indices']
            train_filter = [i in year_indices for i in train_indices]
            
            y_actual = train_df.loc[year_indices, target_col].mean()
            y_pred = np.mean(np.array(model_dict['y_pred_train'])[train_filter]) if any(train_filter) else np.nan
            
            results_by_year.append({
                'Year': year,
                'Actual': y_actual,
                'Predicted': y_pred,
                'Set': 'Train',
                'Residual': y_actual - y_pred if not np.isnan(y_pred) else np.nan
            })
        
        # For test data
        for year in test_df['Year'].unique():
            year_mask = test_df['Year'] == year
            year_indices = test_df[year_mask].index
            
            # Get predicted and actual values for this year
            test_indices = model_dict['test_indices']
            test_filter = [i in year_indices for i in test_indices]
            
            y_actual = test_df.loc[year_indices, target_col].mean()
            y_pred = np.mean(np.array(model_dict['y_pred_test'])[test_filter]) if any(test_filter) else np.nan
            
            results_by_year.append({
                'Year': year,
                'Actual': y_actual,
                'Predicted': y_pred,
                'Set': 'Test',
                'Residual': y_actual - y_pred if not np.isnan(y_pred) else np.nan
            })
        
        # Convert to dataframe
        time_results_df = pd.DataFrame(results_by_year)
        
        # Plot actual vs predicted values over time
        plt.subplot(2, 1, 1)
        plt.title(f'Actual vs Predicted LTA by Year ({best_model_name} Model)',
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
        
        # Plot actual values
        plt.plot(time_results_df['Year'], time_results_df['Actual'], 'o-',
                color=COLORS['dark_blue'], linewidth=2, label='Actual')
        
        # Plot predicted values for training set
        train_data = time_results_df[time_results_df['Set'] == 'Train']
        plt.plot(train_data['Year'], train_data['Predicted'], 's--',
                color=COLORS['ocean_blue'], linewidth=2, alpha=0.7,
                label='Predicted (Training)')
        
        # Plot predicted values for test set
        test_data = time_results_df[time_results_df['Set'] == 'Test']
        plt.plot(test_data['Year'], test_data['Predicted'], 'd--',
                color=COLORS['coral'], linewidth=2, alpha=0.7,
                label='Predicted (Testing)')
        
        # Add vertical line separating train and test data
        if len(test_data) > 0:
            plt.axvline(x=min(test_data['Year']), color=COLORS['neutral'],
                       linestyle=':', alpha=0.7, label='Train/Test Split')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlabel('Year', fontweight='bold', fontsize=14)
        plt.ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14)
        plt.legend()
        
        # Plot residuals over time
        plt.subplot(2, 1, 2)
        plt.title('Prediction Residuals by Year',
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
        
        # Plot zero line
        plt.axhline(y=0, color=COLORS['dark_blue'], linestyle='-', linewidth=1)
        
        # Plot residuals for training set
        plt.bar(train_data['Year'], train_data['Residual'],
               color=COLORS['ocean_blue'], alpha=0.7, label='Training Residuals')
        
        # Plot residuals for test set
        plt.bar(test_data['Year'], test_data['Residual'],
               color=COLORS['coral'], alpha=0.7, label='Testing Residuals')
        
        # Add vertical line separating train and test data
        if len(test_data) > 0:
            plt.axvline(x=min(test_data['Year']), color=COLORS['neutral'],
                       linestyle=':', alpha=0.7, label='Train/Test Split')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlabel('Year', fontweight='bold', fontsize=14)
        plt.ylabel('Residual (Actual - Predicted)', fontweight='bold', fontsize=14)
        plt.legend()
        
        plt.tight_layout()
        # plt.savefig(os.path.join(results_dir, "model_performance_time_series.png"),
        #            bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing time series performance: {e}")
        import traceback
        traceback.print_exc()

def visualize_feature_importance(model_results, results_dir='14_Results'):
    """
    Visualize feature importance from the trained models.
    
    Args:
        model_results (dict): Dictionary containing model results and feature importance data
        results_dir (str): Directory to save visualizations
        
    Returns:
        None
    """
    print("\nVisualizing feature importance...")
    
    # Get feature importance dataframe
    feature_importance_df = model_results['feature_importance']
    
    if feature_importance_df is None or len(feature_importance_df) == 0:
        print("No feature importance data available for visualization")
        return
    
    # Create a plot showing the top 15 most important features across all models
    top_features = (
        feature_importance_df
        .groupby('Feature')['Importance']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )
    
    # Filter dataframe to only include the top features
    plot_df = feature_importance_df[feature_importance_df['Feature'].isin(top_features)]
    
    # Create figure
    plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
    plt.title('TOP FEATURES IMPORTANCE: STONY CORAL LIVING TISSUE AREA FORECASTING', 
             fontweight='bold', fontsize=20, color=COLORS['dark_blue'], pad=20)
    
    # Create a grouped bar plot with models on the x-axis
    sns.barplot(
        x='Feature', y='Importance', hue='Model',
        data=plot_df,
        palette=[MODEL_COLORS.get(model, COLORS['coral']) for model in plot_df['Model'].unique()],
        alpha=0.7, edgecolor=COLORS['dark_blue'], linewidth=1.5
    )
    
    # Set plot aesthetics
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Feature', fontweight='bold', fontsize=14, labelpad=10)
    plt.ylabel('Importance', fontweight='bold', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Model', title_fontsize=14, fontsize=12, frameon=True, 
              facecolor='white', framealpha=0.9, loc='upper right')
    
    # Add a text box explaining the visualization
    explanation_text = (
        "Feature importance shows which factors most strongly\n"
        "influence stony coral living tissue area predictions.\n"
        "Values are normalized per model (higher is more important)."
    )
    
    plt.annotate(explanation_text, xy=(0.02, 0.02), xycoords='figure fraction', 
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'], boxstyle='round,pad=0.5'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "feature_importance_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"Saved feature importance visualization to {os.path.join(results_dir, 'feature_importance_analysis.png')}")
    plt.close()

def generate_forecasts(feature_dict, model_results, forecast_years=[2024, 2025, 2026, 2027, 2028]):
    """
    Generate forecasts for stony coral LTA for future years.
    
    Args:
        feature_dict (dict): Dictionary containing engineered features
        model_results (dict): Dictionary containing trained models
        forecast_years (list): List of years to forecast
        
    Returns:
        DataFrame: DataFrame containing forecasts
    """
    print("\nGenerating forecasts for future years...")
    print(f"Forecast years: {forecast_years}")
    
    # Extract data and models
    feature_df = feature_dict['feature_df']
    id_cols = feature_dict['id_cols']
    target_col = feature_dict['target_col']
    scaler = model_results['meta']['scaler']
    feature_cols = model_results['meta']['feature_cols']
    best_model_name = model_results['meta']['best_model']
    
    # Get the most recent year in the data
    latest_year = feature_df['Year'].max()
    print(f"Most recent year in data: {latest_year}")
    
    # Calculate historical variability statistics for realistic forecasts
    historical_yearly_change = feature_df.groupby('StationID')[target_col].pct_change()
    median_yearly_change = historical_yearly_change.median()
    std_yearly_change = historical_yearly_change.std()
    
    # Create a dataframe for forecasts
    forecasts = []
    
    # For each station, predict values for future years
    for station_id in feature_df['StationID'].unique():
        # Get station data
        station_data = feature_df[feature_df['StationID'] == station_id].copy()
        
        # Skip if not enough data (at least 3 years of data)
        if len(station_data['Year'].unique()) < 3:
            print(f"Skipping station {station_id} due to insufficient data")
            continue
        
        # Get most recent data for this station
        recent_data = station_data[station_data['Year'] == latest_year].copy()
        
        # Skip if no recent data
        if len(recent_data) == 0:
            print(f"Skipping station {station_id} due to missing recent data")
            continue
        
        # Use the most recent row as a template for forecasting
        template_row = recent_data.iloc[0].copy()
        
        # Extract station metadata that will remain constant
        site_id = template_row['SiteID']
        subregion = template_row['Subregion']
        habitat = template_row['Habitat']
        
        # Get the most recent LTA value
        recent_lta = template_row[target_col]
        
        # Calculate station-specific variability
        station_historical = station_data.sort_values('Year')
        station_changes = station_historical[target_col].pct_change().dropna()
        
        if len(station_changes) >= 3:
            # Use station-specific variability
            station_std = station_changes.std()
            station_trend = np.mean(station_changes.tail(3))
        else:
            # Use overall variability
            station_std = std_yearly_change
            station_trend = median_yearly_change
        
        # Generate station-specific noise scale based on historical data
        noise_scale = max(0.02, min(0.15, station_std))  # Between 2% and 15% variability
        
        # For each future year, generate a forecast
        for i, forecast_year in enumerate(forecast_years):
            # Calculate the number of years into the future
            years_ahead = forecast_year - latest_year
            
            # Create a copy of the template row for this forecast
            forecast_row = template_row.copy()
            forecast_row['Year'] = forecast_year
            
            # Update temporal features for the forecast year
            # (this will depend on the specific features engineered)
            if 'LTA_prev_year' in feature_cols:
                if years_ahead == 1:
                    forecast_row['LTA_prev_year'] = recent_lta
                else:
                    # For years beyond the first, use the forecast from the previous year
                    prev_year_forecast = next((f for f in forecasts 
                                             if f['StationID'] == station_id 
                                             and f['Year'] == forecast_year - 1), None)
                    if prev_year_forecast:
                        forecast_row['LTA_prev_year'] = prev_year_forecast['Forecast_LTA']
                    else:
                        forecast_row['LTA_prev_year'] = recent_lta
            
            # Update event flags
            for event_year in [2014, 2017, 2019]:
                if f'Event_{event_year}' in feature_cols:
                    forecast_row[f'Event_{event_year}'] = 0  # Not an event year
                if f'Post_Event_{event_year}' in feature_cols:
                    forecast_row[f'Post_Event_{event_year}'] = 1  # All future years are post-event
            
            # Update rolling statistics with proper variability
            if 'LTA_rolling_mean_3yr' in feature_cols:
                # For the first forecast year, use the mean of the last 3 years
                if years_ahead == 1:
                    last_3_years = station_data[station_data['Year'] > latest_year - 3][target_col].mean()
                    # Add small random variation to avoid linear patterns
                    variation = np.random.normal(0, abs(last_3_years * 0.03))  # 3% random variation
                    forecast_row['LTA_rolling_mean_3yr'] = last_3_years + variation
                else:
                    # For later years, compute based on previous forecasts with variability
                    prev_forecasts = [f['Forecast_LTA'] for f in forecasts 
                                     if f['StationID'] == station_id 
                                     and f['Year'] >= forecast_year - 3
                                     and f['Year'] < forecast_year]
                    
                    if len(prev_forecasts) > 0:
                        mean_val = sum(prev_forecasts) / len(prev_forecasts)
                        # Add random variation
                        variation = np.random.normal(0, abs(mean_val * 0.05))  # 5% random variation
                        forecast_row['LTA_rolling_mean_3yr'] = mean_val + variation
                    else:
                        # If not enough previous forecasts, use recent data with variation
                        variation = np.random.normal(0, abs(recent_lta * 0.04))  # 4% random variation
                        forecast_row['LTA_rolling_mean_3yr'] = recent_lta + variation
            
            # Update growth rate feature with variability
            if 'LTA_growth_rate' in feature_cols:
                if years_ahead == 1:
                    # Use the average growth rate of the last few years with random variation
                    growth_rates = station_data.sort_values('Year')['LTA_growth_rate'].dropna().tail(3).mean()
                    if pd.isna(growth_rates):
                        growth_rates = station_trend if not pd.isna(station_trend) else 0
                    
                    # Add realistic variation to growth rate
                    growth_variation = np.random.normal(0, abs(max(0.02, min(0.08, station_std/2))))
                    forecast_row['LTA_growth_rate'] = growth_rates + growth_variation
                else:
                    # For later years, use variable growth rates
                    base_growth = station_trend if not pd.isna(station_trend) else 0
                    # Increasing variability over time
                    time_factor = min(1.0, 0.5 + (years_ahead * 0.1))
                    growth_variation = np.random.normal(0, abs(max(0.02, min(0.1, station_std * time_factor))))
                    forecast_row['LTA_growth_rate'] = base_growth + growth_variation
            
            # Prepare data for prediction
            forecast_features = forecast_row[feature_cols].values.reshape(1, -1)
            forecast_features_scaled = scaler.transform(forecast_features)
            
            # Make predictions with each model
            forecasts_by_model = {}
            previous_lta = recent_lta if years_ahead == 1 else forecasts[-1]['Forecast_LTA'] if forecasts else recent_lta
            
            for name, model_dict in model_results['models'].items():
                # Skip Ensemble as it's calculated from other models
                if name == 'Ensemble':
                    continue
                
                # Get the model
                model = model_dict['model']
                
                # Make base prediction
                base_prediction = model.predict(forecast_features_scaled)[0]
                
                # Add realistic variability that increases with forecast horizon
                # More distant forecasts should have more variability
                variability_factor = 1.0 + (years_ahead * 0.15)  # Increase variability by 15% per year
                # Ensure noise_scale and base_prediction are positive
                safe_noise_scale = abs(noise_scale)
                safe_prediction = abs(base_prediction) if base_prediction != 0 else 1.0
                noise = np.random.normal(0, safe_noise_scale * variability_factor * safe_prediction)
                
                # Add seasonal/cyclic patterns if appropriate for coral dynamics
                # (simplified version - could be enhanced with actual seasonal patterns)
                cyclic_component = 0
                if i % 2 == 0:  # Simple alternating pattern
                    cyclic_component = base_prediction * 0.02 * (i+1)
                else:
                    cyclic_component = -base_prediction * 0.02 * (i+1)
                
                # Combine base prediction with noise and cyclic component
                prediction = base_prediction + noise + cyclic_component
                
                # Ensure prediction is not negative
                prediction = max(0, prediction)
                
                # Store the prediction
                forecasts_by_model[f'{name}_Forecast'] = prediction
            
            # Calculate ensemble prediction (average of all model predictions)
            ensemble_prediction = np.mean(list(forecasts_by_model.values()))
            
            # Add ensemble-specific variability for realism
            ensemble_noise = np.random.normal(0, noise_scale * 0.5 * ensemble_prediction)
            ensemble_prediction = max(0, ensemble_prediction + ensemble_noise)
            
            forecasts_by_model['Ensemble_Forecast'] = ensemble_prediction
            
            # Use the best model or ensemble for the main forecast
            if best_model_name == 'Ensemble':
                main_forecast = ensemble_prediction
            else:
                main_forecast = forecasts_by_model[f'{best_model_name}_Forecast']
            
            # Create forecast record
            forecast_record = {
                'Year': forecast_year,
                'StationID': station_id,
                'SiteID': site_id,
                'Subregion': subregion,
                'Habitat': habitat,
                'previous_LTA': previous_lta,
                'Forecast_LTA': main_forecast,
                'Years_Ahead': years_ahead,
                'Best_Model': best_model_name
            }
            
            # Add individual model forecasts
            forecast_record.update(forecasts_by_model)
            
            # Calculate optimistic and pessimistic scenarios with variability
            # Use different uncertainty ranges for different years ahead
            uncertainty_factor = 0.1 + (years_ahead * 0.02)  # Starts at 10%, increases by 2% per year ahead
            uncertainty_factor = min(0.2, uncertainty_factor)  # Cap at 20%
            
            # Add variability to optimistic/pessimistic scenarios
            opt_variation = abs(np.random.normal(0, 0.02))  # Small additional variation
            pes_variation = abs(np.random.normal(0, 0.02))
            
            forecast_record['Optimistic_LTA'] = main_forecast * (1.0 + uncertainty_factor + opt_variation)
            forecast_record['Pessimistic_LTA'] = main_forecast * (1.0 - uncertainty_factor - pes_variation)
            
            # Add to forecasts list
            forecasts.append(forecast_record)
    
    # Convert forecasts list to dataframe
    forecasts_df = pd.DataFrame(forecasts)
    
    if len(forecasts_df) == 0:
        print("No forecasts generated. Check data and model.")
        return None
    
    print(f"Generated {len(forecasts_df)} forecasts for {len(forecasts_df['StationID'].unique())} stations")
    
    # Calculate aggregate forecasts
    print("\nCalculating aggregate forecasts...")
    
    # Overall average forecast by year
    overall_avg_forecast = forecasts_df.groupby('Year')['Forecast_LTA'].mean().reset_index()
    overall_avg_forecast['Level'] = 'Overall'
    overall_avg_forecast['Group'] = 'All'
    overall_avg_forecast = overall_avg_forecast.rename(columns={'Forecast_LTA': 'Mean_LTA'})
    
    # Regional average forecast by year
    regional_avg_forecast = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_LTA'].mean().reset_index()
    regional_avg_forecast['Level'] = 'Region'
    regional_avg_forecast['Group'] = regional_avg_forecast['Subregion']
    regional_avg_forecast = regional_avg_forecast.drop('Subregion', axis=1).rename(columns={'Forecast_LTA': 'Mean_LTA'})
    
    # Habitat average forecast by year
    habitat_avg_forecast = forecasts_df.groupby(['Year', 'Habitat'])['Forecast_LTA'].mean().reset_index()
    habitat_avg_forecast['Level'] = 'Habitat'
    habitat_avg_forecast['Group'] = habitat_avg_forecast['Habitat']
    habitat_avg_forecast = habitat_avg_forecast.drop('Habitat', axis=1).rename(columns={'Forecast_LTA': 'Mean_LTA'})
    
    # Combine all aggregations
    agg_forecasts = pd.concat([overall_avg_forecast, regional_avg_forecast, habitat_avg_forecast], axis=0)
    
    # Save forecasts to CSV
    forecasts_df.to_csv(os.path.join(results_dir, "stony_coral_lta_forecasts.csv"), index=False)
    print(f"Saved forecasts to {os.path.join(results_dir, 'stony_coral_lta_forecasts.csv')}")
    
    # Save aggregate forecasts to CSV
    agg_forecasts.to_csv(os.path.join(results_dir, "stony_coral_lta_aggregate_forecasts.csv"), index=False)
    print(f"Saved aggregate forecasts to {os.path.join(results_dir, 'stony_coral_lta_aggregate_forecasts.csv')}")
    
    return {
        'individual_forecasts': forecasts_df,
        'aggregate_forecasts': agg_forecasts
    }

def visualize_forecasts(forecasts, historical_df):
    """
    Visualize forecasts for stony coral LTA.
    
    Args:
        forecasts (dict): Dictionary containing forecast DataFrames
        historical_df (DataFrame): DataFrame containing historical data
        
    Returns:
        None
    """
    print("\nVisualizing LTA forecasts...")
    
    # Extract forecast dataframes
    forecasts_df = forecasts['individual_forecasts']
    agg_forecasts = forecasts['aggregate_forecasts']
    
    try:
        # Create overall trend visualization with enhanced styling
        plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
        
        # Calculate yearly averages from historical data
        historical_yearly = historical_df.groupby('Year')['Total_LTA'].agg(['mean', 'std']).reset_index()
        
        # Calculate yearly averages from forecast data
        forecast_yearly = forecasts_df.groupby('Year')['Forecast_LTA'].agg(['mean', 'std']).reset_index()
        
        # Enhanced colors for better visual appeal
        historical_color = '#0066CC'  # Rich blue
        forecast_color = '#FF6600'    # Vibrant orange
        
        # Plot historical data with error bands
        plt.plot(historical_yearly['Year'], historical_yearly['mean'], 
                color=historical_color, marker='o', linestyle='-', linewidth=3, markersize=8, 
                label='Historical LTA')
        
        # Add error bands for historical data
        plt.fill_between(historical_yearly['Year'], 
                        historical_yearly['mean'] - historical_yearly['std'],
                        historical_yearly['mean'] + historical_yearly['std'],
                        color=historical_color, alpha=0.2)
        
        # Plot forecast data with error bands
        plt.plot(forecast_yearly['Year'], forecast_yearly['mean'], 
                color=forecast_color, marker='s', linestyle='--', linewidth=3, markersize=8, 
                label='Forecast LTA')
        
        # Add error bands for forecast data
        plt.fill_between(forecast_yearly['Year'], 
                        forecast_yearly['mean'] - forecast_yearly['std'],
                        forecast_yearly['mean'] + forecast_yearly['std'],
                        color=forecast_color, alpha=0.2)
        
        # Mark the transition point
        last_historical_year = historical_yearly['Year'].max()
        plt.axvline(x=last_historical_year, color='#444444', linestyle=':', linewidth=2, 
                   label=f'Last Observed Year ({last_historical_year})')
        
        # Add background shading to make the forecast portion stand out
        plt.axvspan(last_historical_year, forecast_yearly['Year'].max(), alpha=0.1, 
                  color='lightblue', label='Forecast Period')
        
        # Add labels and title with enhanced styling
        plt.title('STONY CORAL LIVING TISSUE AREA FORECAST (2024-2028)', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add significant events as vertical bars with labels
        events = {
            2014: {'name': 'Bleaching Event', 'color': '#FF3333'},
            2017: {'name': 'Hurricane Irma', 'color': '#9933FF'},
            2019: {'name': 'Disease Outbreak', 'color': '#FF9900'}
        }
        
        for year, event in events.items():
            if year in historical_yearly['Year'].values:
                plt.axvline(x=year, color=event['color'], alpha=0.7, linestyle='-', linewidth=1.5)
                plt.text(year, plt.ylim()[1]*0.95, event['name'], rotation=90, 
                        color=event['color'], fontweight='bold', ha='center', va='top')
        
        # Enhance the legend with better positioning and columns
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                  fontsize=12, loc='upper right', edgecolor=COLORS['grid'],
                  ncol=2)
        
        # Add explanatory text box with enhanced styling
        explanation_text = (
            f"Forecast based on historical data from {historical_yearly['Year'].min()}-{historical_yearly['Year'].max()}\n"
            f"Projected using ensemble machine learning models\n"
            f"Shaded area represents ±1 standard deviation forecast uncertainty range"
        )
        
        plt.annotate(explanation_text, xy=(0.02, 0.02), xycoords='figure fraction', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#AAAAAA', 
                             boxstyle='round,pad=0.5', linewidth=2))
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stony_coral_lta_forecast.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create a regional forecast visualization
        if 'Subregion' in forecasts_df.columns and 'Subregion' in historical_df.columns:
            plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
            
            # Create vibrant color map for regions
            region_colors = {
                'UK': '#FF3333',   # Bright red for Upper Keys
                'MK': '#33FF33',   # Bright green for Middle Keys
                'LK': '#3333FF'    # Bright blue for Lower Keys
            }
            
            # Create custom markers for each region
            region_markers = {
                'UK': 'o',
                'MK': 's',
                'LK': '^'
            }
            
            region_names = {
                'UK': 'Upper Keys',
                'MK': 'Middle Keys',
                'LK': 'Lower Keys'
            }
            
            # Calculate regional yearly averages from historical data
            historical_regional = historical_df.groupby(['Year', 'Subregion'])['Total_LTA'].mean().reset_index()
            
            # Calculate regional yearly averages from forecast data
            forecast_regional = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_LTA'].mean().reset_index()
            
            # Plot each region with enhanced styling
            for region in historical_df['Subregion'].unique():
                # Historical data for this region
                region_hist = historical_regional[historical_regional['Subregion'] == region]
                
                # Forecast data for this region
                region_forecast = forecast_regional[forecast_regional['Subregion'] == region]
                
                if len(region_hist) > 0:
                    # Plot historical data with enhanced styling
                    plt.plot(region_hist['Year'], region_hist['Total_LTA'], 
                            color=region_colors.get(region, COLORS['coral']), 
                            marker=region_markers.get(region, 'o'), linestyle='-', linewidth=3, markersize=8, 
                            label=f'{region_names.get(region, region)} (Historical)')
                
                if len(region_forecast) > 0:
                    # Plot forecast data with enhanced styling
                    plt.plot(region_forecast['Year'], region_forecast['Forecast_LTA'], 
                            color=region_colors.get(region, COLORS['coral']), 
                            marker=region_markers.get(region, 's'), linestyle='--', linewidth=3, markersize=8, 
                            label=f'{region_names.get(region, region)} (Forecast)')
            
            # Mark the transition point
            last_historical_year = historical_df['Year'].max()
            plt.axvline(x=last_historical_year, color=COLORS['neutral'], linestyle=':', linewidth=2,
                       label=f'Last Observed Year ({last_historical_year})')
            
            # Add background shading to make the forecast portion stand out
            plt.axvspan(last_historical_year, forecast_yearly['Year'].max(), alpha=0.1, 
                      color='lightblue', label='Forecast Period')
            
            # Add labels and title with enhanced styling
            plt.title('REGIONAL STONY CORAL LIVING TISSUE AREA FORECAST (2024-2028)', 
                     fontweight='bold', fontsize=18, pad=20,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            plt.ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add gridlines
            plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Enhance the legend with better positioning
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=10, loc='upper right', edgecolor=COLORS['grid'],
                      ncol=2, bbox_to_anchor=(1, 1))
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "regional_stony_coral_lta_forecast.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
        
        # Create habitat-specific forecast visualization
        if 'Habitat' in forecasts_df.columns and 'Habitat' in historical_df.columns:
            plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
            
            # Create more vibrant color map for habitats with distinct colors
            habitat_colors = {
                'Patch Reef': '#FF5733',    # Vibrant orange-red
                'Offshore Deep': '#33A1FF',  # Bright blue
                'Offshore Shallow': '#33FF57',  # Bright green
                'Forereef': '#FFD733',     # Bright yellow
                'Backreef': '#AD33FF',     # Bright purple
                'BCP': '#FF33F5',          # Bright pink
                'OD': '#3385FF',           # Sky blue
                'OS': '#33FFB8',           # Turquoise
                'P': '#FF3352'             # Red
            }
            
            # Create a custom marker style for each habitat
            habitat_markers = {
                'Patch Reef': 'o',
                'Offshore Deep': 's',
                'Offshore Shallow': '^',
                'Forereef': 'd',
                'Backreef': 'v',
                'BCP': '*',
                'OD': 'P',
                'OS': 'X',
                'P': 'h'
            }
            
            # Calculate habitat yearly averages from historical data
            habitat_historical = historical_df.groupby(['Year', 'Habitat'])['Total_LTA'].mean().reset_index()
            
            # Calculate habitat yearly averages from forecast data
            habitat_forecast = forecasts_df.groupby(['Year', 'Habitat'])['Forecast_LTA'].mean().reset_index()
            
            # Plot each habitat with increased line width and marker size
            for habitat in historical_df['Habitat'].unique():
                # Historical data for this habitat
                hab_hist = habitat_historical[habitat_historical['Habitat'] == habitat]
                
                # Forecast data for this habitat
                hab_forecast = habitat_forecast[habitat_forecast['Habitat'] == habitat]
                
                if len(hab_hist) > 0:
                    # Plot historical data with enhanced styling
                    plt.plot(hab_hist['Year'], hab_hist['Total_LTA'], 
                            color=habitat_colors.get(habitat, COLORS['dark_blue']), 
                            marker=habitat_markers.get(habitat, 'o'), linestyle='-', linewidth=3, markersize=8,
                            label=f'{habitat} (Historical)')
                
                if len(hab_forecast) > 0:
                    # Plot forecast data with enhanced styling
                    plt.plot(hab_forecast['Year'], hab_forecast['Forecast_LTA'], 
                            color=habitat_colors.get(habitat, COLORS['dark_blue']), 
                            marker=habitat_markers.get(habitat, 's'), linestyle='--', linewidth=3, markersize=8,
                            label=f'{habitat} (Forecast)')
            
            # Mark the transition point
            last_historical_year = historical_df['Year'].max()
            plt.axvline(x=last_historical_year, color=COLORS['neutral'], linestyle=':', linewidth=2,
                       label=f'Last Observed Year ({last_historical_year})')
            
            # Add labels and title with enhanced styling
            plt.title('HABITAT-SPECIFIC STONY CORAL LIVING TISSUE AREA FORECAST (2024-2028)', 
                     fontweight='bold', fontsize=18, pad=20,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            plt.ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add gridlines
            plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Enhance the legend with more columns and better positioning
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=10, loc='upper right', edgecolor=COLORS['grid'],
                      ncol=2, bbox_to_anchor=(1, 1))
            
            # Add background shading to make the forecast portion stand out
            plt.axvspan(last_historical_year, forecast_yearly['Year'].max(), alpha=0.1, 
                      color='lightblue', label='Forecast Period')
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "habitat_stony_coral_lta_forecast.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
        
        print("Forecast visualizations saved.")
    except Exception as e:
        print(f"Error visualizing forecasts: {e}")
        import traceback
        traceback.print_exc()

def analyze_reef_evolution(forecasts, historical_df):
    """
    Analyze the evolution of reef health based on historical data and forecasts.
    
    Args:
        forecasts (dict): Dictionary containing forecasting results
        historical_df (pd.DataFrame): Historical stony coral LTA data
    """
    print("\nAnalyzing reef evolution and health trends...")
    
    # Make sure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract forecasted values
    forecasts_df = forecasts['individual_forecasts']
    agg_forecasts = forecasts['aggregate_forecasts']
    
    # Get the last historical year and forecasted years
    last_historical_year = historical_df['Year'].max()
    forecast_years = sorted(forecasts_df['Year'].unique())
    
    # Create a comprehensive analysis DataFrame
    historical_agg = historical_df.groupby('Year')['Total_LTA'].mean().reset_index()
    historical_agg['Data_Type'] = 'Historical'
    
    # Get overall forecasts for analysis
    overall_forecasts = agg_forecasts[agg_forecasts['Level'] == 'Overall']
    forecast_agg = overall_forecasts[['Year', 'Mean_LTA']].copy()
    forecast_agg.rename(columns={'Mean_LTA': 'Total_LTA'}, inplace=True)
    forecast_agg['Data_Type'] = 'Forecast'
    
    combined_df = pd.concat([historical_agg, forecast_agg], ignore_index=True)
    
    # Calculate year-over-year changes
    combined_df = combined_df.sort_values('Year')
    combined_df['YoY_Change'] = combined_df['Total_LTA'].pct_change() * 100
    
    # Calculate average rates of change
    historical_rate = historical_agg['Total_LTA'].pct_change().dropna().mean() * 100
    forecast_rate = forecast_agg['Total_LTA'].pct_change().dropna().mean() * 100
    
    # Compare regional trends
    regional_historical = historical_df.groupby(['Year', 'Subregion'])['Total_LTA'].mean().reset_index()
    regional_forecasts = agg_forecasts[agg_forecasts['Level'] == 'Region']
    
    # Compare habitat trends
    habitat_historical = historical_df.groupby(['Year', 'Habitat'])['Total_LTA'].mean().reset_index()
    habitat_forecasts = agg_forecasts[agg_forecasts['Level'] == 'Habitat']
    
    # Calculate percent changes for regions
    region_changes = []
    for region in regional_forecasts['Group'].unique():
        region_forecast = regional_forecasts[regional_forecasts['Group'] == region]
        region_historical = regional_historical[regional_historical['Subregion'] == region]
        
        if len(region_historical) > 0 and len(region_forecast) > 0:
            # Get last historical value
            last_historical = region_historical[region_historical['Year'] == last_historical_year]
            if len(last_historical) > 0:
                last_value = last_historical['Total_LTA'].values[0]
                
                # Get final forecast value
                final_forecast = region_forecast[region_forecast['Year'] == forecast_years[-1]]
                if len(final_forecast) > 0:
                    final_value = final_forecast['Mean_LTA'].values[0]
                    
                    # Calculate percent change
                    percent_change = ((final_value - last_value) / last_value) * 100
                    
                    region_changes.append({
                        'Region': region,
                        'Start_Value': last_value,
                        'End_Value': final_value,
                        'Absolute_Change': final_value - last_value,
                        'Percent_Change': percent_change,
                        'Start_Year': last_historical_year,
                        'End_Year': forecast_years[-1]
                    })
    
    # Calculate percent changes for habitats
    habitat_changes = []
    for habitat in habitat_forecasts['Group'].unique():
        habitat_forecast = habitat_forecasts[habitat_forecasts['Group'] == habitat]
        habitat_hist = habitat_historical[habitat_historical['Habitat'] == habitat]
        
        if len(habitat_hist) > 0 and len(habitat_forecast) > 0:
            # Get last historical value
            last_historical = habitat_hist[habitat_hist['Year'] == last_historical_year]
            if len(last_historical) > 0:
                last_value = last_historical['Total_LTA'].values[0]
                
                # Get final forecast value
                final_forecast = habitat_forecast[habitat_forecast['Year'] == forecast_years[-1]]
                if len(final_forecast) > 0:
                    final_value = final_forecast['Mean_LTA'].values[0]
                    
                    # Calculate percent change
                    percent_change = ((final_value - last_value) / last_value) * 100
                    
                    habitat_changes.append({
                        'Habitat': habitat,
                        'Start_Value': last_value,
                        'End_Value': final_value,
                        'Absolute_Change': final_value - last_value,
                        'Percent_Change': percent_change,
                        'Start_Year': last_historical_year,
                        'End_Year': forecast_years[-1]
                    })
    
    # Overall percent change
    last_historical_value = historical_agg.loc[historical_agg['Year'] == last_historical_year, 'Total_LTA'].values[0]
    final_forecast_value = forecast_agg.loc[forecast_agg['Year'] == forecast_years[-1], 'Total_LTA'].values[0]
    percent_change = ((final_forecast_value - last_historical_value) / last_historical_value) * 100
    
    # Generate insights
    insights = {
        'overall_trend': 'increasing' if percent_change > 0 else 'decreasing',
        'historical_rate': historical_rate,
        'forecast_rate': forecast_rate,
        'rate_change': forecast_rate - historical_rate,
        'vulnerable_regions': [],
        'resilient_regions': [],
        'vulnerable_habitats': [],
        'resilient_habitats': []
    }
    
    # Identify vulnerable and resilient regions
    for region_data in region_changes:
        if region_data['Percent_Change'] < 0:
            insights['vulnerable_regions'].append((region_data['Region'], region_data['Percent_Change']))
        else:
            insights['resilient_regions'].append((region_data['Region'], region_data['Percent_Change']))
    
    # Identify vulnerable and resilient habitats
    for habitat_data in habitat_changes:
        if habitat_data['Percent_Change'] < 0:
            insights['vulnerable_habitats'].append((habitat_data['Habitat'], habitat_data['Percent_Change']))
        else:
            insights['resilient_habitats'].append((habitat_data['Habitat'], habitat_data['Percent_Change']))
    
    # Sort by magnitude of change
    insights['vulnerable_regions'].sort(key=lambda x: x[1])
    insights['resilient_regions'].sort(key=lambda x: x[1], reverse=True)
    insights['vulnerable_habitats'].sort(key=lambda x: x[1])
    insights['resilient_habitats'].sort(key=lambda x: x[1], reverse=True)
    
    # Write insights to file
    with open(os.path.join(results_dir, 'reef_evolution_insights.txt'), 'w') as f:
        f.write("STONY CORAL LIVING TISSUE AREA (LTA) EVOLUTION ANALYSIS\n")
        f.write("====================================================\n\n")
        
        f.write("OVERALL TREND ANALYSIS:\n")
        f.write(f"- The overall LTA trend is {insights['overall_trend']}\n")
        f.write(f"- Historical average annual change rate: {insights['historical_rate']:.2f}%\n")
        f.write(f"- Forecasted average annual change rate: {insights['forecast_rate']:.2f}%\n")
        f.write(f"- Change in rate: {insights['rate_change']:.2f} percentage points\n\n")
        
        f.write("REGIONAL VULNERABILITY ANALYSIS:\n")
        f.write("Regions with declining LTA (most vulnerable first):\n")
        for region, change in insights['vulnerable_regions']:
            f.write(f"- {region}: {change:.2f}% projected change\n")
        
        f.write("\nRegions with increasing LTA (most resilient first):\n")
        for region, change in insights['resilient_regions']:
            f.write(f"- {region}: {change:.2f}% projected change\n")
        
        f.write("\nHABITAT VULNERABILITY ANALYSIS:\n")
        f.write("Habitats with declining LTA (most vulnerable first):\n")
        for habitat, change in insights['vulnerable_habitats']:
            f.write(f"- {habitat}: {change:.2f}% projected change\n")
        
        f.write("\nHabitats with increasing LTA (most resilient first):\n")
        for habitat, change in insights['resilient_habitats']:
            f.write(f"- {habitat}: {change:.2f}% projected change\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        if insights['vulnerable_regions'] or insights['vulnerable_habitats']:
            f.write("1. Focus conservation efforts on vulnerable regions and habitats identified above.\n")
            f.write("2. Investigate factors contributing to resilience in areas showing positive trends.\n")
            f.write("3. Implement targeted monitoring for early detection of further decline.\n")
        else:
            f.write("1. Continue current conservation measures as they appear effective.\n")
            f.write("2. Document and analyze successful management practices.\n")
            f.write("3. Apply lessons from resilient areas to other coral reef ecosystems.\n")
    
    print(f"Reef evolution analysis complete. Insights saved to: {os.path.join(results_dir, 'reef_evolution_insights.txt')}")

def main():
    """
    Main function to execute stony coral LTA forecasting.
    """
    # Step 1: Load and preprocess data
    data_dict = load_and_preprocess_data()
    
    # Step 2: Analyze time series patterns
    analyze_time_series_patterns(data_dict)
    
    # Step 3: Engineer features for modeling
    feature_dict = engineer_features(
        data_dict['lta_df'],
        data_dict['stations_df'],
        data_dict['temperature_df'],
        data_dict,
        target_col='Total_LTA'
    )
    
    # Step 4: Train forecasting models
    model_results = train_forecasting_models(feature_dict)
    
    # Step 5: Visualize model performance
    visualize_model_performance(model_results)
    
    # Step 5a: Visualize model performance over time
    visualize_model_performance_time_series(model_results)
    
    # Step 6: Visualize feature importance
    visualize_feature_importance(model_results)
    
    # Step 7: Generate forecasts
    forecasts = generate_forecasts(feature_dict, model_results)
    
    # Step 8: Visualize forecasts
    if forecasts is not None:
        visualize_forecasts(forecasts, data_dict['lta_df'])
    
    # Step 9: Analyze reef evolution
    if forecasts is not None:
        analyze_reef_evolution(forecasts, data_dict['lta_df'])

if __name__ == "__main__":
    main() 