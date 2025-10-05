"""
11_stony_coral_cover_forecasting.py - Forecasting Stony Coral Percentage Cover Evolution

This script develops forecasting models to predict the evolution of stony coral percentage 
cover across different monitoring stations over the next five years (2024-2028). It builds on 
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
import warnings
import math
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
results_dir = "11_Results"
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
    Load and preprocess the CREMP datasets for forecasting analysis.
    
    Returns:
        dict: Dictionary containing preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the percentage cover data
        print("Loading percentage cover data...")
        pcover_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_TaxaGroups.csv")
        print(f"Percentage cover data loaded successfully with {len(pcover_df)} rows")
        print(f"First few columns: {pcover_df.columns.tolist()[:5]}")
        
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
        
        # Load stony coral LTA data for correlation
        print("\nLoading stony coral LTA data...")
        stony_lta_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_LTA.csv")
        print(f"Stony coral LTA data loaded successfully with {len(stony_lta_df)} rows")
        print(f"First few columns: {stony_lta_df.columns.tolist()[:5]}")
        
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
        pcover_df['Date'] = pd.to_datetime(pcover_df['Date'])
        pcover_df['Year'] = pcover_df['Year'].astype(int)
        
        if 'Date' in stony_density_df.columns:
            stony_density_df['Date'] = pd.to_datetime(stony_density_df['Date'])
        
        if 'Date' in stony_lta_df.columns:
            stony_lta_df['Date'] = pd.to_datetime(stony_lta_df['Date'])
        
        # Focus on stony coral percentage cover
        print("\nExtracting stony coral cover data...")
        if 'StonyCoralPcover' in pcover_df.columns:
            pcover_df['Stony_Coral_Cover'] = pcover_df['StonyCoralPcover']
            print("Found StonyCoralPcover column")
        else:
            print("StonyCoralPcover column not found. Available columns:", pcover_df.columns.tolist())
            # Try to find alternative column
            coral_columns = [col for col in pcover_df.columns if 'coral' in col.lower() or 'cover' in col.lower()]
            if coral_columns:
                print(f"Found potential coral columns: {coral_columns}")
                pcover_df['Stony_Coral_Cover'] = pcover_df[coral_columns[0]]
            else:
                print("No suitable column found. Using placeholder values.")
                pcover_df['Stony_Coral_Cover'] = 0.0
        
        # Print basic statistics
        print("\nBasic statistics for Stony Coral Cover:")
        print(pcover_df['Stony_Coral_Cover'].describe())
        
        # Get time range
        print(f"\nData spans from {pcover_df['Year'].min()} to {pcover_df['Year'].max()}")
        
        # Create a data dictionary to hold all processed dataframes
        data_dict = {
            'pcover_df': pcover_df,
            'stations_df': stations_df,
            'stony_density_df': stony_density_df,
            'stony_lta_df': stony_lta_df,
            'temp_df': temp_df
        }
        
        print("\nData preprocessing complete.")
        return data_dict
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def analyze_time_series_patterns(data_dict):
    """
    Analyze temporal patterns in stony coral percentage cover data.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        
    Returns:
        dict: Dictionary containing time series analysis results
    """
    print("\nAnalyzing time series patterns in stony coral percentage cover...")
    
    # Extract percentage cover data
    pcover_df = data_dict['pcover_df']
    
    # Create a yearly average time series
    yearly_avg = pcover_df.groupby('Year')['Stony_Coral_Cover'].mean().reset_index()
    print(f"Created yearly average time series with {len(yearly_avg)} points")
    
    # Create regional yearly averages
    regional_yearly_avg = pcover_df.groupby(['Year', 'Subregion'])['Stony_Coral_Cover'].mean().reset_index()
    
    # Create habitat yearly averages
    habitat_yearly_avg = pcover_df.groupby(['Year', 'Habitat'])['Stony_Coral_Cover'].mean().reset_index()
    
    # Perform stationarity test on overall yearly average
    print("\nPerforming Augmented Dickey-Fuller test for stationarity...")
    adf_result = adfuller(yearly_avg['Stony_Coral_Cover'])
    adf_output = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4],
        'Is Stationary': adf_result[1] < 0.05
    }
    print(f"ADF Test Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'} (p-value: {adf_result[1]:.4f})")
    
    # Calculate autocorrelation and partial autocorrelation
    print("\nCalculating autocorrelation and partial autocorrelation...")
    max_lags = min(10, len(yearly_avg) - 1)
    acf_values = acf(yearly_avg['Stony_Coral_Cover'], nlags=max_lags)
    pacf_values = pacf(yearly_avg['Stony_Coral_Cover'], nlags=max_lags)
    
    # Identify change points
    print("\nIdentifying significant change points...")
    
    # Calculate year-to-year percentage changes
    yearly_avg['pct_change'] = yearly_avg['Stony_Coral_Cover'].pct_change() * 100
    
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
    ax1.plot(yearly_avg['Year'], yearly_avg['Stony_Coral_Cover'], 
             marker='o', linestyle='-', color=COLORS['coral'], 
             linewidth=2.5, markersize=8, label='Annual Mean Cover')
    
    # Fit a linear trend line
    X = yearly_avg['Year'].values.reshape(-1, 1)
    y = yearly_avg['Stony_Coral_Cover'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    y_pred = model.predict(X)
    
    # Plot the trend line
    ax1.plot(yearly_avg['Year'], y_pred, '--', color=COLORS['dark_blue'], 
             linewidth=2, label=f'Linear Trend (Slope: {slope:.4f} per year)')
    
    # Mark significant change points
    for _, row in change_points.iterrows():
        ax1.scatter(row['Year'], row['Stony_Coral_Cover'], 
                   s=150, color='red', zorder=5, 
                   marker='*', label='_nolegend_')
    
    # Mark significant events with vertical lines
    for year, event_info in REEF_EVENTS.items():
        if year >= yearly_avg['Year'].min() and year <= yearly_avg['Year'].max():
            ax1.axvline(x=year, color=event_info['color'], linestyle=':', alpha=0.7)
            ax1.text(year, ax1.get_ylim()[1] * 0.95, event_info['name'], 
                    rotation=90, verticalalignment='top', 
                    color=event_info['color'], fontweight='bold')
    
    # Set plot aesthetics
    ax1.set_title('STONY CORAL PERCENT COVER TIME SERIES (1996-2023)', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 2: Time series by region
    ax2 = plt.subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['background'])
    
    # Create a color map for regions
    region_colors = {
        'UK': COLORS['dark_blue'],   # Upper Keys
        'MK': COLORS['ocean_blue'],  # Middle Keys
        'LK': COLORS['light_blue']   # Lower Keys
    }
    
    region_names = {
        'UK': 'Upper Keys',
        'MK': 'Middle Keys',
        'LK': 'Lower Keys'
    }
    
    # Plot each region
    for region in regional_yearly_avg['Subregion'].unique():
        region_data = regional_yearly_avg[regional_yearly_avg['Subregion'] == region]
        ax2.plot(region_data['Year'], region_data['Stony_Coral_Cover'], 
                marker='o', linestyle='-', 
                color=region_colors.get(region, COLORS['coral']), 
                linewidth=2, markersize=6, 
                label=region_names.get(region, region))
    
    # Set plot aesthetics
    ax2.set_title('Regional Trends in Percent Cover', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax2.set_ylabel('Mean Percent Cover', fontweight='bold', fontsize=12, labelpad=8)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 3: Time series by habitat
    ax3 = plt.subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['background'])
    
    # Create a color map for habitats
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Plot each habitat
    for habitat in habitat_yearly_avg['Habitat'].unique():
        habitat_data = habitat_yearly_avg[habitat_yearly_avg['Habitat'] == habitat]
        ax3.plot(habitat_data['Year'], habitat_data['Stony_Coral_Cover'], 
                marker='o', linestyle='-', 
                color=habitat_colors.get(habitat, COLORS['coral']), 
                linewidth=2, markersize=6, 
                label=habitat_names.get(habitat, habitat))
    
    # Set plot aesthetics
    ax3.set_title('Habitat-Specific Trends in Percent Cover', 
                 fontweight='bold', fontsize=16, pad=15)
    ax3.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax3.set_ylabel('Mean Percent Cover', fontweight='bold', fontsize=12, labelpad=8)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 4: Autocorrelation function
    ax4 = plt.subplot(gs[2, 0])
    ax4.set_facecolor(COLORS['background'])
    
    # Plot ACF
    lags = range(len(acf_values))
    ax4.bar(lags, acf_values, width=0.3, color=COLORS['ocean_blue'], 
           edgecolor=COLORS['dark_blue'], alpha=0.7)
    
    # Add confidence intervals
    confidence = 1.96 / np.sqrt(len(yearly_avg))
    ax4.axhline(y=confidence, linestyle='--', color='red', alpha=0.7)
    ax4.axhline(y=-confidence, linestyle='--', color='red', alpha=0.7)
    
    # Set plot aesthetics
    ax4.set_title('Autocorrelation Function (ACF)', 
                 fontweight='bold', fontsize=16, pad=15)
    ax4.set_xlabel('Lag (Years)', fontweight='bold', fontsize=12, labelpad=8)
    ax4.set_ylabel('Autocorrelation', fontweight='bold', fontsize=12, labelpad=8)
    ax4.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 5: Partial Autocorrelation function
    ax5 = plt.subplot(gs[2, 1])
    ax5.set_facecolor(COLORS['background'])
    
    # Plot PACF
    lags = range(len(pacf_values))
    ax5.bar(lags, pacf_values, width=0.3, color=COLORS['reef_green'], 
           edgecolor=COLORS['dark_blue'], alpha=0.7)
    
    # Add confidence intervals
    confidence = 1.96 / np.sqrt(len(yearly_avg))
    ax5.axhline(y=confidence, linestyle='--', color='red', alpha=0.7)
    ax5.axhline(y=-confidence, linestyle='--', color='red', alpha=0.7)
    
    # Set plot aesthetics
    ax5.set_title('Partial Autocorrelation Function (PACF)', 
                 fontweight='bold', fontsize=16, pad=15)
    ax5.set_xlabel('Lag (Years)', fontweight='bold', fontsize=12, labelpad=8)
    ax5.set_ylabel('Partial Autocorrelation', fontweight='bold', fontsize=12, labelpad=8)
    ax5.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 6: Yearly percentage change
    ax6 = plt.subplot(gs[3, 0])
    ax6.set_facecolor(COLORS['background'])
    
    # Plot yearly percentage change
    bars = ax6.bar(yearly_avg['Year'][1:], yearly_avg['pct_change'][1:], 
                  color=[COLORS['coral'] if x >= 0 else COLORS['dark_blue'] for x in yearly_avg['pct_change'][1:]], 
                  alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add threshold lines
    ax6.axhline(y=std_threshold, linestyle='--', color='red', alpha=0.7, 
               label=f'±{std_threshold:.1f}% (2σ threshold)')
    ax6.axhline(y=-std_threshold, linestyle='--', color='red', alpha=0.7)
    
    # Set plot aesthetics
    ax6.set_title('Year-to-Year Percentage Change in Coral Cover', 
                 fontweight='bold', fontsize=16, pad=15)
    ax6.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax6.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=12, labelpad=8)
    ax6.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax6.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 7: Stationarity test results
    ax7 = plt.subplot(gs[3, 1])
    ax7.set_facecolor(COLORS['background'])
    ax7.set_axis_off()  # Turn off axis
    
    # Create a text box with stationarity test results
    stationarity_text = (
        "STATIONARITY TEST RESULTS (ADF Test):\n\n"
        f"ADF Statistic: {adf_output['ADF Statistic']:.4f}\n"
        f"p-value: {adf_output['p-value']:.4f}\n\n"
        "Critical Values:\n"
        f"  1%: {adf_output['Critical Values']['1%']:.4f}\n"
        f"  5%: {adf_output['Critical Values']['5%']:.4f}\n"
        f"  10%: {adf_output['Critical Values']['10%']:.4f}\n\n"
        f"Interpretation: {'Stationary' if adf_output['Is Stationary'] else 'Non-stationary'}\n"
        f"(The time series {'rejects' if adf_output['Is Stationary'] else 'fails to reject'} the null hypothesis of a unit root)"
    )
    
    # Add the stationarity results box with enhanced styling
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box in the center
    ax7.text(0.5, 0.5, stationarity_text, fontsize=12, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', 
           bbox=props, transform=ax7.transAxes)
    
    # Add a summary of the time series analysis
    fig.text(0.5, 0.01, 
            "Time Series Analysis Summary: The stony coral percent cover shows overall declining trends with significant changes following major disturbance events.", 
            ha='center', va='center', fontsize=12, fontstyle='italic', 
            fontweight='bold', color=COLORS['text'])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(results_dir, "stony_coral_cover_time_series_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close(fig)
    
    print("Time series analysis visualization saved.")
    
    # Return analysis results
    analysis_results = {
        'yearly_avg': yearly_avg,
        'regional_yearly_avg': regional_yearly_avg,
        'habitat_yearly_avg': habitat_yearly_avg,
        'adf_output': adf_output,
        'acf_values': acf_values,
        'pacf_values': pacf_values,
        'change_points': change_points,
        'trend_slope': slope
    }
    
    return analysis_results

def visualize_feature_importance(df):
    """
    Create a visualization showing the correlation of features with the target variable.
    
    Args:
        df (DataFrame): DataFrame containing the features and target
    """
    print("\nAnalyzing feature importance...")
    
    # Calculate correlation with target
    target = 'Stony_Coral_Cover'
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    # Remove the target itself from correlation analysis
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # Calculate correlation
    corr_with_target = [df[col].corr(df[target]) for col in numeric_cols]
    
    # Create a DataFrame for visualization
    corr_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Correlation': corr_with_target
    })
    
    # Sort by absolute correlation
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(20)
    
    # Create the visualization
    plt.figure(figsize=(12, 10), facecolor=COLORS['background'])
    
    # Create a bar plot
    bars = plt.barh(
        corr_df['Feature'], 
        corr_df['Correlation'], 
        color=[COLORS['ocean_blue'] if x > 0 else COLORS['coral'] for x in corr_df['Correlation']],
        edgecolor=COLORS['dark_blue'],
        alpha=0.7,
        linewidth=1.5
    )
    
    # Add a vertical line at 0
    plt.axvline(x=0, color=COLORS['grid'], linestyle='--', alpha=0.7)
    
    # Set plot aesthetics
    plt.title('TOP 20 FEATURES BY CORRELATION WITH STONY CORAL COVER', 
             fontweight='bold', fontsize=18, color=COLORS['dark_blue'],
             pad=20, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    plt.xlabel('Correlation Coefficient', fontweight='bold', fontsize=14, labelpad=10)
    plt.ylabel('Feature', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add a grid
    plt.grid(axis='x', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.07
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', ha='left' if width > 0 else 'right', 
                fontsize=10, fontweight='bold', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "stony_coral_cover_feature_importance.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Feature importance visualization saved.")

def engineer_features(cover_data, stations_df, temperature_data, target_col='Stony_Coral_Cover'):
    """
    Engineer features for stony coral cover forecasting.
    
    Args:
        cover_data (pd.DataFrame): Percentage cover data
        stations_df (pd.DataFrame): Station information data
        temperature_data (pd.DataFrame): Temperature data
        target_col (str): Target column name
    
    Returns:
        dict: Dictionary containing engineered features
    """
    print("\nEngineering features for forecasting models...")
    
    # Create a copy of the cover data
    df = cover_data.copy()
    
    # Ensure the target column exists
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found. Using Stony_coral as target column.")
        if 'Stony_coral' in df.columns:
            target_col = 'Stony_coral'
        else:
            print("Error: Neither the specified target column nor 'Stony_coral' is available.")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError("Target column not found in data")
        
    # Rename to standardize the column name
    df = df.rename(columns={target_col: 'Stony_Coral_Cover'})
    target_col = 'Stony_Coral_Cover'
    
    # 1. Temporal features
    print("Creating temporal features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Date'].dt.month.map(lambda m: 'Winter' if m in [12, 1, 2] else
                                         'Spring' if m in [3, 4, 5] else
                                         'Summer' if m in [6, 7, 8] else 'Fall')
    
    # 2. Create lagged features
    print("Creating lagged features...")
    # Group by site to create site-specific lags
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Date')
        
        # Create 1-year and 2-year lags for the target variable
        if len(site_data) > 1:
            df.loc[site_mask, 'Lag1_Coral_Cover'] = site_data['Stony_Coral_Cover'].shift(1)
            if len(site_data) > 2:
                df.loc[site_mask, 'Lag2_Coral_Cover'] = site_data['Stony_Coral_Cover'].shift(2)
    
    # Calculate rolling averages
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Date')
        
        if len(site_data) > 2:
            df.loc[site_mask, 'RollingAvg2_Coral_Cover'] = site_data['Stony_Coral_Cover'].rolling(window=2, min_periods=1).mean().values
    
    # 3. Generate spatial features - check available columns
    print("Adding spatial features...")
    
    # Check which columns exist in stations_df
    print(f"Available station columns: {stations_df.columns.tolist()}")
    available_spatial_columns = []
    
    # Define potential spatial columns to use
    spatial_columns = ['StationID']
    
    # Check which other spatial columns are available
    for col in ['Longitude', 'Latitude', 'Depth_m', 'Depth', 'X_COORD', 'Y_COORD']:
        if col in stations_df.columns:
            spatial_columns.append(col)
            available_spatial_columns.append(col)
    
    if 'Subregion' not in df.columns and 'Subregion' in stations_df.columns:
        spatial_columns.append('Subregion')
    
    if 'Habitat' not in df.columns and 'Habitat' in stations_df.columns:
        spatial_columns.append('Habitat')
    
    # If no additional spatial columns found, use what's available
    if len(available_spatial_columns) == 0:
        print("Warning: No spatial coordinates found in stations_df. Using available columns.")
        # Use the columns that exist in both DataFrames for the merge
        common_cols = [col for col in stations_df.columns if col in ['StationID', 'SiteID', 'Site_name']]
        if len(common_cols) > 0:
            # Merge only on common columns
            print(f"Merging on common columns: {common_cols}")
            # Use left merge to keep all rows from df
            if len(common_cols) == 1 and common_cols[0] == 'StationID':
                df = df.merge(stations_df, on='StationID', how='left', suffixes=('', '_station'))
            else:
                # Pick the first common column for merge
                df = df.merge(stations_df, on=common_cols[0], how='left', suffixes=('', '_station'))
        else:
            print("Warning: No common columns found for merge. Skipping spatial feature addition.")
    else:
        # Merge with station data to get spatial information
        print(f"Merging with spatial columns: {spatial_columns}")
        try:
            df = df.merge(stations_df[spatial_columns], on='StationID', how='left', suffixes=('', '_station'))
        except KeyError as e:
            print(f"Error merging with stations_df: {e}")
            print("Available columns in stations_df:", stations_df.columns.tolist())
            # Fallback to basic merge
            print("Falling back to basic merge on StationID only")
            df = df.merge(stations_df[['StationID']], on='StationID', how='left')
    
    # 4. Add temperature data
    print("Adding temperature data...")
    # Check if temperature data exists
    if temperature_data is None:
        print("No temperature data provided. Skipping temperature features.")
    else:
        # Need temperature data by year, site, region
        if 'Temperature' in temperature_data.columns:
            print(f"Temperature data has {len(temperature_data)} rows")
            
            # Create site-specific temperature features
            site_temps = {}
            
            try:
                for site in df['Site_name'].unique():
                    site_mask = temperature_data['Site_name'] == site
                    if site_mask.any():
                        site_temp_data = temperature_data[site_mask]
                        
                        # Calculate temperature statistics by year
                        temp_stats = site_temp_data.groupby(site_temp_data['Date'].dt.year).agg({
                            'Temperature': ['mean', 'max', 'min', 'std']
                        })
                        
                        temp_stats.columns = ['_'.join(col).strip() for col in temp_stats.columns.values]
                        temp_stats = temp_stats.reset_index().rename(columns={'Date': 'Year'})
                        
                        site_temps[site] = temp_stats
            except Exception as e:
                print(f"Error processing temperature data: {e}")
                print("Continuing without temperature features")
                site_temps = {}
            
            # Add temperature features to each site in main dataframe
            for site, temp_df in site_temps.items():
                site_mask = df['Site_name'] == site
                if site_mask.any():
                    try:
                        # Merge temperature data with site data
                        df_site = df[site_mask].merge(temp_df, on='Year', how='left')
                        
                        # Update the main dataframe
                        for col in temp_df.columns:
                            if col != 'Year':
                                df.loc[site_mask, col] = df_site[col].values
                    except Exception as e:
                        print(f"Error adding temperature data for site {site}: {e}")
        else:
            print("Temperature column not found in temperature data.")
            # Check what columns are available
            print(f"Available columns: {temperature_data.columns}")
            
            # Try to see if we can use another column for temperature
            if 'Bottom_Temperature_degC' in temperature_data.columns:
                print("Using Bottom_Temperature_degC instead")
                try:
                    temperature_data['Temperature'] = temperature_data['Bottom_Temperature_degC']
                    
                    # Create site-specific temperature features
                    site_temps = {}
                    
                    for site in df['Site_name'].unique():
                        site_mask = temperature_data['Site_name'] == site
                        if site_mask.any():
                            site_temp_data = temperature_data[site_mask]
                            
                            # Calculate temperature statistics by year
                            temp_stats = site_temp_data.groupby(site_temp_data['Date'].dt.year).agg({
                                'Temperature': ['mean', 'max', 'min', 'std']
                            })
                            
                            temp_stats.columns = ['_'.join(col).strip() for col in temp_stats.columns.values]
                            temp_stats = temp_stats.reset_index().rename(columns={'Date': 'Year'})
                            
                            site_temps[site] = temp_stats
                    
                    # Add temperature features to each site in main dataframe
                    for site, temp_df in site_temps.items():
                        site_mask = df['Site_name'] == site
                        if site_mask.any():
                            # Merge temperature data with site data
                            df_site = df[site_mask].merge(temp_df, on='Year', how='left')
                            
                            # Update the main dataframe
                            for col in temp_df.columns:
                                if col != 'Year':
                                    df.loc[site_mask, col] = df_site[col].values
                except Exception as e:
                    print(f"Error processing temperature data: {e}")
                    print("Continuing without temperature features")
            else:
                print("No suitable temperature column found. Continuing without temperature features.")
    
    # 5. One-hot encode categorical variables
    print("One-hot encoding categorical variables...")
    # One-hot encode Season
    season_dummies = pd.get_dummies(df['Season'], prefix='Season')
    df = pd.concat([df, season_dummies], axis=1)
    
    # One-hot encode Subregion if it exists
    if 'Subregion' in df.columns:
        subregion_dummies = pd.get_dummies(df['Subregion'], prefix='Subregion')
        df = pd.concat([df, subregion_dummies], axis=1)
    
    # One-hot encode Habitat if it exists
    if 'Habitat' in df.columns:
        habitat_dummies = pd.get_dummies(df['Habitat'], prefix='Habitat')
        df = pd.concat([df, habitat_dummies], axis=1)
    
    # Fill missing values
    print(f"Missing values before imputation: {df.isna().sum().sum()}")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    feature_dict = {
        'feature_df': df,
        'target_col': target_col
    }
    
    print(f"Engineered feature dataframe has {len(df)} rows and {len(df.columns)} features")
    return feature_dict

def visualize_model_performance(model_results):
    """
    Visualize model performance comparisons and test predictions.
    
    Args:
        model_results (dict): Dictionary containing model results and metrics
    """
    print("\nVisualizing model performance comparisons...")
    
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
        
        for name, metrics in model_results['metrics'].items():
            model_names.append(name)
            train_r2.append(metrics['train_r2'])
            test_r2.append(metrics['test_r2'])
        
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
        
        for name, metrics in model_results['metrics'].items():
            train_rmse.append(metrics['train_rmse'])
            test_rmse.append(metrics['test_rmse'])
        
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
        
        # Get test data predictions for each model
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        
        # Sort test data by actual values for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test.iloc[sort_idx]
        
        # Plot actual values
        ax3.plot(range(len(y_test)), y_test_sorted, 'o-', 
                label='Actual Values', color=COLORS['dark_blue'],
                linewidth=2, markersize=4)
        
        # Plot predictions for each model
        for name, model in model_results['models'].items():
            y_pred = model.predict(X_test)
            y_pred_sorted = y_pred[sort_idx]
            ax3.plot(range(len(y_test)), y_pred_sorted, 'o-',
                    label=f'{name} Predictions', alpha=0.6,
                    linewidth=1.5, markersize=3)
        
        # Customize plot
        ax3.set_title('Model Predictions vs Actual Values (Test Data)',
                     fontweight='bold', fontsize=14, pad=15)
        ax3.set_xlabel('Sample Index (sorted by actual value)', fontweight='bold')
        ax3.set_ylabel('Coral Cover', fontweight='bold')
        ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_performance_comparison.png"),
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create scatter plots for each model
        n_models = len(model_results['models'])
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        plt.figure(figsize=(15, 5*n_rows), facecolor=COLORS['background'])
        
        for i, (name, model) in enumerate(model_results['models'].items(), 1):
            ax = plt.subplot(n_rows, n_cols, i)
            ax.set_facecolor(COLORS['background'])
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create scatter plot
            ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS['ocean_blue'])
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], '--', 
                   color=COLORS['coral'], label='Perfect Prediction')
            
            # Add best fit line
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(y_test, p(y_test), '-', color=COLORS['dark_blue'],
                   label=f'Best Fit (R² = {r2_score(y_test, y_pred):.3f})')
            
            # Customize plot
            ax.set_title(f'{name} - Test Data Predictions',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Actual Values', fontweight='bold')
            ax.set_ylabel('Predicted Values', fontweight='bold')
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_predictions_scatter.png"),
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Model performance visualizations saved.")
        
    except Exception as e:
        print(f"Error visualizing model performance: {e}")
        import traceback
        traceback.print_exc()

def train_forecasting_models(feature_dict, test_size=0.2, random_state=42):
    """
    Train and evaluate forecasting models.
    
    Args:
        feature_dict (dict): Dictionary with engineered features
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with trained models and evaluation metrics
    """
    print("\nTraining forecasting models...")
    
    # Get the feature dataframe and target column
    df = feature_dict['feature_df']
    target_col = feature_dict['target_col']
    
    # Print dataframe info for debugging
    print(f"Feature dataframe has {len(df)} rows and {len(df.columns)} columns")
    print(f"Target column: {target_col}")
    
    # Split data into training and testing sets - keep latest years for testing
    df = df.sort_values('Date')
    train_size = int(len(df) * (1 - test_size))
    
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"Training data: {len(df_train)} rows (years {df_train['Year'].min()}-{df_train['Year'].max()})")
    print(f"Testing data: {len(df_test)} rows (years {df_test['Year'].min()}-{df_test['Year'].max()})")
    
    # Define features to exclude from modeling - added 'Stony_coral' and 'Octocoral' to prevent data leakage
    exclude_cols = ['Date', 'Site_name', 'StationID', 'SiteID', 'Stony_Coral_Cover', 
                     'Subregion', 'Habitat', 'Season', target_col, 
                     'Stony_coral', 'Octocoral']  # Exclude these columns to prevent data leakage
    
    # Prepare X and y for training and testing sets
    X_train = df_train.drop(columns=[col for col in exclude_cols if col in df_train.columns])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[col for col in exclude_cols if col in df_test.columns])
    y_test = df_test[target_col]
    
    # Select only numeric columns for features
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train = X_train[numeric_cols]
    X_test = X_test[numeric_cols]
    
    # Verify there are no NaN values
    print(f"NaN values in X_train: {X_train.isna().sum().sum()}")
    print(f"NaN values in X_test: {X_test.isna().sum().sum()}")
    
    # If any NaN values remain, fill them with the mean
    if X_train.isna().sum().sum() > 0:
        print("Filling remaining NaN values with mean")
        X_train = X_train.fillna(X_train.mean())
    
    if X_test.isna().sum().sum() > 0:
        print("Filling remaining NaN values in test data with mean")
        X_test = X_test.fillna(X_train.mean())  # Use training mean for test data
    
    # Print feature names for debugging
    print(f"Features used: {X_train.columns.tolist()}")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Dictionary to store trained models and metrics
    results = {
        'models': {},
        'metrics': {},
        'feature_importance': {},
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"{name} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        
        # Store model and metrics
        results['models'][name] = model
        results['metrics'][name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        # Extract feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'][name] = feature_importance
            
            # Print top 10 important features
            print(f"\nTop 10 important features for {name}:")
            print(feature_importance.head(10))
        
        # For linear regression, use coefficients as feature importance
        elif name == 'Linear Regression':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'][name] = feature_importance
            
            # Print top 10 important features
            print(f"\nTop 10 important features for {name}:")
            print(feature_importance.head(10))
    
    # Identify the best model based on test R²
    best_model_name = max(results['metrics'], key=lambda k: results['metrics'][k]['test_r2'])
    results['best_model'] = {
        'name': best_model_name,
        'model': results['models'][best_model_name],
        'metrics': results['metrics'][best_model_name]
    }
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model test R²: {results['metrics'][best_model_name]['test_r2']:.4f}")
    print(f"Best model test RMSE: {results['metrics'][best_model_name]['test_rmse']:.4f}")
    
    # Add visualization of model performance
    visualize_model_performance(results)
    
    return results

def generate_forecasts(feature_dict, model_results, forecast_years=[2024, 2025, 2026, 2027, 2028]):
    """
    Generate forecasts for future years using the best trained model.
    
    Args:
        feature_dict (dict): Dictionary with engineered features
        model_results (dict): Dictionary with trained models and metrics
        forecast_years (list): List of years to forecast
        
    Returns:
        DataFrame: DataFrame with forecasts for each site and year
    """
    print("\nGenerating forecasts for future years...")
    
    try:
        # Get the best model
        best_model_name = model_results['best_model']['name']
        best_model = model_results['best_model']['model']
        print(f"Using {best_model_name} to generate forecasts")
        
        # Get the feature dataframe and extract unique sites and last year data
        df = feature_dict['feature_df']
        target_col = feature_dict['target_col']
        
        # Get the most recent year in the data
        last_year = df['Year'].max()
        print(f"Most recent year in data: {last_year}")
        
        # Get unique site names
        sites = df['Site_name'].unique()
        print(f"Generating forecasts for {len(sites)} sites")
        
        # Create a DataFrame to store forecasts
        forecasts = []
        
        # For each site, generate forecasts for future years
        for site in sites:
            # Get the last few years of data for this site to establish trends
            site_data = df[df['Site_name'] == site].sort_values('Date').tail(5).copy()
            
            if len(site_data) == 0:
                print(f"Warning: No data for site {site}")
                continue
            
            # Calculate site-specific trend and variability
            if len(site_data) > 1:
                site_trend = np.polyfit(range(len(site_data)), site_data[target_col], 1)[0]
                site_std = site_data[target_col].std()
            else:
                site_trend = 0
                site_std = df[target_col].std() * 0.1  # Use 10% of overall std as fallback
            
            # Get the most recent values for lagged features
            last_value = site_data[target_col].iloc[-1]
            last_lag1 = site_data['Lag1_Coral_Cover'].iloc[-1] if 'Lag1_Coral_Cover' in site_data.columns else last_value
            last_rolling_avg = site_data['RollingAvg2_Coral_Cover'].iloc[-1] if 'RollingAvg2_Coral_Cover' in site_data.columns else last_value
            
            # For each forecast year, predict the coral cover
            current_value = last_value
            current_lag1 = last_lag1
            current_rolling_avg = last_rolling_avg
            
            for year in forecast_years:
                # Create a copy of the most recent data for this forecast year
                forecast_data = site_data.iloc[-1:].copy()
                
                # Update the year
                forecast_data['Year'] = year
                
                # Create a new date
                try:
                    last_date = forecast_data['Date'].iloc[0]
                    if pd.isna(last_date) or last_date is None:
                        new_date = pd.Timestamp(year=int(year), month=7, day=1)
                    else:
                        month = int(last_date.month)
                        day = int(last_date.day)
                        new_date = pd.Timestamp(year=int(year), month=month, day=day)
                except Exception as e:
                    print(f"Error creating date for site {site}: {e}")
                    new_date = pd.Timestamp(year=int(year), month=7, day=1)
                
                forecast_data['Date'] = new_date
                
                # Update lagged features with previous predictions
                if 'Lag1_Coral_Cover' in forecast_data.columns:
                    forecast_data['Lag1_Coral_Cover'] = current_value
                
                if 'Lag2_Coral_Cover' in forecast_data.columns:
                    forecast_data['Lag2_Coral_Cover'] = current_lag1
                
                if 'RollingAvg2_Coral_Cover' in forecast_data.columns:
                    forecast_data['RollingAvg2_Coral_Cover'] = current_rolling_avg
                
                # Select feature columns for prediction
                exclude_cols = ['Date', 'Site_name', 'StationID', 'SiteID', target_col, 
                               'Subregion', 'Habitat', 'Season']
                
                X_pred = forecast_data.drop(columns=[col for col in exclude_cols if col in forecast_data.columns])
                
                # Select only numeric columns for features
                numeric_cols = X_pred.select_dtypes(include=['float64', 'int64']).columns
                X_pred = X_pred[numeric_cols]
                
                # Make sure X_pred has all columns used during training
                training_cols = model_results['X_train'].columns
                for col in training_cols:
                    if col not in X_pred.columns:
                        X_pred[col] = 0
                
                # Ensure column order matches training data
                X_pred = X_pred[training_cols]
                
                # Fill NaN values with 0
                X_pred = X_pred.fillna(0)
                
                try:
                    # Make base prediction
                    base_prediction = best_model.predict(X_pred)[0]
                    
                    # Add trend and random variation
                    random_factor = np.random.normal(0, site_std * 0.5)  # Reduced randomness
                    prediction = base_prediction + site_trend + random_factor
                    
                    # Ensure prediction is within reasonable bounds (0-100%)
                    prediction = max(0, min(1, prediction))
                except Exception as e:
                    print(f"Error making prediction for site {site}: {e}")
                    continue
                
                # Create forecast entry
                forecast_entry = {
                    'Site_name': site,
                    'StationID': site_data['StationID'].iloc[0] if 'StationID' in site_data.columns else "Unknown",
                    'Year': year,
                    'Date': new_date,
                    'Forecast_Coral_Cover': prediction,
                    'Last_Observed_Cover': last_value,
                    'Last_Observed_Year': last_year
                }
                
                # Add region and habitat information
                if 'Subregion' in site_data.columns:
                    forecast_entry['Subregion'] = site_data['Subregion'].iloc[0]
                else:
                    forecast_entry['Subregion'] = "Unknown"
                    
                if 'Habitat' in site_data.columns:
                    forecast_entry['Habitat'] = site_data['Habitat'].iloc[0]
                else:
                    forecast_entry['Habitat'] = "Unknown"
                
                forecasts.append(forecast_entry)
                
                # Update current values for next iteration
                current_lag1 = current_value
                current_value = prediction
                current_rolling_avg = (current_value + current_lag1) / 2
        
        # Convert to DataFrame
        forecasts_df = pd.DataFrame(forecasts)
        
        if len(forecasts_df) == 0:
            print("No forecasts could be generated!")
            return pd.DataFrame()
        
        # Calculate summary statistics
        yearly_summary = forecasts_df.groupby('Year')['Forecast_Coral_Cover'].agg(['mean', 'min', 'max', 'std']).reset_index()
        print("\nForecast Summary by Year:")
        print(yearly_summary)
        
        if 'Subregion' in forecasts_df.columns:
            region_summary = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Coral_Cover'].mean().reset_index()
            print("\nForecast Summary by Region:")
            print(region_summary)
        
        # Create visualizations
        visualize_forecasts(forecasts_df, feature_dict['feature_df'])
        
        return forecasts_df
    except Exception as e:
        print(f"Error generating forecasts: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def visualize_forecasts(forecasts_df, historical_df):
    """
    Visualize forecasts for future years.
    
    Args:
        forecasts_df (DataFrame): DataFrame with forecasts
        historical_df (DataFrame): DataFrame with historical data
    """
    print("\nVisualizing forecasts...")
    
    try:
        # Create overall trend visualization
        plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
        
        # Calculate yearly averages from historical data
        historical_yearly = historical_df.groupby('Year')['Stony_Coral_Cover'].agg(['mean', 'std']).reset_index()
        
        # Calculate yearly averages from forecast data
        forecast_yearly = forecasts_df.groupby('Year')['Forecast_Coral_Cover'].agg(['mean', 'std']).reset_index()
        
        # Plot historical data with error bands
        plt.plot(historical_yearly['Year'], historical_yearly['mean'], 
                color=COLORS['dark_blue'], marker='o', linestyle='-', linewidth=2, 
                label='Historical Coral Cover')
        
        # Add error bands for historical data
        plt.fill_between(historical_yearly['Year'], 
                        historical_yearly['mean'] - historical_yearly['std'],
                        historical_yearly['mean'] + historical_yearly['std'],
                        color=COLORS['dark_blue'], alpha=0.2)
        
        # Plot forecast data with error bands
        plt.plot(forecast_yearly['Year'], forecast_yearly['mean'], 
                color=COLORS['coral'], marker='s', linestyle='--', linewidth=2, 
                label='Forecast Coral Cover')
        
        # Add error bands for forecast data
        plt.fill_between(forecast_yearly['Year'], 
                        forecast_yearly['mean'] - forecast_yearly['std'],
                        forecast_yearly['mean'] + forecast_yearly['std'],
                        color=COLORS['coral'], alpha=0.2)
        
        # Mark the transition point
        last_historical_year = historical_yearly['Year'].max()
        plt.axvline(x=last_historical_year, color=COLORS['neutral'], linestyle=':', 
                   label=f'Last Observed Year ({last_historical_year})')
        
        # Add labels and title
        plt.title('STONY CORAL COVER FORECAST (2024-2028)', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Enhance the legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                  fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stony_coral_cover_forecast.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create a regional forecast visualization
        if 'Subregion' in forecasts_df.columns and 'Subregion' in historical_df.columns:
            plt.figure(figsize=(14, 8), facecolor=COLORS['background'])
            
            # Calculate regional yearly averages from historical data
            historical_regional = historical_df.groupby(['Year', 'Subregion'])['Stony_Coral_Cover'].mean().reset_index()
            
            # Calculate regional yearly averages from forecast data
            forecast_regional = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Coral_Cover'].mean().reset_index()
            
            # Create color map for regions
            region_colors = {
                'UK': COLORS['dark_blue'],   # Upper Keys
                'MK': COLORS['ocean_blue'],  # Middle Keys
                'LK': COLORS['light_blue']   # Lower Keys
            }
            
            region_names = {
                'UK': 'Upper Keys',
                'MK': 'Middle Keys',
                'LK': 'Lower Keys'
            }
            
            # Plot each region
            for region in historical_df['Subregion'].unique():
                # Historical data for this region
                region_hist = historical_regional[historical_regional['Subregion'] == region]
                
                # Forecast data for this region
                region_forecast = forecast_regional[forecast_regional['Subregion'] == region]
                
                # Plot historical data
                plt.plot(region_hist['Year'], region_hist['Stony_Coral_Cover'], 
                        color=region_colors.get(region, COLORS['coral']), 
                        marker='o', linestyle='-', linewidth=2, 
                        label=f'{region_names.get(region, region)} (Historical)')
                
                # Plot forecast data
                plt.plot(region_forecast['Year'], region_forecast['Forecast_Coral_Cover'], 
                        color=region_colors.get(region, COLORS['coral']), 
                        marker='s', linestyle='--', linewidth=2, 
                        label=f'{region_names.get(region, region)} (Forecast)')
            
            # Mark the transition point
            last_historical_year = historical_df['Year'].max()
            plt.axvline(x=last_historical_year, color=COLORS['neutral'], linestyle=':', 
                       label=f'Last Observed Year ({last_historical_year})')
            
            # Add labels and title
            plt.title('REGIONAL STONY CORAL COVER FORECAST (2024-2028)', 
                     fontweight='bold', fontsize=18, pad=20,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add gridlines
            plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Enhance the legend
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
            
            # Save the visualization
            plt.tight_layout()
            # plt.savefig(os.path.join(results_dir, "stony_coral_cover_regional_forecast.png"), 
            #            bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
        
        print("Forecast visualizations saved.")
    except Exception as e:
        print(f"Error visualizing forecasts: {e}")
        import traceback
        traceback.print_exc()

def visualize_feature_importance(model_results, results_dir='11_Results'):
    """
    Creates a detailed visualization of feature importance for all models.
    
    Args:
        model_results (dict): Dictionary containing model results and feature importance
        results_dir (str): Directory to save results
    """
    try:
        print("Creating detailed feature importance visualizations...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Extract feature importance data
        feature_importance_data = {}
        for model_name, importance_df in model_results['feature_importance'].items():
            # Get top 15 features
            top_features = importance_df.head(15).copy()
            feature_importance_data[model_name] = top_features
        
        # Create a figure with subplots for each model
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor=COLORS['background'])
        axes = axes.flatten()
        
        for i, (model_name, importance_df) in enumerate(feature_importance_data.items()):
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            # Plot horizontal bar chart
            ax = axes[i]
            ax.set_facecolor(COLORS['background'])
            
            # Create color gradient based on importance
            max_importance = importance_df['importance'].max()
            colors = [plt.cm.viridis(x/max_importance) for x in importance_df['importance']]
            
            # Plot bars
            bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
            
            # Add values to the end of each bar
            for bar in bars:
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', va='center', fontsize=10, color='black')
            
            # Set title and labels
            ax.set_title(f'Top Features - {model_name}', fontsize=16, 
                         fontweight='bold', color=COLORS['dark_blue'],
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12, fontweight='bold')
            
            # Customize grid
            ax.grid(True, linestyle='--', alpha=0.6, color=COLORS['grid'])
            
            # Get the best R² score
            best_r2 = model_results['metrics'][model_name]['test_r2']
            ax.text(0.5, -0.1, f'Test R² Score: {best_r2:.4f}', 
                    transform=ax.transAxes, ha='center', fontsize=14, 
                    fontweight='bold', color=COLORS['dark_blue'])
        
        # Add overall title
        plt.suptitle('CORAL REEF PREDICTION - FEATURE IMPORTANCE ANALYSIS', 
                     fontsize=20, fontweight='bold', color=COLORS['dark_blue'], 
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                     y=0.98)
        
        # Add subtitle with explanation
        plt.figtext(0.5, 0.92, 
                   'Features with higher importance have greater influence on coral cover predictions', 
                   fontsize=14, ha='center', color=COLORS['dark_blue'])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(results_dir, 'feature_importance_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
        print("Feature importance visualization saved.")
    except Exception as e:
        print(f"Error in feature importance visualization: {e}")
        import traceback
        traceback.print_exc()

def analyze_reef_evolution(forecasts_df, historical_df, results_dir='11_Results'):
    """
    Performs a detailed analysis of coral reef evolution over time and predicts future trends.
    
    Args:
        forecasts_df (pd.DataFrame): DataFrame with forecast data
        historical_df (pd.DataFrame): DataFrame with historical data
        results_dir (str): Directory to save results
    """
    try:
        print("\nPerforming detailed coral reef evolution analysis...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 1. Calculate growth rates and evolution metrics
        print("Calculating growth rates and evolution metrics...")
        
        # Yearly changes in historical data
        yearly_avg = historical_df.groupby('Year')['Stony_Coral_Cover'].mean().reset_index()
        yearly_avg['Annual_Change'] = yearly_avg['Stony_Coral_Cover'].pct_change() * 100
        yearly_avg['5yr_Rolling_Avg'] = yearly_avg['Stony_Coral_Cover'].rolling(window=5, min_periods=1).mean()
        
        # Regional yearly changes
        regional_yearly = historical_df.groupby(['Year', 'Subregion'])['Stony_Coral_Cover'].mean().reset_index()
        regional_pivot = regional_yearly.pivot(index='Year', columns='Subregion', values='Stony_Coral_Cover').reset_index()
        
        # Calculate regional growth rates
        for region in regional_yearly['Subregion'].unique():
            regional_pivot[f'{region}_Change'] = regional_pivot[region].pct_change() * 100
        
        # Habitat specific yearly changes
        habitat_yearly = historical_df.groupby(['Year', 'Habitat'])['Stony_Coral_Cover'].mean().reset_index()
        
        # Future trend estimation
        forecast_yearly = forecasts_df.groupby('Year')['Forecast_Coral_Cover'].mean().reset_index()
        forecast_regional = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Coral_Cover'].mean().reset_index()
        
        # 2. Create evolution trend visualization
        print("Creating evolution trend visualization...")
        
        # Create a figure for evolution trends
        plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
        
        # Plot historical data
        plt.plot(yearly_avg['Year'], yearly_avg['Stony_Coral_Cover'], 
                marker='o', linestyle='-', color=COLORS['coral'], 
                linewidth=2.5, markersize=8, label='Historical Coral Cover')
        
        # Plot 5-year rolling average
        plt.plot(yearly_avg['Year'], yearly_avg['5yr_Rolling_Avg'], 
                linestyle='--', color=COLORS['light_blue'], 
                linewidth=2.5, label='5-Year Rolling Average')
        
        # Add forecast data
        plt.plot(forecast_yearly['Year'], forecast_yearly['Forecast_Coral_Cover'], 
                marker='s', linestyle='-', color=COLORS['dark_blue'], 
                linewidth=3, markersize=10, label='Forecast (2024-2028)')
        
        # Fill the area between the lines for visual effect
        last_historical_year = yearly_avg['Year'].max()
        last_historical_cover = yearly_avg[yearly_avg['Year'] == last_historical_year]['Stony_Coral_Cover'].values[0]
        
        # Add forecast range (uncertainty)
        forecast_years = forecast_yearly['Year'].tolist()
        forecast_means = forecast_yearly['Forecast_Coral_Cover'].tolist()
        
        # Calculate forecast confidence intervals (20% range as a simplified example)
        forecast_std = forecasts_df.groupby('Year')['Forecast_Coral_Cover'].std().reset_index()
        forecast_upper = forecast_yearly['Forecast_Coral_Cover'] + 1.28 * forecast_std['Forecast_Coral_Cover']
        forecast_lower = forecast_yearly['Forecast_Coral_Cover'] - 1.28 * forecast_std['Forecast_Coral_Cover']
        forecast_lower = forecast_lower.clip(lower=0)  # Ensure no negative values
        
        # Plot confidence interval
        plt.fill_between(forecast_years, forecast_lower, forecast_upper, 
                        color=COLORS['dark_blue'], alpha=0.2, 
                        label='80% Confidence Interval')
        
        # Mark the transition point
        plt.axvline(x=last_historical_year, color='gray', linestyle=':')
        plt.text(last_historical_year + 0.2, plt.ylim()[1] * 0.9, 
                f"Forecast\nStart", fontsize=12, color='gray')
        
        # Highlight important events
        for year, event_info in REEF_EVENTS.items():
            if year >= yearly_avg['Year'].min() and year <= yearly_avg['Year'].max():
                plt.axvline(x=year, color=event_info['color'], linestyle=':', alpha=0.7)
                plt.text(year, plt.ylim()[1] * 0.95, event_info['name'], 
                        rotation=90, verticalalignment='top', 
                        color=event_info['color'], fontweight='bold')
        
        # Add trend lines
        # Historical trend
        X_hist = yearly_avg['Year'].values.reshape(-1, 1)
        y_hist = yearly_avg['Stony_Coral_Cover'].values
        model_hist = LinearRegression().fit(X_hist, y_hist)
        hist_slope = model_hist.coef_[0]
        
        # Future trend
        X_future = np.array(forecast_years).reshape(-1, 1)
        y_future = np.array(forecast_means)
        model_future = LinearRegression().fit(X_future, y_future)
        future_slope = model_future.coef_[0]
        
        # Plot trends
        hist_years = np.linspace(yearly_avg['Year'].min(), yearly_avg['Year'].max(), 100)
        plt.plot(hist_years, model_hist.predict(hist_years.reshape(-1, 1)), 
                '--', color='green', linewidth=1.5, 
                label=f'Historical Trend ({hist_slope:.5f} per year)')
        
        future_years = np.linspace(forecast_years[0], forecast_years[-1], 100)
        plt.plot(future_years, model_future.predict(future_years.reshape(-1, 1)), 
                '--', color='red', linewidth=1.5, 
                label=f'Forecast Trend ({future_slope:.5f} per year)')
        
        # Set title and labels
        plt.title('CORAL REEF EVOLUTION ANALYSIS (1996-2028)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                fontsize=12, loc='upper left', edgecolor=COLORS['grid'])
        
        # Add annotations
        # Calculate overall change
        first_cover = yearly_avg.iloc[0]['Stony_Coral_Cover']
        last_cover = yearly_avg.iloc[-1]['Stony_Coral_Cover']
        projected_cover = forecast_yearly.iloc[-1]['Forecast_Coral_Cover']
        
        overall_change = ((last_cover - first_cover) / first_cover) * 100
        projected_change = ((projected_cover - last_cover) / last_cover) * 100
        
        # Add annotation box
        info_text = (
            f"Historical Period (1996-2023):\n"
            f"• Starting Cover: {first_cover:.3f}\n"
            f"• Ending Cover: {last_cover:.3f}\n"
            f"• Net Change: {overall_change:.1f}%\n"
            f"• Annual Growth Rate: {hist_slope:.5f}\n\n"
            f"Forecast Period (2024-2028):\n"
            f"• Projected Cover (2028): {projected_cover:.3f}\n"
            f"• Projected Change: {projected_change:.1f}%\n"
            f"• Forecast Growth Rate: {future_slope:.5f}"
        )
        
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                    fontsize=11, verticalalignment='bottom')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "coral_reef_evolution_analysis.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # 3. Create regional evolution visualization
        print("Creating regional evolution visualization...")
        
        # Create a figure
        plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
        
        # Color mapping for regions
        region_colors = {
            'UK': COLORS['dark_blue'],   # Upper Keys
            'MK': COLORS['ocean_blue'],  # Middle Keys
            'LK': COLORS['light_blue']   # Lower Keys
        }
        
        region_names = {
            'UK': 'Upper Keys',
            'MK': 'Middle Keys',
            'LK': 'Lower Keys'
        }
        
        # Plot each region's historical data and forecast
        for region in regional_yearly['Subregion'].unique():
            # Historical data
            region_data = regional_yearly[regional_yearly['Subregion'] == region]
            plt.plot(region_data['Year'], region_data['Stony_Coral_Cover'], 
                    marker='o', linestyle='-', color=region_colors[region], 
                    linewidth=2, markersize=6, 
                    label=f'{region_names[region]} (Historical)')
            
            # Forecast data
            region_forecast = forecast_regional[forecast_regional['Subregion'] == region]
            plt.plot(region_forecast['Year'], region_forecast['Forecast_Coral_Cover'], 
                    marker='s', linestyle='--', color=region_colors[region], 
                    linewidth=2, markersize=8,
                    label=f'{region_names[region]} (Forecast)')
            
            # Add trend lines for each region
            # Historical
            X_reg = region_data['Year'].values.reshape(-1, 1)
            y_reg = region_data['Stony_Coral_Cover'].values
            if len(X_reg) > 1:  # Ensure there's enough data for regression
                model_reg = LinearRegression().fit(X_reg, y_reg)
                reg_slope = model_reg.coef_[0]
                
                # Add trend annotation
                plt.annotate(f"{region_names[region]} Trend: {reg_slope:.5f}/year", 
                            xy=(X_reg[-1, 0], y_reg[-1]), 
                            xytext=(10, 0), textcoords='offset points',
                            fontsize=9, color=region_colors[region])
        
        # Mark the transition point
        plt.axvline(x=last_historical_year, color='gray', linestyle=':')
        plt.text(last_historical_year + 0.2, plt.ylim()[1] * 0.9, 
                f"Forecast\nStart", fontsize=12, color='gray')
        
        # Set title and labels
        plt.title('REGIONAL CORAL REEF EVOLUTION (1996-2028)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "regional_coral_reef_evolution.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # 4. Create habitat-specific evolution visualization
        print("Creating habitat-specific evolution visualization...")
        
        # Create a figure
        plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
        
        # Color mapping for habitats
        habitat_mapping = {
            'HB': 'Hardbottom',
            'BCP': 'Backcountry Patch',
            'P': 'Patch Reef',
            'OS': 'Offshore Shallow',
            'OD': 'Offshore Deep'
        }
        
        habitat_colors = {
            'HB': COLORS['coral'],
            'BCP': COLORS['dark_blue'],
            'P': COLORS['ocean_blue'],
            'OS': COLORS['light_blue'],
            'OD': COLORS['neutral']
        }
        
        # Plot each habitat's historical data
        for habitat in habitat_yearly['Habitat'].unique():
            # Historical data
            habitat_data = habitat_yearly[habitat_yearly['Habitat'] == habitat]
            plt.plot(habitat_data['Year'], habitat_data['Stony_Coral_Cover'], 
                    marker='o', linestyle='-', color=habitat_colors[habitat], 
                    linewidth=2, markersize=6, 
                    label=f'{habitat_mapping.get(habitat, habitat)}')
            
            # Add trend lines for each habitat
            X_hab = habitat_data['Year'].values.reshape(-1, 1)
            y_hab = habitat_data['Stony_Coral_Cover'].values
            if len(X_hab) > 1:  # Ensure there's enough data for regression
                model_hab = LinearRegression().fit(X_hab, y_hab)
                hab_slope = model_hab.coef_[0]
                hab_years = np.linspace(habitat_data['Year'].min(), habitat_data['Year'].max(), 100)
                plt.plot(hab_years, model_hab.predict(hab_years.reshape(-1, 1)), 
                        '--', color=habitat_colors[habitat], linewidth=1, alpha=0.5)
                
                # Add trend annotation
                plt.annotate(f"{habitat_mapping.get(habitat, habitat)} Trend: {hab_slope:.5f}/year", 
                            xy=(X_hab[-1, 0], y_hab[-1]), 
                            xytext=(10, 0), textcoords='offset points',
                            fontsize=9, color=habitat_colors[habitat])
        
        # Set title and labels
        plt.title('HABITAT-SPECIFIC CORAL REEF EVOLUTION (1996-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "habitat_coral_reef_evolution.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Coral reef evolution analysis visualizations saved.")
        
        # 5. Save evolution analysis data
        print("Saving evolution analysis data...")
        
        # Create evolution summary DataFrame
        evolution_summary = pd.DataFrame({
            'Metric': [
                'Historical Starting Cover (1996)',
                'Historical Ending Cover (2023)',
                'Historical Net Change (%)',
                'Historical Annual Growth Rate',
                'Forecast Cover (2028)',
                'Forecast Change from 2023 (%)',
                'Forecast Annual Growth Rate'
            ],
            'Value': [
                first_cover,
                last_cover,
                overall_change,
                hist_slope,
                projected_cover,
                projected_change,
                future_slope
            ]
        })
        
        # Save to CSV
        evolution_summary.to_csv(os.path.join(results_dir, "coral_reef_evolution_summary.csv"), index=False)
        
        # Save regional trends
        regional_trends = pd.DataFrame(columns=['Region', 'Historical_Growth_Rate', 'Forecast_Growth_Rate'])
        
        for region in regional_yearly['Subregion'].unique():
            # Historical data
            region_data = regional_yearly[regional_yearly['Subregion'] == region]
            X_reg = region_data['Year'].values.reshape(-1, 1)
            y_reg = region_data['Stony_Coral_Cover'].values
            
            if len(X_reg) > 1:
                model_reg = LinearRegression().fit(X_reg, y_reg)
                hist_rate = model_reg.coef_[0]
            else:
                hist_rate = np.nan
            
            # Forecast data
            region_forecast = forecast_regional[forecast_regional['Subregion'] == region]
            X_forecast = region_forecast['Year'].values.reshape(-1, 1)
            y_forecast = region_forecast['Forecast_Coral_Cover'].values
            
            if len(X_forecast) > 1:
                model_forecast = LinearRegression().fit(X_forecast, y_forecast)
                forecast_rate = model_forecast.coef_[0]
            else:
                forecast_rate = np.nan
            
            # Add to DataFrame
            regional_trends = pd.concat([regional_trends, pd.DataFrame({
                'Region': [region_names.get(region, region)],
                'Historical_Growth_Rate': [hist_rate],
                'Forecast_Growth_Rate': [forecast_rate]
            })], ignore_index=True)
        
        # Save to CSV
        regional_trends.to_csv(os.path.join(results_dir, "regional_coral_reef_trends.csv"), index=False)
        
        print("Evolution analysis data saved.")
        
        return {
            'evolution_summary': evolution_summary,
            'regional_trends': regional_trends
        }
        
    except Exception as e:
        print(f"Error in coral reef evolution analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_model_performance_time_series(model_results, results_dir='11_Results'):
    """
    Creates a time series visualization showing model predictions against actual test data.
    
    Args:
        model_results (dict): Dictionary containing model results and evaluation metrics
        results_dir (str): Directory to save results
    """
    try:
        print("Creating time series visualization of model performance...")
        
        # Extract test data and predictions
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        
        # Get best model name and model object
        best_model_name = model_results['best_model']['name']
        best_model = model_results['best_model']['model']
        
        # Generate predictions
        y_pred = best_model.predict(X_test)
        
        # Create a DataFrame with test data and predictions
        if 'Year' in X_test.columns:
            test_df = pd.DataFrame({
                'Year': X_test['Year'],
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            # Just use year for date representation - simplified approach
            test_df['Date'] = pd.to_datetime(test_df['Year'].astype(int), format='%Y')
        else:
            # Create index based data if no Year column
            test_df = pd.DataFrame({
                'Index': range(len(y_test)),
                'Actual': y_test,
                'Predicted': y_pred
            })
            test_df['Date'] = test_df.index
        
        # Sort by date for proper time series visualization
        if 'Date' in test_df.columns:
            test_df = test_df.sort_values('Date')
        
        # Calculate yearly averages for smoother visualization
        yearly_test = test_df.groupby('Year').agg({
            'Actual': 'mean',
            'Predicted': 'mean'
        }).reset_index()
        
        # Calculate error metrics by year
        yearly_test['AbsError'] = np.abs(yearly_test['Actual'] - yearly_test['Predicted'])
        yearly_test['Error%'] = np.abs((yearly_test['Actual'] - yearly_test['Predicted']) / yearly_test['Actual']) * 100
        
        # Create figure
        plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
        
        # Plot actual vs predicted values
        plt.plot(yearly_test['Year'], yearly_test['Actual'], 
                marker='o', linestyle='-', color=COLORS['coral'], 
                linewidth=2.5, markersize=10, label='Actual Coral Cover')
        
        plt.plot(yearly_test['Year'], yearly_test['Predicted'], 
                marker='s', linestyle='--', color=COLORS['dark_blue'], 
                linewidth=2.5, markersize=8, label=f'Predicted Cover ({best_model_name})')
        
        # Add confidence band - showing prediction error range
        error_margin = yearly_test['AbsError']
        plt.fill_between(yearly_test['Year'], 
                         yearly_test['Actual'] - error_margin,
                         yearly_test['Actual'] + error_margin,
                         alpha=0.2, color=COLORS['dark_blue'], 
                         label='Prediction Error Range')
        
        # Add error percentage annotations
        for i, row in yearly_test.iterrows():
            plt.annotate(f"{row['Error%']:.1f}%", 
                       (row['Year'], max(row['Actual'], row['Predicted']) + 0.01),
                       textcoords="offset points", xytext=(0,5),
                       ha='center', fontsize=9, color=COLORS['dark_blue'])
        
        # Set labels and title
        plt.title('MODEL PERFORMANCE: ACTUAL VS PREDICTED CORAL COVER', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Percent Cover', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add performance metrics in an annotation box
        r2 = model_results['metrics'][best_model_name]['test_r2']
        rmse = model_results['metrics'][best_model_name]['test_rmse']
        mae = model_results['metrics'][best_model_name]['test_mae']
        
        metrics_text = (
            f"Model Performance Metrics ({best_model_name}):\n"
            f"• R² Score: {r2:.4f}\n"
            f"• RMSE: {rmse:.4f}\n"
            f"• MAE: {mae:.4f}\n"
            f"• Mean Error %: {yearly_test['Error%'].mean():.2f}%"
        )
        
        plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                    fontsize=11, verticalalignment='bottom')
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Enhance the legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                fontsize=12, loc='upper left', edgecolor=COLORS['grid'])
        
        # Set tight layout
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(os.path.join(results_dir, "model_performance_time_series.png"), 
                  bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Model performance time series visualization saved.")
        
        # Create a scatter plot of actual vs. predicted with residuals
        plt.figure(figsize=(18, 9), facecolor=COLORS['background'])
        
        # Create a 1x2 subplot layout
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        
        # Scatter plot of actual vs predicted
        ax1 = plt.subplot(gs[0])
        ax1.set_facecolor(COLORS['background'])
        
        scatter = ax1.scatter(test_df['Actual'], test_df['Predicted'], 
                             alpha=0.6, c=test_df['Year'] if 'Year' in test_df.columns else 'blue',
                             cmap='viridis', s=50)
        
        # Add perfect prediction line
        max_val = max(test_df['Actual'].max(), test_df['Predicted'].max())
        min_val = min(test_df['Actual'].min(), test_df['Predicted'].min())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add best fit line
        slope, intercept = np.polyfit(test_df['Actual'], test_df['Predicted'], 1)
        x_line = np.linspace(min_val, max_val, 100)
        ax1.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2, 
                label=f'Best Fit (y = {slope:.3f}x + {intercept:.3f})')
        
        # Add colorbar to show year gradient
        if 'Year' in test_df.columns:
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Year', fontsize=12, fontweight='bold')
        
        # Set labels and title
        ax1.set_title('Actual vs. Predicted Values', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])
        ax1.set_xlabel('Actual Coral Cover', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Coral Cover', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax1.legend(fontsize=12)
        
        # Residual plot
        ax2 = plt.subplot(gs[1])
        ax2.set_facecolor(COLORS['background'])
        
        # Calculate residuals
        test_df['Residuals'] = test_df['Actual'] - test_df['Predicted']
        
        # Scatter plot of residuals
        if 'Year' in test_df.columns:
            ax2.scatter(test_df['Year'], test_df['Residuals'], alpha=0.6, c=test_df['Year'], cmap='viridis', s=50)
            # Add trend line for residuals
            if len(test_df) > 1:
                res_slope, res_intercept = np.polyfit(test_df['Year'], test_df['Residuals'], 1)
                years = np.linspace(test_df['Year'].min(), test_df['Year'].max(), 100)
                ax2.plot(years, res_slope * years + res_intercept, 'g-', linewidth=2, 
                        label=f'Residual Trend ({res_slope:.5f}/year)')
        else:
            ax2.scatter(test_df.index, test_df['Residuals'], alpha=0.6, color='blue', s=50)
        
        # Add zero line
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        # Set labels and title
        ax2.set_title('Residuals Over Time', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])
        ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        if 'Year' in test_df.columns and len(test_df) > 1:
            ax2.legend(fontsize=12)
        
        # Add overall title
        plt.suptitle('MODEL VALIDATION: PREDICTION ACCURACY AND RESIDUAL ANALYSIS', 
                   fontsize=20, fontweight='bold', color=COLORS['dark_blue'],
                   path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                   y=0.98)
        
        # Save the visualization
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(results_dir, "model_residual_analysis.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Model residual analysis visualization saved.")
        
        return True
        
    except Exception as e:
        print(f"Error in model performance time series visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution function
def main():
    """Main execution function."""
    print("\n=== Stony Coral Percentage Cover Forecasting Analysis ===\n")
    
    try:
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        else:
            print(f"Directory already exists: {results_dir}")
        
        # Load and preprocess data
        print("Starting data loading process...")
        data_dict = load_and_preprocess_data()
        
        # Basic validation
        if data_dict is not None:
            pcover_df = data_dict['pcover_df']
            stations_df = data_dict['stations_df']
            
            # Print some information for validation
            print(f"\nPercentage cover data shape: {pcover_df.shape}")
            print(f"Stations data shape: {stations_df.shape}")
            
            # Print station columns for debugging
            print(f"Station columns: {stations_df.columns.tolist()}")
            
            # Unique regions and habitats
            print(f"\nUnique regions: {pcover_df['Subregion'].unique().tolist()}")
            print(f"Unique habitats: {pcover_df['Habitat'].unique().tolist()}")
            
            # Years covered
            years = sorted(pcover_df['Year'].unique().tolist())
            print(f"\nYears covered in the dataset: {years}")
            print(f"Total time span: {years[-1] - years[0]} years")
            
            # Analyze time series patterns
            try:
                print("\nAnalyzing time series patterns...")
                ts_analysis = analyze_time_series_patterns(data_dict)
                print("Time series analysis complete.")
            except Exception as e:
                print(f"Error during time series analysis: {e}")
                print("Continuing with feature engineering...")
                import traceback
                traceback.print_exc()
            
            # Engineer features for modeling
            try:
                print("\nEngineering features...")
                feature_dict = engineer_features(pcover_df, stations_df, data_dict['temp_df'])
                print("Feature engineering complete.")
            except Exception as e:
                print(f"Error during feature engineering: {e}")
                import traceback
                traceback.print_exc()
                print("Cannot continue without engineered features. Exiting.")
                return
            
            # Train and evaluate models
            try:
                print("\nTraining and evaluating models...")
                model_results = train_forecasting_models(feature_dict)
                
                # Print best model information
                best_model_name = model_results['best_model']['name']
                best_model_metrics = model_results['metrics'][best_model_name]
                print(f"\nBest model: {best_model_name}")
                print(f"Test R²: {best_model_metrics['test_r2']:.4f}")
                print(f"Test RMSE: {best_model_metrics['test_rmse']:.4f}")
                
                # Create detailed feature importance visualization
                visualize_feature_importance(model_results)
                
                # Create time series performance visualization
                visualize_model_performance_time_series(model_results)
            except Exception as e:
                print(f"Error during model training: {e}")
                import traceback
                traceback.print_exc()
                print("Cannot continue without trained models. Exiting.")
                return
            
            # Generate forecasts
            try:
                print("\n=== Generating Stony Coral Cover Forecasts (2024-2028) ===")
                forecasts_df = generate_forecasts(feature_dict, model_results)
                
                if len(forecasts_df) > 0:
                    # Save forecast results to CSV
                    forecast_file = os.path.join(results_dir, "stony_coral_cover_forecasts.csv")
                    forecasts_df.to_csv(forecast_file, index=False)
                    print(f"Forecast results saved to {forecast_file}")
                    
                    # Save the best model for future use
                    model_file = os.path.join(results_dir, f"stony_coral_cover_{best_model_name.lower().replace(' ', '_')}_model.pkl")
                    joblib.dump(model_results['best_model']['model'], model_file)
                    print(f"Best model saved to {model_file}")
                    
                    # Visualize forecasts
                    visualize_forecasts(forecasts_df, pcover_df)
                    
                    # Perform detailed coral reef evolution analysis
                    analyze_reef_evolution(forecasts_df, pcover_df)
                    
                    print("\n=== Forecast Generation Complete ===")
                else:
                    print("No forecasts were generated.")
            except Exception as e:
                print(f"Error during forecast generation: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Model Training and Evaluation Complete ===\n")

# Execute main function if script is run directly
if __name__ == "__main__":
    main() 