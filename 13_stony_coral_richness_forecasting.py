"""
13_stony_coral_richness_forecasting.py - Stony Coral Species Richness Forecasting

This script analyzes historical stony coral species richness data, builds predictive models,
and forecasts future trends for the next five years. It includes time series analysis, 
feature engineering, model training, and visualization components.

Author: Shivam Vashishtha
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "13_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

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

# Define a modern color palette - enhanced with more vivid colors
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
    'background': '#F8F9FA'  # Very light grey background
}

# Create a custom colormap for coral reef visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP datasets for stony coral species richness forecasting.
    
    Returns:
        dict: Dictionary containing preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load Stony Coral Species data
        species_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
        print(f"Stony coral species data loaded successfully with {len(species_df)} rows")
        
        # Load Station data
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Load stony coral density data for correlation
        stony_density_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Density.csv")
        print(f"Stony coral density data loaded successfully with {len(stony_density_df)} rows")
        
        # Load Temperature data
        try:
            temperature_df = pd.read_csv("CREMP_CSV_files/CREMP_Temperatures_2023.csv")
            print(f"Temperature data loaded successfully with {len(temperature_df)} rows")
        except Exception as e:
            print(f"Error loading temperature data: {str(e)}")
            print("Trying alternate temperature file...")
            try:
                temperature_df = pd.read_csv("CREMP_CSV_files/CREMP_Temperature_Daily_2023.csv")
                print(f"Temperature data loaded successfully with {len(temperature_df)} rows")
            except Exception as e2:
                print(f"Error loading alternate temperature data: {str(e2)}")
                temperature_df = None
        
        # Convert date columns to datetime
        species_df['Date'] = pd.to_datetime(species_df['Date'])
        species_df['Year'] = species_df['Year'].astype(int)
        
        if 'Date' in stony_density_df.columns:
            stony_density_df['Date'] = pd.to_datetime(stony_density_df['Date'])
            stony_density_df['Year'] = stony_density_df['Year'].astype(int)
        
        if temperature_df is not None and 'Date' in temperature_df.columns:
            temperature_df['Date'] = pd.to_datetime(temperature_df['Date'])
            if 'Year' not in temperature_df.columns:
                temperature_df['Year'] = temperature_df['Date'].dt.year
        
        # Get list of all coral species columns (excluding metadata columns)
        metadata_cols = ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID', 'Surveyed_all_years', 'points']
        species_cols = [col for col in species_df.columns if col not in metadata_cols]
        
        print(f"Identified {len(species_cols)} coral species in the dataset")
        
        # Calculate species richness (number of species present at each station)
        # A species is considered present if its cover value is greater than 0
        species_df['Species_Richness'] = species_df[species_cols].apply(
            lambda row: sum(row > 0), axis=1
        )
        
        # Basic statistics
        print("\nBasic statistics for Species Richness:")
        print(species_df['Species_Richness'].describe())
        
        # Get time range
        print(f"\nData spans from {species_df['Year'].min()} to {species_df['Year'].max()}")
        
        # Create a data dictionary to hold all processed dataframes
        data_dict = {
            'species_df': species_df,
            'stations_df': stations_df,
            'stony_density_df': stony_density_df,
            'temperature_df': temperature_df,
            'species_cols': species_cols
        }
        
        return data_dict
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def analyze_time_series_patterns(data_dict):
    """
    Analyze temporal patterns in stony coral species richness data.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        
    Returns:
        dict: Dictionary containing time series analysis results
    """
    print("\nAnalyzing time series patterns in stony coral species richness...")
    
    # Extract species richness data
    species_df = data_dict['species_df']
    
    # Create a yearly average time series
    yearly_avg = species_df.groupby('Year')['Species_Richness'].mean().reset_index()
    print(f"Created yearly average time series with {len(yearly_avg)} points")
    
    # Create regional yearly averages
    regional_yearly_avg = species_df.groupby(['Year', 'Subregion'])['Species_Richness'].mean().reset_index()
    
    # Create habitat yearly averages
    habitat_yearly_avg = species_df.groupby(['Year', 'Habitat'])['Species_Richness'].mean().reset_index()
    
    # Perform stationarity test on overall yearly average
    print("\nPerforming Augmented Dickey-Fuller test for stationarity...")
    adf_result = adfuller(yearly_avg['Species_Richness'])
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
    acf_values = acf(yearly_avg['Species_Richness'], nlags=max_lags)
    pacf_values = pacf(yearly_avg['Species_Richness'], nlags=max_lags)
    
    # Identify change points
    print("\nIdentifying significant change points...")
    
    # Calculate year-to-year percentage changes
    yearly_avg['pct_change'] = yearly_avg['Species_Richness'].pct_change() * 100
    
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
    ax1.plot(yearly_avg['Year'], yearly_avg['Species_Richness'], 
             marker='o', linestyle='-', color=COLORS['coral'], 
             linewidth=2.5, markersize=8, label='Annual Mean Species Richness')
    
    # Fit a linear trend line
    X = yearly_avg['Year'].values.reshape(-1, 1)
    y = yearly_avg['Species_Richness'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    y_pred = model.predict(X)
    
    # Plot the trend line
    ax1.plot(yearly_avg['Year'], y_pred, '--', color=COLORS['dark_blue'], 
             linewidth=2, label=f'Linear Trend (Slope: {slope:.4f} per year)')
    
    # Mark significant change points
    for _, row in change_points.iterrows():
        ax1.scatter(row['Year'], row['Species_Richness'], 
                   s=150, color='red', zorder=5, 
                   marker='*', label='_nolegend_')
        
        # Add annotations for significant changes
        ax1.annotate(f"{row['pct_change']:.1f}%", 
                    xy=(row['Year'], row['Species_Richness']),
                    xytext=(row['Year'], row['Species_Richness'] + 0.5),
                    fontsize=10, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.7, color='white'))
    
    # Set plot aesthetics
    ax1.set_title('STONY CORAL SPECIES RICHNESS TREND (1996-2023)', 
                 fontweight='bold', fontsize=18, pad=15, color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Add text explaining trend
    trend_direction = "increasing" if slope > 0 else "decreasing"
    trend_text = (
        f"Overall Trend: Species richness is {trend_direction} at a rate of {abs(slope):.4f} species per year.\n"
        f"Current (2023) mean richness: {yearly_avg.iloc[-1]['Species_Richness']:.2f} species\n"
        f"Historical (1996) mean richness: {yearly_avg.iloc[0]['Species_Richness']:.2f} species"
    )
    
    # Add the trend text with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    
    # Position the text box in the upper left or right depending on the trend
    ax1.text(0.02, 0.92, trend_text, transform=ax1.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Plot 2: Regional time series
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
        ax2.plot(region_data['Year'], region_data['Species_Richness'], 
                marker='o', linestyle='-', 
                color=region_colors.get(region, COLORS['coral']), 
                linewidth=2, markersize=6, 
                label=region_names.get(region, region))
    
    # Set plot aesthetics
    ax2.set_title('Regional Trends in Species Richness', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax2.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=12, labelpad=8)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 3: Habitat time series
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
        ax3.plot(habitat_data['Year'], habitat_data['Species_Richness'], 
                marker='o', linestyle='-', 
                color=habitat_colors.get(habitat, COLORS['coral']), 
                linewidth=2, markersize=6, 
                label=habitat_names.get(habitat, habitat))
    
    # Set plot aesthetics
    ax3.set_title('Habitat-Specific Trends in Species Richness', 
                 fontweight='bold', fontsize=16, pad=15)
    ax3.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax3.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=12, labelpad=8)
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
    ax5.axhline(y=confidence, linestyle='--', color='red', alpha=0.7)
    ax5.axhline(y=-confidence, linestyle='--', color='red', alpha=0.7)
    
    # Set plot aesthetics
    ax5.set_title('Partial Autocorrelation Function (PACF)', 
                 fontweight='bold', fontsize=16, pad=15)
    ax5.set_xlabel('Lag (Years)', fontweight='bold', fontsize=12, labelpad=8)
    ax5.set_ylabel('Partial Autocorrelation', fontweight='bold', fontsize=12, labelpad=8)
    ax5.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 6: Year-to-year changes
    ax6 = plt.subplot(gs[3, 0])
    ax6.set_facecolor(COLORS['background'])
    
    # Bar plot of percentage changes
    bars = ax6.bar(yearly_avg['Year'][1:], yearly_avg['pct_change'][1:], 
                  width=0.7, color=COLORS['light_blue'], 
                  edgecolor=COLORS['dark_blue'], alpha=0.7)
    
    # Highlight significant changes
    for i, row in change_points.iterrows():
        if i > 0:  # Skip the first year which has no change
            idx = yearly_avg.index[yearly_avg['Year'] == row['Year']].tolist()[0]
            if idx < len(bars):
                bars[idx-1].set_color(COLORS['coral'])
                bars[idx-1].set_alpha(0.9)
    
    # Add horizontal lines for standard deviation thresholds
    ax6.axhline(y=std_threshold, linestyle='--', color='red', alpha=0.7, 
               label=f'+2σ ({std_threshold:.1f}%)')
    ax6.axhline(y=-std_threshold, linestyle='--', color='red', alpha=0.7, 
               label=f'-2σ ({-std_threshold:.1f}%)')
    ax6.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    
    # Set plot aesthetics
    ax6.set_title('Year-to-Year Percentage Changes in Species Richness', 
                 fontweight='bold', fontsize=16, pad=15)
    ax6.set_xlabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    ax6.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=12, labelpad=8)
    ax6.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax6.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
    
    # Rotate x-axis labels for better readability
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 7: Heatmap of richness over time
    ax7 = plt.subplot(gs[3, 1])
    ax7.set_facecolor(COLORS['background'])
    
    # Create a pivot table for the heatmap
    richness_pivot = species_df.pivot_table(
        index='Year', 
        columns='Subregion', 
        values='Species_Richness', 
        aggfunc='mean'
    )
    
    # Plot the heatmap
    sns.heatmap(richness_pivot, cmap=coral_cmap, annot=True, fmt='.1f', 
               linewidths=0.5, ax=ax7, cbar_kws={'label': 'Mean Species Richness'})
    
    # Set plot aesthetics
    ax7.set_title('Temporal Heatmap of Species Richness by Region', 
                 fontweight='bold', fontsize=16, pad=15)
    ax7.set_xlabel('Region', fontweight='bold', fontsize=12, labelpad=8)
    ax7.set_ylabel('Year', fontweight='bold', fontsize=12, labelpad=8)
    
    # Add note about key events in coral reef history
    key_events = {
        2005: "2005 Caribbean-wide\nbleaching event",
        2010: "2010 Cold water\nmass mortality",
        2014: "2014-2015 Global\nbleaching event",
        2017: "2017 Hurricane Irma",
        2019: "2019 SCTLD\ndisease peak"
    }
    
    # Add vertical lines for key events on main plot
    for year, event in key_events.items():
        if year >= yearly_avg['Year'].min() and year <= yearly_avg['Year'].max():
            ax1.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
            ax1.annotate(event, xy=(year, ax1.get_ylim()[0] + 0.5), xytext=(year, ax1.get_ylim()[0] + 0.2),
                        rotation=90, va='bottom', ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_time_series_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
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

def engineer_features(species_data, stations_df, temperature_data, target_col='Species_Richness'):
    """
    Engineer features for stony coral species richness forecasting.
    
    Args:
        species_data (pd.DataFrame): Species richness data
        stations_df (pd.DataFrame): Station information data
        temperature_data (pd.DataFrame): Temperature data
        target_col (str): Target column name
    
    Returns:
        dict: Dictionary containing engineered features
    """
    print("\nEngineering features for forecasting models...")
    
    # Create a copy of the species data
    df = species_data.copy()
    
    # Ensure the target column exists
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found in the dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Target column not found in data")
    
    # 1. Temporal features
    print("Creating temporal features...")
    # Extract year from date if not already present
    if 'Year' not in df.columns and 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
    
    # Add month if available
    if 'Month' not in df.columns and 'Date' in df.columns:
        df['Month'] = df['Date'].dt.month
    
    # Add season
    if 'Season' not in df.columns and 'Month' in df.columns:
        df['Season'] = df['Month'].map(lambda m: 'Winter' if m in [12, 1, 2] else
                                     'Spring' if m in [3, 4, 5] else
                                     'Summer' if m in [6, 7, 8] else 'Fall')
    
    # 2. Create lagged features
    print("Creating lagged features...")
    # Group by site to create site-specific lags
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Year')
        
        # Create 1-year and 2-year lags for the target variable
        if len(site_data) > 1:
            df.loc[site_mask, 'Lag1_Richness'] = site_data[target_col].shift(1)
            if len(site_data) > 2:
                df.loc[site_mask, 'Lag2_Richness'] = site_data[target_col].shift(2)
    
    # Calculate rolling averages
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Year')
        
        if len(site_data) > 2:
            # 2-year and 3-year rolling averages
            df.loc[site_mask, 'RollingAvg2_Richness'] = site_data[target_col].rolling(window=2, min_periods=1).mean().values
            if len(site_data) > 3:
                df.loc[site_mask, 'RollingAvg3_Richness'] = site_data[target_col].rolling(window=3, min_periods=1).mean().values
    
    # Calculate running rate of change for species richness
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Year')
        
        if len(site_data) > 1:
            df.loc[site_mask, 'ROC_1yr'] = site_data[target_col].pct_change(periods=1) * 100
            if len(site_data) > 2:
                df.loc[site_mask, 'ROC_2yr'] = site_data[target_col].pct_change(periods=2) * 100
    
    # 3. Add spatial features
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
    
    # Merge station data with species data
    print(f"Merging with station data...")
    merge_cols = ['StationID']  # Adjust based on common columns
    
    # Standardize column names if needed
    if 'StationID' in df.columns and 'StationID' in stations_df.columns:
        # Stations data to merge
        station_subset = stations_df[spatial_columns]
        
        # Depth is important for species richness
        if 'Depth_ft' in stations_df.columns and 'Depth' not in stations_df.columns:
            station_subset['Depth'] = stations_df['Depth_ft']
        
        # Merge on StationID
        df = pd.merge(df, station_subset, on='StationID', how='left', suffixes=('', '_station'))
    
    # 4. Add temperature features if available
    if temperature_data is None:
        print("No temperature data available. Skipping temperature features.")
    else:
        print("Adding temperature features...")
        # Check for temperature column
        temp_column = None
        possible_temp_columns = ['Temperature', 'TempC', 'Bottom_Temperature_degC', 'Water_Temperature_C']
        
        for col in possible_temp_columns:
            if col in temperature_data.columns:
                temp_column = col
                print(f"Using '{col}' as temperature column")
                break
        
        if temp_column is None:
            print(f"No temperature column found. Available columns: {temperature_data.columns.tolist()}")
            print("Continuing without temperature features.")
        else:
            # Rename to standardize
            temperature_data = temperature_data.rename(columns={temp_column: 'Temperature'})
            
            # Group temperature data by site and year
            try:
                # Ensure date column is in datetime format
                if 'Date' in temperature_data.columns:
                    temperature_data['Date'] = pd.to_datetime(temperature_data['Date'])
                    if 'Year' not in temperature_data.columns:
                        temperature_data['Year'] = temperature_data['Date'].dt.year
                
                # Create site-specific temperature features
                site_temps = {}
                
                for site in df['Site_name'].unique():
                    # Try to match site in temperature data
                    if 'Site_name' in temperature_data.columns:
                        site_mask = temperature_data['Site_name'] == site
                    elif 'SiteID' in temperature_data.columns and 'SiteID' in df.columns:
                        site_id = df[df['Site_name'] == site]['SiteID'].iloc[0]
                        site_mask = temperature_data['SiteID'] == site_id
                    else:
                        print(f"Could not find site {site} in temperature data.")
                        continue
                    
                    if site_mask.any():
                        site_temp_data = temperature_data[site_mask]
                        
                        # Calculate temperature statistics by year
                        temp_stats = site_temp_data.groupby('Year').agg({
                            'Temperature': ['mean', 'max', 'min', 'std']
                        })
                        
                        # Fix column naming to be more robust
                        if isinstance(temp_stats.columns, pd.MultiIndex):
                            # Convert multi-index columns to single-index with appropriate names
                            temp_stats.columns = [f'Temp_{col[1]}' for col in temp_stats.columns]
                        else:
                            # If not a multi-index, just prefix with Temp_
                            temp_stats.columns = ['Temp_' + col for col in temp_stats.columns]
                        
                        temp_stats = temp_stats.reset_index()
                        
                        site_temps[site] = temp_stats
                
                # Add temperature features to each site in main dataframe
                for site, temp_df in site_temps.items():
                    site_mask = df['Site_name'] == site
                    if site_mask.any():
                        # Merge temperature data with site data
                        try:
                            df_site = df[site_mask].merge(temp_df, on='Year', how='left')
                            
                            # Update the main dataframe
                            for col in temp_df.columns:
                                if col != 'Year':
                                    if col in df_site.columns:
                                        df.loc[site_mask, col] = df_site[col].values
                                    else:
                                        print(f"Warning: Column {col} not found in merged dataframe. Skipping.")
                        except Exception as merge_error:
                            print(f"Error merging temperature data for site {site}: {merge_error}")
                            continue
            except Exception as e:
                print(f"Error processing temperature data: {e}")
                print("Continuing without temperature features.")
                import traceback
                traceback.print_exc()
    
    # 5. Calculate time since first observation
    print("Adding time-based features...")
    for site in df['Site_name'].unique():
        site_mask = df['Site_name'] == site
        site_data = df[site_mask].sort_values('Year')
        if len(site_data) > 0:
            first_year = site_data['Year'].min()
            df.loc[site_mask, 'Years_Since_First_Observation'] = df.loc[site_mask, 'Year'] - first_year
    
    # 6. One-hot encode categorical variables
    print("One-hot encoding categorical variables...")
    # One-hot encode Season if it exists
    if 'Season' in df.columns:
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
    
    # 7. Fill missing values
    print(f"Missing values before imputation: {df.isna().sum().sum()}")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    feature_dict = {
        'feature_df': df,
        'target_col': target_col
    }
    
    print(f"Engineered feature dataframe has {len(df)} rows and {len(df.columns)} columns")
    
    return feature_dict

def train_forecasting_models(feature_dict, test_size=0.2, random_state=42):
    """
    Train and evaluate forecasting models for stony coral species richness.
    
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
    df = df.sort_values('Year')
    train_size = int(len(df) * (1 - test_size))
    
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"Training data: {len(df_train)} rows (years {df_train['Year'].min()}-{df_train['Year'].max()})")
    print(f"Testing data: {len(df_test)} rows (years {df_test['Year'].min()}-{df_test['Year'].max()})")
    
    # Define features to exclude from modeling
    exclude_cols = ['Date', 'Site_name', 'StationID', 'SiteID', 'Species_Richness',
                    'Subregion', 'Habitat', 'Season', target_col]
    
    # Add any species columns that might have leaked in
    for col in df.columns:
        if col.startswith('Acrop') or col.startswith('Montas') or 'coral' in col.lower():
            if col not in exclude_cols and col != target_col:
                exclude_cols.append(col)
    
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
        # Replace the previous imputation with a more robust approach
        # First drop columns that are all NaN
        all_nan_cols = X_train.columns[X_train.isna().sum() == len(X_train)]
        if len(all_nan_cols) > 0:
            print(f"Dropping columns with all NaN values: {list(all_nan_cols)}")
            X_train = X_train.drop(columns=all_nan_cols)
            X_test = X_test.drop(columns=all_nan_cols)
        
        # Then calculate means for remaining columns and fill NaNs
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        
        # Force fill any remaining NaNs with 0
        X_train = X_train.fillna(0)
    
    if X_test.isna().sum().sum() > 0:
        print("Filling remaining NaN values in test data with mean")
        # Use the training means for test data
        X_test = X_test.fillna(train_means)
        # Force fill any remaining NaNs with 0
        X_test = X_test.fillna(0)
    
    # Handle infinity values
    print("Checking for infinity values...")
    
    # Replace infinity with NaN, then fill with column max/min
    inf_mask_train = np.isinf(X_train)
    inf_count_train = inf_mask_train.sum().sum()
    if inf_count_train > 0:
        print(f"Found {inf_count_train} infinity values in training data")
        # Replace positive infinity with column max * 2 (a large but finite value)
        cols_with_pos_inf = X_train.columns[(np.isposinf(X_train)).any()]
        for col in cols_with_pos_inf:
            col_max = X_train[~np.isinf(X_train[col])][col].max()
            X_train.loc[np.isposinf(X_train[col]), col] = col_max * 2 if col_max > 0 else 100
            
        # Replace negative infinity with column min * 2 (a small but finite value)
        cols_with_neg_inf = X_train.columns[(np.isneginf(X_train)).any()]
        for col in cols_with_neg_inf:
            col_min = X_train[~np.isinf(X_train[col])][col].min()
            X_train.loc[np.isneginf(X_train[col]), col] = col_min * 2 if col_min < 0 else -100
    
    # Do the same for test data
    inf_mask_test = np.isinf(X_test)
    inf_count_test = inf_mask_test.sum().sum()
    if inf_count_test > 0:
        print(f"Found {inf_count_test} infinity values in test data")
        # Replace positive infinity with column max * 2 (a large but finite value)
        cols_with_pos_inf = X_test.columns[(np.isposinf(X_test)).any()]
        for col in cols_with_pos_inf:
            col_max = X_test[~np.isinf(X_test[col])][col].max()
            if pd.isna(col_max) and col in X_train.columns:
                col_max = X_train[~np.isinf(X_train[col])][col].max()
            X_test.loc[np.isposinf(X_test[col]), col] = col_max * 2 if not pd.isna(col_max) and col_max > 0 else 100
            
        # Replace negative infinity with column min * 2 (a small but finite value)
        cols_with_neg_inf = X_test.columns[(np.isneginf(X_test)).any()]
        for col in cols_with_neg_inf:
            col_min = X_test[~np.isinf(X_test[col])][col].min()
            if pd.isna(col_min) and col in X_train.columns:
                col_min = X_train[~np.isinf(X_train[col])][col].min()
            X_test.loc[np.isneginf(X_test[col]), col] = col_min * 2 if not pd.isna(col_min) and col_min < 0 else -100
    
    # Final check for any remaining problematic values
    print(f"NaN values in X_train after imputation: {X_train.isna().sum().sum()}")
    print(f"NaN values in X_test after imputation: {X_test.isna().sum().sum()}")
    print(f"Infinity values in X_train after processing: {np.isinf(X_train).sum().sum()}")
    print(f"Infinity values in X_test after processing: {np.isinf(X_test).sum().sum()}")
    
    # Handle extreme values that might cause numerical instability
    print("Checking for extreme values...")
    # Set a reasonable threshold for extreme values based on the data
    max_threshold = 1e10
    min_threshold = -1e10
    
    # Count extreme values
    extreme_values_train = ((X_train > max_threshold) | (X_train < min_threshold)).sum().sum()
    extreme_values_test = ((X_test > max_threshold) | (X_test < min_threshold)).sum().sum()
    
    if extreme_values_train > 0:
        print(f"Found {extreme_values_train} extreme values in training data")
        # Clip values to be within reasonable bounds
        X_train = X_train.clip(lower=min_threshold, upper=max_threshold)
    
    if extreme_values_test > 0:
        print(f"Found {extreme_values_test} extreme values in test data")
        # Clip values to be within reasonable bounds
        X_test = X_test.clip(lower=min_threshold, upper=max_threshold)
    
    # Print feature names for debugging
    print(f"Features used: {X_train.columns.tolist()}")
    
    # Initialize models with improved hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=random_state
        )
    }
    
    # Prepare results dictionary
    results = {
        'models': {},
        'metrics': {},
        'feature_importance': {},
        'X_train': X_train,  # Save for future prediction
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
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'][name] = feature_importance
        elif name == 'Linear Regression':
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
            results['feature_importance'][name] = feature_importance
    
    # Find the best model based on test R² score
    best_model_name = max(results['metrics'], key=lambda k: results['metrics'][k]['test_r2'])
    best_model = results['models'][best_model_name]
    best_metrics = results['metrics'][best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model Test R²: {best_metrics['test_r2']:.4f}")
    print(f"Best model Test RMSE: {best_metrics['test_rmse']:.4f}")
    
    # Save the best model for later use
    best_model_file = os.path.join(results_dir, "stony_coral_richness_best_model.pkl")
    joblib.dump(best_model, best_model_file)
    print(f"Best model saved to {best_model_file}")
    
    # Add best model info to results
    results['best_model'] = {
        'name': best_model_name,
        'model': best_model,
        'metrics': best_metrics
    }
    
    return results

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
        bars1 = ax1.bar(x - width/2, train_r2, width, label='Training R²',
                color=COLORS['ocean_blue'], alpha=0.7)
        bars2 = ax1.bar(x + width/2, test_r2, width, label='Testing R²',
                color=COLORS['coral'], alpha=0.7)
        
        # Add R² values on top of each bar
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
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
        bars3 = ax2.bar(x - width/2, train_rmse, width, label='Training RMSE',
                color=COLORS['reef_green'], alpha=0.7)
        bars4 = ax2.bar(x + width/2, test_rmse, width, label='Testing RMSE',
                color=COLORS['sand'], alpha=0.7)
        
        # Add RMSE values on top of each bar
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
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
        ax3.set_ylabel('Species Richness', fontweight='bold')
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
            
            # Much more aggressive jittering approach to completely eliminate vertical lines
            # Create a copy of the actual values to add jitter
            jittered_x = y_test.copy().values
            
            # Calculate the average distance between adjacent unique values
            unique_vals = np.sort(np.unique(jittered_x))
            if len(unique_vals) >= 2:
                avg_step = np.mean(np.diff(unique_vals))
                # Apply significant jitter scaled to the average step size (50% of step size)
                jitter_scale = avg_step * 0.5
            else:
                # Fallback if we can't calculate step size
                jitter_scale = 0.5
                
            # Apply random jitter to spread points horizontally
            jittered_x = jittered_x + np.random.uniform(-jitter_scale, jitter_scale, size=len(jittered_x))
            
            # Create scatter plot with jittered x values
            ax.scatter(jittered_x, y_pred, alpha=0.5, color=COLORS['ocean_blue'], s=40)
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Add best fit line for the original (non-jittered) data
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min_val, max_val, 100)
            ax.plot(x_line, p(x_line), '-', color=COLORS['dark_blue'],
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

def visualize_feature_importance(model_results, results_dir='13_Results'):
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
        plt.suptitle('SPECIES RICHNESS PREDICTION - FEATURE IMPORTANCE ANALYSIS', 
                     fontsize=20, fontweight='bold', color=COLORS['dark_blue'], 
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                     y=0.98)
        
        # Add subtitle with explanation
        plt.figtext(0.5, 0.92, 
                   'Features with higher importance have greater influence on species richness predictions', 
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

def visualize_model_performance_time_series(model_results, results_dir='13_Results'):
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
        test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        # Check if 'Year' is in X_test and add it to test_df if available
        has_year = False
        if 'Year' in X_test.columns:
            test_df['Year'] = X_test['Year']
            has_year = True
        else:
            # Create a dummy Year column if not available
            print("Year column not found in test data, using index instead.")
            test_df['Year'] = np.arange(len(test_df)) + 1
        
        # Sort by actual value for better visualization when year not available
        if not has_year:
            test_df = test_df.sort_values('Actual')
        else:
            # Sort by date for proper time series visualization
            test_df = test_df.sort_values('Year')
        
        # Create figure for prediction vs actual plot
        plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
        
        # If year is available, calculate yearly averages for smoother visualization
        if has_year:
            yearly_test = test_df.groupby('Year').agg({
                'Actual': 'mean',
                'Predicted': 'mean'
            }).reset_index()
            
            # Calculate error metrics by year
            yearly_test['AbsError'] = np.abs(yearly_test['Actual'] - yearly_test['Predicted'])
            yearly_test['Error%'] = np.abs((yearly_test['Actual'] - yearly_test['Predicted']) / yearly_test['Actual']) * 100
            
            # Plot actual vs predicted values by year
            plt.plot(yearly_test['Year'], yearly_test['Actual'], 
                    marker='o', linestyle='-', color=COLORS['coral'], 
                    linewidth=2.5, markersize=10, label='Actual Species Richness')
            
            plt.plot(yearly_test['Year'], yearly_test['Predicted'], 
                    marker='s', linestyle='--', color=COLORS['dark_blue'], 
                    linewidth=2.5, markersize=8, label=f'Predicted Richness ({best_model_name})')
            
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
                
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            
            # Calculate average error percentage for annotation
            mean_error_pct = yearly_test['Error%'].mean()
        else:
            # If no year, just plot the test samples
            plt.plot(test_df.index, test_df['Actual'], 
                    marker='o', linestyle='-', color=COLORS['coral'], 
                    linewidth=2.5, markersize=10, label='Actual Species Richness')
            
            plt.plot(test_df.index, test_df['Predicted'], 
                    marker='s', linestyle='--', color=COLORS['dark_blue'], 
                    linewidth=2.5, markersize=8, label=f'Predicted Richness ({best_model_name})')
            
            # Add error information
            test_df['AbsError'] = np.abs(test_df['Actual'] - test_df['Predicted'])
            test_df['Error%'] = np.abs((test_df['Actual'] - test_df['Predicted']) / test_df['Actual']) * 100
            
            plt.xlabel('Sample Index', fontweight='bold', fontsize=14, labelpad=10)
            
            # Calculate average error percentage for annotation
            mean_error_pct = test_df['Error%'].mean()
        
        # Set labels and title
        plt.title('MODEL PERFORMANCE: ACTUAL VS PREDICTED SPECIES RICHNESS', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add performance metrics in an annotation box
        r2 = model_results['metrics'][best_model_name]['test_r2']
        rmse = model_results['metrics'][best_model_name]['test_rmse']
        mae = model_results['metrics'][best_model_name]['test_mae']
        
        metrics_text = (
            f"Model Performance Metrics ({best_model_name}):\n"
            f"• R² Score: {r2:.4f}\n"
            f"• RMSE: {rmse:.4f}\n"
            f"• MAE: {mae:.4f}\n"
            f"• Mean Error %: {mean_error_pct:.2f}%"
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
        
        # We don't save this plot as requested
        # plt.savefig(os.path.join(results_dir, "model_performance_time_series.png"), 
        #          bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Model performance time series visualization created.")
        
        # Create a scatter plot of actual vs. predicted with residuals
        plt.figure(figsize=(18, 9), facecolor=COLORS['background'])
        
        # Create a 1x2 subplot layout
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        
        # Scatter plot of actual vs predicted
        ax1 = plt.subplot(gs[0])
        ax1.set_facecolor(COLORS['background'])
        
        # Add color by year if available
        if has_year:
            scatter = ax1.scatter(test_df['Actual'], test_df['Predicted'], 
                                alpha=0.6, c=test_df['Year'],
                                cmap='viridis', s=50)
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Year', fontsize=12, fontweight='bold')
        else:
            scatter = ax1.scatter(test_df['Actual'], test_df['Predicted'], 
                                alpha=0.6, color=COLORS['ocean_blue'], s=50)
        
        # Add perfect prediction line
        max_val = max(test_df['Actual'].max(), test_df['Predicted'].max())
        min_val = min(test_df['Actual'].min(), test_df['Predicted'].min())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add best fit line
        slope, intercept = np.polyfit(test_df['Actual'], test_df['Predicted'], 1)
        x_line = np.linspace(min_val, max_val, 100)
        ax1.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2, 
                label=f'Best Fit (y = {slope:.3f}x + {intercept:.3f})')
        
        # Set labels and title
        ax1.set_title('Actual vs. Predicted Values', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])
        ax1.set_xlabel('Actual Species Richness', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted Species Richness', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        ax1.legend(fontsize=12)
        
        # Residual plot
        ax2 = plt.subplot(gs[1])
        ax2.set_facecolor(COLORS['background'])
        
        # Calculate residuals
        test_df['Residuals'] = test_df['Actual'] - test_df['Predicted']
        
        # Plot residuals vs predicted
        if has_year:
            ax2.scatter(test_df['Predicted'], test_df['Residuals'], 
                      alpha=0.6, c=test_df['Year'],
                      cmap='viridis', s=50)
        else:
            ax2.scatter(test_df['Predicted'], test_df['Residuals'], 
                      alpha=0.6, color=COLORS['ocean_blue'], s=50)
        
        # Add zero line
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax2.set_title('Residuals Analysis', fontsize=16, fontweight='bold', color=COLORS['dark_blue'])
        ax2.set_xlabel('Predicted Species Richness', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add annotation for model quality
        model_quality_text = (
            f"Residual Analysis Metrics:\n"
            f"• Mean Residual: {test_df['Residuals'].mean():.4f}\n"
            f"• Std. Deviation: {test_df['Residuals'].std():.4f}\n"
            f"• Max Error: {test_df['Residuals'].abs().max():.4f}"
        )
        
        ax2.annotate(model_quality_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                    fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "model_residuals_analysis.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Model residuals analysis saved.")
        
    except Exception as e:
        print(f"Error in model performance time series visualization: {e}")
        import traceback
        traceback.print_exc()

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
            last_lag1 = site_data['Lag1_Richness'].iloc[-1] if 'Lag1_Richness' in site_data.columns else last_value
            last_rolling_avg = site_data['RollingAvg2_Richness'].iloc[-1] if 'RollingAvg2_Richness' in site_data.columns else last_value
            
            # For each forecast year, predict the species richness
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
                if 'Lag1_Richness' in forecast_data.columns:
                    forecast_data['Lag1_Richness'] = current_value
                
                if 'Lag2_Richness' in forecast_data.columns:
                    forecast_data['Lag2_Richness'] = current_lag1
                
                if 'RollingAvg2_Richness' in forecast_data.columns:
                    forecast_data['RollingAvg2_Richness'] = current_rolling_avg
                
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
                
                # Handle potential NaN and infinity values
                X_pred = X_pred.fillna(0)
                
                # Check for infinity values and replace them
                inf_mask = np.isinf(X_pred)
                if inf_mask.any().any():
                    print(f"Found infinity values in prediction data for site {site}, replacing with finite values")
                    for col in X_pred.columns[inf_mask.any()]:
                        # Replace positive infinity with a large value
                        X_pred.loc[np.isposinf(X_pred[col]), col] = 1e6
                        # Replace negative infinity with a small value
                        X_pred.loc[np.isneginf(X_pred[col]), col] = -1e6
                
                # Clip extreme values
                X_pred = X_pred.clip(lower=-1e8, upper=1e8)
                
                try:
                    # Make base prediction
                    base_prediction = best_model.predict(X_pred)[0]
                    
                    # Add trend and random variation
                    random_factor = np.random.normal(0, site_std * 0.5)  # Reduced randomness
                    prediction = base_prediction + site_trend + random_factor
                    
                    # Ensure prediction is within reasonable bounds (0-20 species)
                    prediction = max(0, min(20, prediction))
                except Exception as e:
                    print(f"Error making prediction for site {site}: {e}")
                    continue
                
                # Create forecast entry
                forecast_entry = {
                    'Site_name': site,
                    'StationID': site_data['StationID'].iloc[0] if 'StationID' in site_data.columns else "Unknown",
                    'Year': year,
                    'Date': new_date,
                    'Forecast_Species_Richness': prediction,
                    'Last_Observed_Richness': last_value,
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
        yearly_summary = forecasts_df.groupby('Year')['Forecast_Species_Richness'].agg(['mean', 'min', 'max', 'std']).reset_index()
        print("\nForecast Summary by Year:")
        print(yearly_summary)
        
        if 'Subregion' in forecasts_df.columns:
            region_summary = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Species_Richness'].mean().reset_index()
            print("\nForecast Summary by Region:")
            print(region_summary)
        
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
        historical_yearly = historical_df.groupby('Year')['Species_Richness'].agg(['mean', 'std']).reset_index()
        
        # Calculate yearly averages from forecast data
        forecast_yearly = forecasts_df.groupby('Year')['Forecast_Species_Richness'].agg(['mean', 'std']).reset_index()
        
        # Plot historical data with error bands
        plt.plot(historical_yearly['Year'], historical_yearly['mean'], 
                color=COLORS['dark_blue'], marker='o', linestyle='-', linewidth=2, 
                label='Historical Species Richness')
        
        # Add error bands for historical data
        plt.fill_between(historical_yearly['Year'], 
                        historical_yearly['mean'] - historical_yearly['std'],
                        historical_yearly['mean'] + historical_yearly['std'],
                        color=COLORS['dark_blue'], alpha=0.2)
        
        # Plot forecast data with error bands
        plt.plot(forecast_yearly['Year'], forecast_yearly['mean'], 
                color=COLORS['coral'], marker='s', linestyle='--', linewidth=2, 
                label='Forecast Species Richness')
        
        # Add error bands for forecast data
        plt.fill_between(forecast_yearly['Year'], 
                        forecast_yearly['mean'] - forecast_yearly['std'],
                        forecast_yearly['mean'] + forecast_yearly['std'],
                        color=COLORS['coral'], alpha=0.2)
        
        # Mark the transition point
        last_historical_year = historical_yearly['Year'].max()
        plt.axvline(x=last_historical_year, color=COLORS['accent'], linestyle=':', 
                   label=f'Last Observed Year ({last_historical_year})')
        
        # Add labels and title
        plt.title('STONY CORAL SPECIES RICHNESS FORECAST (2024-2028)', 
                 fontweight='bold', fontsize=18, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Enhance the legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                  fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_forecast.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create regional forecast visualization if region data exists
        if 'Subregion' in forecasts_df.columns:
            plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
            
            # Calculate regional averages from historical data
            historical_regional = historical_df.groupby(['Year', 'Subregion'])['Species_Richness'].mean().reset_index()
            
            # Calculate regional averages from forecast data
            forecast_regional = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Species_Richness'].mean().reset_index()
            
            # Plot data for each region
            regions = historical_df['Subregion'].unique()
            region_colors = {
                'UK': COLORS['dark_blue'],
                'MK': COLORS['ocean_blue'],
                'LK': COLORS['light_blue'],
                'DT': COLORS['reef_green']
            }
            
            for region in regions:
                # Historical data for this region
                hist_region = historical_regional[historical_regional['Subregion'] == region]
                
                # Forecast data for this region
                fore_region = forecast_regional[forecast_regional['Subregion'] == region]
                
                # Plot historical data
                plt.plot(hist_region['Year'], hist_region['Species_Richness'], 
                        marker='o', linestyle='-', 
                        color=region_colors.get(region, COLORS['coral']), 
                        linewidth=2, markersize=6, 
                        label=f'{region} - Historical')
                
                # Plot forecast data
                plt.plot(fore_region['Year'], fore_region['Forecast_Species_Richness'], 
                        marker='s', linestyle='--', 
                        color=region_colors.get(region, COLORS['coral']), 
                        linewidth=2, markersize=6, 
                        label=f'{region} - Forecast')
            
            # Mark transition point
            plt.axvline(x=last_historical_year, color=COLORS['accent'], linestyle=':', alpha=0.7,
                       label=f'Last Observed Year ({last_historical_year})')
            
            # Add labels and title
            plt.title('REGIONAL FORECAST: STONY CORAL SPECIES RICHNESS (2024-2028)', 
                     fontweight='bold', fontsize=18, pad=20,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            plt.ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add gridlines
            plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Enhance the legend
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), 
                      edgecolor=COLORS['grid'])
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_regional_forecast.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
        
        # Create habitat-specific forecast visualization if habitat data exists
        if 'Habitat' in forecasts_df.columns:
            plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
            
            # Calculate habitat averages from historical data
            historical_habitat = historical_df.groupby(['Year', 'Habitat'])['Species_Richness'].mean().reset_index()
            
            # Calculate habitat averages from forecast data
            forecast_habitat = forecasts_df.groupby(['Year', 'Habitat'])['Forecast_Species_Richness'].mean().reset_index()
            
            # Plot data for each habitat
            habitats = historical_df['Habitat'].unique()
            habitat_colors = {
                'OS': COLORS['coral'],
                'OD': COLORS['sand'],
                'P': COLORS['reef_green'],
                'HB': COLORS['ocean_blue'],
                'BCP': COLORS['dark_blue']
            }
            
            for habitat in habitats:
                # Historical data for this habitat
                hist_habitat = historical_habitat[historical_habitat['Habitat'] == habitat]
                
                # Forecast data for this habitat
                fore_habitat = forecast_habitat[forecast_habitat['Habitat'] == habitat]
                
                # Plot historical data
                plt.plot(hist_habitat['Year'], hist_habitat['Species_Richness'], 
                        marker='o', linestyle='-', 
                        color=habitat_colors.get(habitat, COLORS['coral']), 
                        linewidth=2, markersize=6, 
                        label=f'{habitat} - Historical')
                
                # Plot forecast data
                plt.plot(fore_habitat['Year'], fore_habitat['Forecast_Species_Richness'], 
                        marker='s', linestyle='--', 
                        color=habitat_colors.get(habitat, COLORS['coral']), 
                        linewidth=2, markersize=6, 
                        label=f'{habitat} - Forecast')
            
            # Mark transition point
            plt.axvline(x=last_historical_year, color=COLORS['accent'], linestyle=':', alpha=0.7,
                       label=f'Last Observed Year ({last_historical_year})')
            
            # Add labels and title
            plt.title('HABITAT-SPECIFIC FORECAST: STONY CORAL SPECIES RICHNESS (2024-2028)', 
                     fontweight='bold', fontsize=18, pad=20,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            plt.ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add gridlines
            plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Enhance the legend
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), 
                      edgecolor=COLORS['grid'])
            
            # Save the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_habitat_forecast.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
        
        print("Forecast visualizations saved.")
        
    except Exception as e:
        print(f"Error visualizing forecasts: {e}")
        import traceback
        traceback.print_exc()

def analyze_reef_evolution(forecasts_df, historical_df, results_dir='13_Results'):
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
        yearly_avg = historical_df.groupby('Year')['Species_Richness'].mean().reset_index()
        yearly_avg['Annual_Change'] = yearly_avg['Species_Richness'].pct_change() * 100
        yearly_avg['5yr_Rolling_Avg'] = yearly_avg['Species_Richness'].rolling(window=5, min_periods=1).mean()
        
        # Regional yearly changes
        regional_yearly = historical_df.groupby(['Year', 'Subregion'])['Species_Richness'].mean().reset_index()
        regional_pivot = regional_yearly.pivot(index='Year', columns='Subregion', values='Species_Richness').reset_index()
        
        # Calculate regional growth rates
        for region in regional_yearly['Subregion'].unique():
            if region in regional_pivot.columns:
                regional_pivot[f'{region}_Change'] = regional_pivot[region].pct_change() * 100
        
        # Habitat specific yearly changes
        habitat_yearly = historical_df.groupby(['Year', 'Habitat'])['Species_Richness'].mean().reset_index()
        
        # Future trend estimation
        forecast_yearly = forecasts_df.groupby('Year')['Forecast_Species_Richness'].mean().reset_index()
        forecast_regional = forecasts_df.groupby(['Year', 'Subregion'])['Forecast_Species_Richness'].mean().reset_index()
        
        # 2. Create evolution trend visualization
        print("Creating evolution trend visualization...")
        
        # Create a figure for evolution trends
        plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
        
        # Plot historical data
        plt.plot(yearly_avg['Year'], yearly_avg['Species_Richness'], 
                marker='o', linestyle='-', color=COLORS['coral'], 
                linewidth=2.5, markersize=8, label='Historical Species Richness')
        
        # Plot 5-year rolling average
        plt.plot(yearly_avg['Year'], yearly_avg['5yr_Rolling_Avg'], 
                linestyle='--', color=COLORS['light_blue'], 
                linewidth=2.5, label='5-Year Rolling Average')
        
        # Add forecast data
        plt.plot(forecast_yearly['Year'], forecast_yearly['Forecast_Species_Richness'], 
                marker='s', linestyle='-', color=COLORS['dark_blue'], 
                linewidth=3, markersize=10, label='Forecast (2024-2028)')
        
        # Fill the area between the lines for visual effect
        last_historical_year = yearly_avg['Year'].max()
        last_historical_richness = yearly_avg[yearly_avg['Year'] == last_historical_year]['Species_Richness'].values[0]
        
        # Add forecast range (uncertainty)
        forecast_years = forecast_yearly['Year'].tolist()
        forecast_means = forecast_yearly['Forecast_Species_Richness'].tolist()
        
        # Calculate forecast confidence intervals
        forecast_std = forecasts_df.groupby('Year')['Forecast_Species_Richness'].std().reset_index()
        forecast_upper = forecast_yearly['Forecast_Species_Richness'] + 1.28 * forecast_std['Forecast_Species_Richness']
        forecast_lower = forecast_yearly['Forecast_Species_Richness'] - 1.28 * forecast_std['Forecast_Species_Richness']
        forecast_lower = forecast_lower.clip(lower=0)
        
        plt.fill_between(forecast_years, forecast_lower, forecast_upper, 
                         color=COLORS['dark_blue'], alpha=0.2, label='Forecast Uncertainty (80% CI)')
        
        # Add vertical line at the transition point
        plt.axvline(x=last_historical_year, color=COLORS['accent'], linestyle=':', 
                   label=f'Last Observed Year ({last_historical_year})')
        
        # Annotate the last historical data point
        plt.annotate(f"Last observation:\n{last_historical_richness:.2f} species",
                    xy=(last_historical_year, last_historical_richness),
                    xytext=(last_historical_year-2, last_historical_richness + 1),
                    fontsize=12, fontweight='bold',
                    arrowprops=dict(facecolor=COLORS['dark_blue'], shrink=0.05, width=2, headwidth=8))
        
        # Add trend line for historical data
        years = yearly_avg['Year'].values
        richness = yearly_avg['Species_Richness'].values
        hist_trend = np.polyfit(years, richness, 1)
        hist_line = np.poly1d(hist_trend)
        
        plt.plot(years, hist_line(years), '--', color=COLORS['sand'], 
                linewidth=2, label=f'Historical Trend (Slope: {hist_trend[0]:.4f}/year)')
        
        # Add trend line for forecast data
        forecast_years_values = forecast_yearly['Year'].values
        forecast_richness = forecast_yearly['Forecast_Species_Richness'].values
        forecast_trend = np.polyfit(forecast_years_values, forecast_richness, 1)
        forecast_line = np.poly1d(forecast_trend)
        
        plt.plot(forecast_years_values, forecast_line(forecast_years_values), '--', 
                color=COLORS['reef_green'], linewidth=2, 
                label=f'Forecast Trend (Slope: {forecast_trend[0]:.4f}/year)')
        
        # Add major historical events
        historical_events = {
            2005: "2005 Caribbean-wide\nbleaching event",
            2010: "2010 Cold water\nmass mortality",
            2014: "2014-2015 Global\nbleaching event",
            2017: "2017 Hurricane Irma",
            2019: "2019 SCTLD\ndisease peak"
        }
        
        for year, event in historical_events.items():
            if year >= yearly_avg['Year'].min() and year <= yearly_avg['Year'].max():
                plt.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
                plt.annotate(event, xy=(year, plt.ylim()[0] + 0.5), xytext=(year, plt.ylim()[0] + 0.2),
                            rotation=90, va='bottom', ha='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add title and labels
        plt.title('CORAL REEF EVOLUTION: STONY CORAL SPECIES RICHNESS TRENDS (1996-2028)', 
                 fontweight='bold', fontsize=18, pad=20, color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add gridlines
        plt.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Calculate overall change percentages
        historical_start_richness = yearly_avg.iloc[0]['Species_Richness']
        historical_end_richness = yearly_avg.iloc[-1]['Species_Richness']
        forecast_end_richness = forecast_yearly.iloc[-1]['Forecast_Species_Richness']
        
        historical_change_pct = ((historical_end_richness - historical_start_richness) / historical_start_richness) * 100
        forecast_change_pct = ((forecast_end_richness - historical_end_richness) / historical_end_richness) * 100
        
        # Add annotations for key metrics
        metrics_text = (
            f"Historical Change (1996-{last_historical_year}): {historical_change_pct:.1f}%\n"
            f"Forecast Change ({last_historical_year}-2028): {forecast_change_pct:.1f}%\n"
            f"Historical Trend: {hist_trend[0]:.4f} species/year\n"
            f"Forecast Trend: {forecast_trend[0]:.4f} species/year"
        )
        
        plt.annotate(metrics_text, xy=(0.02, 0.96), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                    fontsize=11, va='top', ha='left')
        
        # Enhance the legend
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, 
                  fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                  ncol=3, edgecolor=COLORS['grid'])
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_evolution.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Create a heatmap of historical and forecasted data
        print("Creating richness heatmap visualization...")
        
        # Combine historical and forecast data
        historical_pivot = historical_df.pivot_table(
            index='Year', 
            columns='Subregion', 
            values='Species_Richness', 
            aggfunc='mean'
        )
        
        forecast_pivot = forecasts_df.pivot_table(
            index='Year', 
            columns='Subregion', 
            values='Forecast_Species_Richness', 
            aggfunc='mean'
        )
        
        # Create combined dataframe
        combined_pivot = pd.concat([historical_pivot, forecast_pivot])
        
        # Create the heatmap
        plt.figure(figsize=(14, 10), facecolor=COLORS['background'])
        
        # Define a custom colormap that aligns with the project colors
        coral_cmap = LinearSegmentedColormap.from_list(
            'coral_cmap', 
            [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
        )
        
        # Create heatmap
        ax = sns.heatmap(combined_pivot, cmap=coral_cmap, annot=True, fmt='.1f', 
                         linewidths=0.5, cbar_kws={'label': 'Mean Species Richness'})
        
        # Add a horizontal line to separate historical and forecast data
        historical_years = historical_pivot.shape[0]
        ax.axhline(y=historical_years, color='black', linewidth=2)
        
        # Add labels and title
        plt.title('STONY CORAL SPECIES RICHNESS BY REGION (1996-2028)', 
                 fontweight='bold', fontsize=18, pad=20, color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
        plt.ylabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add annotation to indicate the separation
        plt.annotate('Historical Data', xy=(-0.1, historical_years/2), 
                    xycoords=('axes fraction', 'data'), rotation=90, 
                    fontsize=12, fontweight='bold', va='center', ha='center')
        
        plt.annotate('Forecast Data', xy=(-0.1, historical_years + (forecast_pivot.shape[0]/2)), 
                    xycoords=('axes fraction', 'data'), rotation=90, 
                    fontsize=12, fontweight='bold', va='center', ha='center')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_heatmap.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Coral reef evolution analysis visualizations saved.")
        
    except Exception as e:
        print(f"Error in reef evolution analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""
    print("\n=== Stony Coral Species Richness Forecasting Analysis ===\n")
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data_dict = load_and_preprocess_data()
        
        # Analyze time series patterns
        print("\nAnalyzing temporal patterns...")
        ts_analysis = analyze_time_series_patterns(data_dict)
        
        # Engineer features
        try:
            print("\nEngineering features...")
            feature_dict = engineer_features(data_dict['species_df'], data_dict['stations_df'], data_dict['temperature_df'])
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
            
            # Visualize model performance
            visualize_model_performance(model_results)
            
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
            print("\n=== Generating Stony Coral Species Richness Forecasts (2024-2028) ===")
            forecasts_df = generate_forecasts(feature_dict, model_results)
            
            if len(forecasts_df) > 0:
                # Save forecast results to CSV
                forecast_file = os.path.join(results_dir, "stony_coral_species_richness_forecasts.csv")
                forecasts_df.to_csv(forecast_file, index=False)
                print(f"Forecast results saved to {forecast_file}")
                
                # Save the best model for future use
                model_file = os.path.join(results_dir, f"stony_coral_richness_{best_model_name.lower().replace(' ', '_')}_model.pkl")
                joblib.dump(model_results['best_model']['model'], model_file)
                print(f"Best model saved to {model_file}")
                
                # Visualize forecasts
                visualize_forecasts(forecasts_df, data_dict['species_df'])
                
                # Perform detailed coral reef evolution analysis
                analyze_reef_evolution(forecasts_df, data_dict['species_df'])
                
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
    
    print("\n=== Analysis Complete ===\n")
    
    return data_dict, ts_analysis, feature_dict, model_results

# Execute main function if script is run directly
if __name__ == "__main__":
    main() 