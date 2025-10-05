"""
07_octocoral_temperature_correlations.py - Analysis of correlations between octocoral density, water temperature, and environmental factors

This script analyzes the relationships between octocoral density and water temperature data
in the Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP). It explores correlations,
seasonal patterns, impact of temperature fluctuations on octocoral communities, and potential
threshold effects.

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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.patheffects as pe  # For enhanced visual effects
from matplotlib.patches import Patch
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import math
import matplotlib.ticker as ticker
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "07_Results"
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
plt.rcParams['axes.titlepad'] = 12  # Add more padding to title
plt.rcParams['axes.labelpad'] = 8   # Add more padding to axis labels
plt.rcParams['axes.spines.top'] = False  # Remove top spine for cleaner look
plt.rcParams['axes.spines.right'] = False  # Remove right spine for cleaner look

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

# Create a custom colormap for octocoral visualization
octocoral_cmap = LinearSegmentedColormap.from_list(
    'octocoral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Create a temperature colormap
temperature_cmap = LinearSegmentedColormap.from_list(
    'temperature_cmap',
    ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026']
)

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP dataset for octocoral density and temperature analysis.
    
    Returns:
        tuple: (octo_df, temp_df, stations_df, species_cols) - Preprocessed DataFrames and species columns
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Octocoral Density dataset
        octo_df = pd.read_csv("CREMP_CSV_files/CREMP_OCTO_Summaries_2023_Density.csv")
        print(f"Octocoral density data loaded successfully with {len(octo_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Load temperature data
        try:
            temp_df = pd.read_csv("CREMP_CSV_files/CREMP_Temperatures_2023.csv")
            print(f"Temperature data loaded successfully with {len(temp_df)} rows")
        except Exception as e:
            print(f"Error loading temperature data: {str(e)}")
            temp_df = None
            
        # Convert date column to datetime format
        octo_df['Date'] = pd.to_datetime(octo_df['Date'])
        
        # Extract just the year for easier grouping
        octo_df['Year'] = octo_df['Year'].astype(int)
        
        # Process temperature data if available
        if temp_df is not None:
            # Ensure columns are in the expected format
            if 'Year' not in temp_df.columns and 'Date' in temp_df.columns:
                temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                temp_df['Year'] = temp_df['Date'].dt.year
                temp_df['Month'] = temp_df['Date'].dt.month
                temp_df['Day'] = temp_df['Date'].dt.day
            elif 'Year' in temp_df.columns and 'Month' in temp_df.columns and 'Day' in temp_df.columns:
                # Create a date column if it doesn't exist
                if 'Date' not in temp_df.columns:
                    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])
        
        # Get list of all octocoral species columns (excluding metadata columns)
        metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID', 'Total_Octocorals']
        species_cols = [col for col in octo_df.columns if col not in metadata_cols]
        
        print(f"\nIdentified {len(species_cols)} octocoral species in the dataset")
        
        # Calculate total density for each station if not already present
        if 'Total_Octocorals' in octo_df.columns:
            # Fill missing values in Total_Octocorals by summing species columns
            octo_df.loc[octo_df['Total_Octocorals'].isna(), 'Total_Octocorals'] = octo_df.loc[octo_df['Total_Octocorals'].isna(), species_cols].sum(axis=1, skipna=True)
        else:
            # Create Total_Octocorals column by summing all species columns
            octo_df['Total_Octocorals'] = octo_df[species_cols].sum(axis=1, skipna=True)
        
        print(f"\nData loaded: {len(octo_df)} octocoral records from {octo_df['Year'].min()} to {octo_df['Year'].max()}")
        
        return octo_df, temp_df, stations_df, species_cols
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to prepare temperature data for analysis
def prepare_temperature_data(temp_df, octo_df):
    """
    Process temperature data to calculate metrics for correlation with octocoral data.
    
    Args:
        temp_df (DataFrame): Raw temperature measurements
        octo_df (DataFrame): Octocoral density data
        
    Returns:
        DataFrame: Processed temperature data with metrics per site and year
    """
    print("Preparing temperature data for analysis...")
    
    if temp_df is None:
        print("Temperature data is not available")
        return None
    
    # Ensure temperature data has datetime column
    if 'Date' not in temp_df.columns and 'Time' in temp_df.columns:
        # Try to create a full datetime column
        try:
            temp_df['DateTime'] = pd.to_datetime(
                temp_df['Year'].astype(str) + '-' + 
                temp_df['Month'].astype(str) + '-' + 
                temp_df['Day'].astype(str) + ' ' + 
                temp_df['Time']
            )
        except:
            # If Time column format is problematic, just use the date
            temp_df['DateTime'] = pd.to_datetime(
                temp_df['Year'].astype(str) + '-' + 
                temp_df['Month'].astype(str) + '-' + 
                temp_df['Day'].astype(str)
            )
    elif 'Date' in temp_df.columns:
        temp_df['DateTime'] = pd.to_datetime(temp_df['Date'])
    
    # Make sure we have temperature in Celsius
    if 'TempC' not in temp_df.columns and 'TempF' in temp_df.columns:
        temp_df['TempC'] = (temp_df['TempF'] - 32) * 5/9
    
    # Group temperature data by site and year to calculate metrics
    # We'll calculate multiple metrics to see which one correlates best with octocoral density
    metrics = []
    
    for site_id in temp_df['SiteID'].unique():
        site_data = temp_df[temp_df['SiteID'] == site_id]
        
        for year in site_data['Year'].unique():
            year_data = site_data[site_data['Year'] == year]
            
            if len(year_data) > 0:
                metrics.append({
                    'SiteID': site_id,
                    'Year': year,
                    'MeanTemp': year_data['TempC'].mean(),
                    'MaxTemp': year_data['TempC'].max(),
                    'MinTemp': year_data['TempC'].min(),
                    'TempRange': year_data['TempC'].max() - year_data['TempC'].min(),
                    'StdTemp': year_data['TempC'].std(),
                    'Days_Above_30C': len(year_data[year_data['TempC'] > 30]),
                    'Days_Above_31C': len(year_data[year_data['TempC'] > 31]),
                    'Days_Below_20C': len(year_data[year_data['TempC'] < 20]),
                    'Q90_Temp': year_data['TempC'].quantile(0.9),  # 90th percentile
                    'Q10_Temp': year_data['TempC'].quantile(0.1),  # 10th percentile
                })
    
    temp_metrics_df = pd.DataFrame(metrics)
    
    print(f"Created temperature metrics for {len(temp_metrics_df)} site-year combinations")
    
    return temp_metrics_df

# Function to analyze basic correlations between octocoral density and temperature
def analyze_basic_correlations(octo_df, temp_metrics_df, species_cols):
    """
    Analyze and visualize basic correlations between octocoral density and temperature metrics.
    
    Args:
        octo_df (DataFrame): Processed octocoral density data
        temp_metrics_df (DataFrame): Processed temperature metrics
        species_cols (list): List of octocoral species columns
        
    Returns:
        DataFrame: Correlation results
    """
    print("Analyzing basic correlations between octocoral density and temperature...")
    
    if temp_metrics_df is None:
        print("Temperature metrics not available for correlation analysis")
        return None
    
    # Merge octocoral data with temperature metrics
    merged_df = octo_df.copy()
    merged_df = pd.merge(
        merged_df,
        temp_metrics_df,
        on=['SiteID', 'Year'],
        how='inner'
    )
    
    print(f"Merged dataset has {len(merged_df)} records")
    
    # Calculate correlations for total octocoral density
    correlation_data = []
    temp_metrics = ['MeanTemp', 'MaxTemp', 'MinTemp', 'TempRange', 'StdTemp', 
                    'Days_Above_30C', 'Days_Above_31C', 'Days_Below_20C', 
                    'Q90_Temp', 'Q10_Temp']
    
    for metric in temp_metrics:
        if metric in merged_df.columns:
            # Calculate correlation with total octocoral density
            valid_data = merged_df.dropna(subset=[metric, 'Total_Octocorals'])
            if len(valid_data) > 10:  # Ensure we have enough data points
                corr, p_value = stats.pearsonr(valid_data[metric], valid_data['Total_Octocorals'])
                correlation_data.append({
                    'Temperature_Metric': metric,
                    'Coral_Metric': 'Total Octocoral Density',
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'N': len(valid_data)
                })
    
    # Calculate correlations for each octocoral species
    for species in species_cols:
        for metric in temp_metrics:
            if metric in merged_df.columns:
                valid_data = merged_df.dropna(subset=[metric, species])
                if len(valid_data) > 10:  # Ensure we have enough data points
                    corr, p_value = stats.pearsonr(valid_data[metric], valid_data[species])
                    correlation_data.append({
                        'Temperature_Metric': metric,
                        'Coral_Metric': species.replace('_', ' '),
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05,
                        'N': len(valid_data)
                    })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_data)
    
    # Sort by absolute correlation value
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
    
    # Create visualization: correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Select a subset of most common octocoral species and most important temperature metrics
    # Focus on key species with significant correlations
    key_species = ['Pseudoplexaura porosa', 'Eunicea flexuosa', 'Gorgonia ventalina', 
                  'Pseudopterogorgia americana', 'Pseudopterogorgia bipinnata', 'Total Octocoral Density']
    key_species = [sp.replace(' ', '_') for sp in key_species]
    key_species = [sp for sp in key_species if sp in octo_df.columns or sp == 'Total_Octocorals']
    
    # Select key temperature metrics
    key_metrics = ['Days_Above_30C', 'Days_Above_31C', 'Days_Below_20C', 'MaxTemp', 'MeanTemp',
                  'MinTemp', 'Q10_Temp', 'Q90_Temp', 'StdTemp', 'TempRange']
    
    # Filter correlation data for key species and metrics
    filtered_corrs = corr_df[
        (corr_df['Coral_Metric'].isin([sp.replace('_', ' ') for sp in key_species]) | 
         (corr_df['Coral_Metric'] == 'Total Octocoral Density')) &
        corr_df['Temperature_Metric'].isin(key_metrics)
    ]
    
    # Create a pivot table for plotting - selecting only species that have correlation data
    pivot_species = [sp.replace('_', ' ') for sp in key_species if 
                    sp.replace('_', ' ') in filtered_corrs['Coral_Metric'].values or 
                    (sp == 'Total_Octocorals' and 'Total Octocoral Density' in filtered_corrs['Coral_Metric'].values)]
    
    # Replace Total_Octocorals with its display name
    if 'Total_Octocorals' in key_species:
        pivot_species = [sp if sp != 'Total_Octocorals' else 'Total Octocoral Density' for sp in pivot_species]
    
    # Create the pivot table
    heatmap_data = filtered_corrs.pivot_table(
        index='Coral_Metric', 
        columns='Temperature_Metric', 
        values='Correlation',
        aggfunc='first'  # Take the first value if there are duplicates
    )
    
    # Filter to include only the species we care about
    if len(pivot_species) > 0:
        heatmap_data = heatmap_data.loc[heatmap_data.index.isin(pivot_species)]
    
    # Create the mask for missing values
    mask = heatmap_data.isna()
    
    # Create a custom colormap with white for missing values
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create the heatmap with improved formatting for better clarity
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        center=0, 
        linewidths=0.8,  # Increased for better cell separation
        linecolor='white',  # White gridlines
        cbar_kws={'label': 'Correlation Coefficient'},
        fmt='.2f', 
        ax=ax,
        mask=mask,
        annot_kws={"size": 12, "weight": "bold"}  # Larger, bolder correlation numbers
    )
    
    # Style the heatmap
    ax.set_title('Correlation Between Octocoral Density and Temperature Metrics', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Temperature Metric', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Species / Total Density', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add asterisks for significant correlations in a way that doesn't overlap
    # Create a significance mask
    significance_mask = np.zeros_like(heatmap_data, dtype=bool)
    
    for i, coral_metric in enumerate(heatmap_data.index):
        for j, temp_metric in enumerate(heatmap_data.columns):
            significant_entries = filtered_corrs[
                (filtered_corrs['Coral_Metric'] == coral_metric) & 
                (filtered_corrs['Temperature_Metric'] == temp_metric) &
                (filtered_corrs['Significant'] == True)
            ]
            
            if len(significant_entries) > 0:
                significance_mask[i, j] = True
    
    # Add asterisks to significant cells without overlapping the correlation values
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if significance_mask[i, j] and not np.isnan(heatmap_data.iloc[i, j]):
                # Place asterisk at bottom of cell to avoid overlapping with correlation value
                ax.text(j + 0.5, i + 0.75, '*', fontsize=18, ha='center', va='center',
                       color='black', weight='bold',
                       path_effects=[pe.withStroke(linewidth=1, foreground='white')])
    
    # Add a note about significance
    ax.text(0.6, -0.20, '* indicates statistically significant correlation (p < 0.05)',
           ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(results_dir, "octocoral_temperature_correlation_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Basic correlation analysis saved.")
    
    return corr_df, merged_df

def visualize_key_correlations(merged_df, corr_df, species_cols):
    """
    Create detailed scatter plots for the strongest correlations between 
    octocoral density and temperature metrics.
    
    Args:
        merged_df (DataFrame): Merged data with octocoral density and temperature metrics
        corr_df (DataFrame): Correlation analysis results
        species_cols (list): List of octocoral species columns
    """
    print("Creating detailed visualizations for key correlations...")
    
    # Get the top correlations (by absolute value)
    significant_corrs = corr_df[corr_df['Significant'] == True].copy()
    significant_corrs['Abs_Correlation'] = significant_corrs['Correlation'].abs()
    top_corrs = significant_corrs.sort_values('Abs_Correlation', ascending=False).head(6)
    
    if len(top_corrs) == 0:
        print("No significant correlations found for detailed visualization")
        return
    
    # Create a figure with subplots for each top correlation
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor=COLORS['background'])
    axs = axs.flatten()
    
    # Create scatter plots with regression lines for top correlations
    for i, (_, row) in enumerate(top_corrs.iterrows()):
        if i >= 6:  # Limit to 6 plots
            break
            
        temp_metric = row['Temperature_Metric']
        coral_metric = row['Coral_Metric']
        correlation = row['Correlation']
        p_value = row['P_Value']
        
        # Get the actual coral metric column name (without space replacement)
        if coral_metric == 'Total Octocoral Density':
            coral_col = 'Total_Octocorals'
        else:
            coral_col = coral_metric.replace(' ', '_')
        
        # Extract the data for the scatter plot
        valid_data = merged_df.dropna(subset=[temp_metric, coral_col])
        
        # Create the scatter plot
        ax = axs[i]
        ax.set_facecolor(COLORS['background'])
        
        # First create a simple scatter plot
        ax.scatter(
            valid_data[temp_metric], 
            valid_data[coral_col],
            alpha=0.6, 
            s=70, 
            color=COLORS['coral'],
            edgecolor='white'
        )
        
        # Then add regression line
        # Calculate regression manually
        X = valid_data[temp_metric].values.reshape(-1, 1)
        y = valid_data[coral_col].values
        model = LinearRegression().fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        # Plot the regression line
        ax.plot(x_range, y_pred, color=COLORS['dark_blue'], linewidth=2)
        
        # Enhance the plot
        ax.set_title(f"{coral_metric} vs {temp_metric}", 
                    fontweight='bold', fontsize=16, pad=15,
                    color=COLORS['dark_blue'])
        
        ax.set_xlabel(f"{temp_metric}", fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel(f"{coral_metric} (colonies/m²)", fontweight='bold', fontsize=14, labelpad=10)
        
        # Add correlation statistics
        stats_text = (
            f"Correlation: {correlation:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"n = {row['N']}"
        )
        
        # Add the stats box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space - adjust position based on correlation
        x_pos = 0.05 if correlation > 0 else 0.65
        ax.text(x_pos, 0.95, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Handle case with fewer than 6 significant correlations
    for i in range(len(top_corrs), 6):
        axs[i].set_visible(False)
    
    # Add overall title
    fig.suptitle('KEY CORRELATIONS BETWEEN OCTOCORAL DENSITY AND TEMPERATURE METRICS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "octocoral_temperature_key_correlations.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Key correlations visualization saved.")

def analyze_temperature_trends(temp_df):
    """
    Analyze and visualize temperature trends over time.
    
    Args:
        temp_df (DataFrame): Raw temperature data
    """
    print("Analyzing temperature trends...")
    
    if temp_df is None:
        print("Temperature data not available for trend analysis")
        return
    
    # Ensure we have datetime information
    if 'DateTime' not in temp_df.columns:
        if 'Date' in temp_df.columns:
            temp_df['DateTime'] = pd.to_datetime(temp_df['Date'])
        else:
            try:
                temp_df['DateTime'] = pd.to_datetime(
                    temp_df['Year'].astype(str) + '-' + 
                    temp_df['Month'].astype(str) + '-' + 
                    temp_df['Day'].astype(str)
                )
            except:
                print("Could not create DateTime column for temperature trend analysis")
                return
    
    # Make sure we have temperature in Celsius
    if 'TempC' not in temp_df.columns and 'TempF' in temp_df.columns:
        temp_df['TempC'] = (temp_df['TempF'] - 32) * 5/9
    
    # Calculate daily averages across all sites
    temp_df['Date'] = temp_df['DateTime'].dt.date
    daily_avg = temp_df.groupby('Date')['TempC'].agg(['mean', 'min', 'max']).reset_index()
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    daily_avg['Year'] = daily_avg['Date'].dt.year
    daily_avg['Month'] = daily_avg['Date'].dt.month
    
    # Create figure for temperature trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), facecolor=COLORS['background'], 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Set background colors
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Temperature time series
    # Daily min-max range
    ax1.fill_between(
        daily_avg['Date'], 
        daily_avg['min'], 
        daily_avg['max'], 
        alpha=0.3, 
        color=COLORS['light_blue'], 
        label='Daily Min-Max Range'
    )
    
    # Daily mean
    ax1.plot(
        daily_avg['Date'], 
        daily_avg['mean'], 
        color=COLORS['dark_blue'], 
        linewidth=1.5, 
        alpha=0.8,
        label='Daily Mean Temperature'
    )
    
    # Add a 30-day rolling average for clarity
    daily_avg['rolling_mean'] = daily_avg['mean'].rolling(window=30, center=True).mean()
    ax1.plot(
        daily_avg['Date'], 
        daily_avg['rolling_mean'], 
        color=COLORS['coral'], 
        linewidth=3, 
        label='30-Day Rolling Average'
    )
    
    # Add horizontal lines for critical temperature thresholds
    ax1.axhline(y=30, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Coral Stress Threshold (30°C)')
    
    # Style the plot
    ax1.set_title('WATER TEMPERATURE TRENDS IN FLORIDA KEYS (2011-2023)', 
                 fontweight='bold', fontsize=20, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax1.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Format the x-axis to show years
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Add legend with better positioning and styling
    legend = ax1.legend(loc='upper right', frameon=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor(COLORS['grid'])
    
    # Plot 2: Monthly temperature boxplots
    # Calculate monthly statistics
    monthly_data = []
    for year in daily_avg['Year'].unique():
        for month in range(1, 13):
            month_data = daily_avg[(daily_avg['Year'] == year) & (daily_avg['Month'] == month)]
            if len(month_data) > 0:
                monthly_data.append({
                    'Year': year,
                    'Month': month,
                    'MonthName': datetime(2000, month, 1).strftime('%b'),
                    'MeanTemp': month_data['mean'].mean(),
                    'MedianTemp': month_data['mean'].median(),
                    'MaxTemp': month_data['max'].max(),
                    'MinTemp': month_data['min'].min(),
                    'StdTemp': month_data['mean'].std()
                })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Create a seasonal pattern plot
    sns.boxplot(
        x='MonthName', 
        y='MeanTemp', 
        data=monthly_df,
        ax=ax2,
        hue='MonthName',  # Add hue to match the palette
        palette='coolwarm',
        width=0.7,
        fliersize=3,
        legend=False  # Hide the legend since it's redundant
    )
    
    # Add a line connecting the medians to show the seasonal pattern
    medians = monthly_df.groupby('Month')['MeanTemp'].median().reset_index()
    medians = medians.sort_values('Month')
    
    ax2.plot(
        range(len(medians)), 
        medians['MeanTemp'], 
        'o-', 
        color=COLORS['dark_blue'], 
        linewidth=2.5,
        markersize=8,
        markerfacecolor=COLORS['coral'],
        markeredgecolor='white',
        markeredgewidth=1.5
    )
    
    # Style the seasonal plot
    ax2.set_title('Monthly Temperature Distribution', 
                 fontweight='bold', fontsize=16, pad=15)
    
    ax2.set_xlabel('Month', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Mean Temperature (°C)', fontweight='bold', fontsize=14, labelpad=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    
    plt.savefig(os.path.join(results_dir, "water_temperature_trends.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temperature trend analysis saved.")

def analyze_regional_temperature_effects(merged_df, octo_df, temp_metrics_df):
    """
    Analyze how temperature affects octocoral density across different regions.
    
    Args:
        merged_df (DataFrame): Merged data with octocoral density and temperature metrics
        octo_df (DataFrame): Octocoral density data
        temp_metrics_df (DataFrame): Temperature metrics data
    """
    print("Analyzing regional temperature effects...")
    
    if merged_df is None or temp_metrics_df is None:
        print("Required data not available for regional temperature effects analysis")
        return
    
    # Create a figure for regional comparisons
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), facecolor=COLORS['background'])
    
    # Define regions and colors for consistent plotting
    regions = ['UK', 'MK', 'LK']
    region_names = {'UK': 'Upper Keys', 'MK': 'Middle Keys', 'LK': 'Lower Keys'}
    region_colors = {'UK': COLORS['dark_blue'], 'MK': COLORS['ocean_blue'], 'LK': COLORS['light_blue']}
    
    # Find the temperature metric with the strongest correlation to total octocoral density
    # Filter to only numeric columns for correlation analysis
    numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
    if 'Total_Octocorals' in numeric_cols:
        temp_metrics_in_data = [col for col in temp_metrics_df.columns if col in numeric_cols]
        
        # Calculate correlations only with numeric columns
        numeric_df = merged_df[numeric_cols]
        corr_with_total = numeric_df.corr()['Total_Octocorals'].abs().sort_values(ascending=False)
        
        # Get top temperature metric
        top_temp_metrics = [col for col in corr_with_total.index if col in temp_metrics_in_data]
        top_temp_metric = top_temp_metrics[0] if len(top_temp_metrics) > 0 else 'MeanTemp'
    else:
        # Default to MeanTemp if Total_Octocorals is not in numeric columns
        top_temp_metric = 'MeanTemp'
    
    # Plot 1: Temperature distribution by region
    for i, region in enumerate(regions):
        ax = axs[i]
        ax.set_facecolor(COLORS['background'])
        
        # Filter data for the region
        region_data = merged_df[merged_df['Subregion'] == region]
        
        if len(region_data) == 0:
            ax.text(0.5, 0.5, f"No data available for {region_names[region]}", 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
            continue
        
        # Create scatter plot
        ax.scatter(
            region_data[top_temp_metric], 
            region_data['Total_Octocorals'],
            alpha=0.7, 
            s=100, 
            color=region_colors[region],
            edgecolor='white'
        )
        
        # Add regression line
        # Calculate regression manually
        X = region_data[top_temp_metric].values.reshape(-1, 1)
        y = region_data['Total_Octocorals'].values
        model = LinearRegression().fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        # Plot the regression line
        ax.plot(x_range, y_pred, color=COLORS['coral'], linewidth=2.5)
        
        # Calculate correlation for this region
        corr, p_value = stats.pearsonr(
            region_data[top_temp_metric].values, 
            region_data['Total_Octocorals'].values
        )
        
        # Add correlation statistics
        stats_text = (
            f"Correlation: {corr:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"n = {len(region_data)}"
        )
        
        # Add the stats box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=region_colors[region], linewidth=2)
        
        # Position the text box
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Style the plot
        ax.set_title(f"{region_names[region]}", 
                    fontweight='bold', fontsize=18, pad=15,
                    color=region_colors[region],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        ax.set_xlabel(f"{top_temp_metric}", fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Total Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add overall title
    fig.suptitle(f'REGIONAL VARIATIONS IN OCTOCORAL DENSITY RESPONSE TO TEMPERATURE ({top_temp_metric})', 
                fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                y=0.98,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "octocoral_temperature_regional_variation.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional temperature effects analysis saved.")

def analyze_habitat_temperature_effects(merged_df, octo_df, temp_metrics_df):
    """
    Analyze how temperature affects octocoral density across different habitat types.
    
    Args:
        merged_df (DataFrame): Merged data with octocoral density and temperature metrics
        octo_df (DataFrame): Octocoral density data
        temp_metrics_df (DataFrame): Temperature metrics data
    """
    print("Analyzing habitat-based temperature effects...")
    
    if merged_df is None or temp_metrics_df is None:
        print("Required data not available for habitat temperature effects analysis")
        return
    
    # Identify the habitat types present in the data
    habitats = merged_df['Habitat'].unique().tolist()
    
    # Define a mapping for full habitat names
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Define colors for habitats
    habitat_colors = {
        'OS': COLORS['coral'],
        'OD': COLORS['sand'],
        'P': COLORS['reef_green'],
        'HB': COLORS['ocean_blue'],
        'BCP': COLORS['dark_blue']
    }
    
    # Find the temperature metric with the strongest correlation to total octocoral density
    # Filter to only numeric columns for correlation analysis
    numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
    if 'Total_Octocorals' in numeric_cols:
        temp_metrics_in_data = [col for col in temp_metrics_df.columns if col in numeric_cols]
        
        # Calculate correlations only with numeric columns
        numeric_df = merged_df[numeric_cols]
        corr_with_total = numeric_df.corr()['Total_Octocorals'].abs().sort_values(ascending=False)
        
        # Get top temperature metric
        top_temp_metrics = [col for col in corr_with_total.index if col in temp_metrics_in_data]
        top_temp_metric = top_temp_metrics[0] if len(top_temp_metrics) > 0 else 'MeanTemp'
    else:
        # Default to MeanTemp if Total_Octocorals is not in numeric columns
        top_temp_metric = 'MeanTemp'
    
    # Create figure for analysis
    num_habitats = len(habitats)
    fig_rows = (num_habitats + 1) // 2  # Calculate rows needed (2 plots per row)
    
    fig, axs = plt.subplots(fig_rows, 2, figsize=(20, 6 * fig_rows), facecolor=COLORS['background'])
    axs = axs.flatten() if num_habitats > 1 else [axs]
    
    # Create scatter plots for each habitat
    for i, habitat in enumerate(habitats):
        if i >= len(axs):  # Safety check
            break
            
        ax = axs[i]
        ax.set_facecolor(COLORS['background'])
        
        # Filter data for the habitat
        habitat_data = merged_df[merged_df['Habitat'] == habitat]
        
        if len(habitat_data) == 0:
            ax.text(0.5, 0.5, f"No data available for {habitat_names.get(habitat, habitat)}", 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=ax.transAxes)
            continue
        
        # Create scatter plot
        scatter = ax.scatter(
            habitat_data[top_temp_metric], 
            habitat_data['Total_Octocorals'],
            alpha=0.7, 
            s=100, 
            c=habitat_data['Year'],  # Color by year to see temporal patterns
            cmap='viridis',
            edgecolor='white'
        )
        
        # Add regression line
        # Calculate regression manually
        X = habitat_data[top_temp_metric].values.reshape(-1, 1)
        y = habitat_data['Total_Octocorals'].values
        model = LinearRegression().fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        # Plot the regression line
        line_color = habitat_colors.get(habitat, COLORS['dark_blue'])
        ax.plot(x_range, y_pred, color=line_color, linewidth=2.5)
        
        # Calculate correlation for this habitat
        corr, p_value = stats.pearsonr(
            habitat_data[top_temp_metric].values, 
            habitat_data['Total_Octocorals'].values
        )
        
        # Add correlation statistics
        stats_text = (
            f"Correlation: {corr:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"n = {len(habitat_data)}"
        )
        
        # Add the stats box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=line_color, linewidth=2)
        
        # Position the text box
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Style the plot
        habitat_display_name = habitat_names.get(habitat, habitat)
        ax.set_title(f"{habitat_display_name}", 
                    fontweight='bold', fontsize=18, pad=15,
                    color=line_color,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        ax.set_xlabel(f"{top_temp_metric}", fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Total Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add colorbar to show year
        if i == 0:  # Only add colorbar to first plot
            cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label('Year', fontweight='bold', fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Hide any unused subplots
    for i in range(num_habitats, len(axs)):
        axs[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'HABITAT VARIATIONS IN OCTOCORAL DENSITY RESPONSE TO TEMPERATURE ({top_temp_metric})', 
                fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                y=0.98,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "octocoral_temperature_habitat_variation.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Habitat temperature effects analysis saved.")

def analyze_seasonal_temperature_effects(merged_df, octo_df, temp_df):
    """
    Analyze seasonal patterns of temperature effects on octocoral density.
    
    Args:
        merged_df (DataFrame): Merged data with octocoral density and temperature metrics
        octo_df (DataFrame): Octocoral density data
        temp_df (DataFrame): Raw temperature data
    """
    print("Analyzing seasonal temperature effects...")
    
    if merged_df is None or temp_df is None:
        print("Required data not available for seasonal temperature effects analysis")
        return
    
    # Ensure we have date information in the merged data
    if 'Date' not in merged_df.columns:
        print("Date information not available in merged data")
        return
    
    # Extract month information
    merged_df['Month'] = pd.to_datetime(merged_df['Date']).dt.month
    merged_df['MonthName'] = pd.to_datetime(merged_df['Date']).dt.strftime('%b')
    
    # Create figure for analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=COLORS['background'])
    
    # Plot 1: Distribution of octocoral density by month
    # Calculate monthly statistics
    monthly_stats = merged_df.groupby('Month')['Total_Octocorals'].agg(['mean', 'median', 'std', 'count']).reset_index()
    monthly_stats['MonthName'] = monthly_stats['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
    monthly_stats = monthly_stats.sort_values('Month')
    
    # Add standard error and confidence interval
    monthly_stats['se'] = monthly_stats['std'] / np.sqrt(monthly_stats['count'])
    monthly_stats['ci_95'] = 1.96 * monthly_stats['se']
    
    # Create bar plot with error bars
    bars = ax1.bar(
        monthly_stats['MonthName'], 
        monthly_stats['mean'],
        yerr=monthly_stats['ci_95'],
        color=[COLORS['ocean_blue'] if m in [6, 7, 8, 9] else COLORS['light_blue'] for m in monthly_stats['Month']],
        edgecolor=COLORS['dark_blue'],
        linewidth=1.5,
        alpha=0.8,
        capsize=5
    )
    
    # Highlight summer months
    ax1.axvspan(2.5, 6.5, alpha=0.2, color=COLORS['coral'], zorder=0)
    
    # Add month labels
    ax1.set_xticks(range(len(monthly_stats)))
    ax1.set_xticklabels(monthly_stats['MonthName'], rotation=45, ha='right')
    
    # Style the plot
    ax1.set_title('Octocoral Density by Month', 
                 fontweight='bold', fontsize=18, pad=15,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax1.set_xlabel('Month', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add annotations for summer months
    ax1.text(4.5, ax1.get_ylim()[1] * 0.95, 'Summer Months', 
             ha='center', va='top', fontsize=14, fontweight='bold', 
             color=COLORS['dark_blue'],
             bbox=dict(facecolor='white', alpha=0.7, edgecolor=COLORS['coral']))
    
    # Plot 2: Temperature vs Density by Season
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    # Add season to merged data
    merged_df['Season'] = merged_df['Month'].apply(get_season)
    
    # Define season colors
    season_colors = {
        'Winter': 'blue',
        'Spring': 'green',
        'Summer': 'red',
        'Fall': 'orange'
    }
    
    # Find a suitable temperature metric
    if 'MeanTemp' in merged_df.columns:
        temp_metric = 'MeanTemp'
    elif 'MaxTemp' in merged_df.columns:
        temp_metric = 'MaxTemp'
    else:
        # Use the first temperature metric available
        temp_metrics = [col for col in merged_df.columns if 'Temp' in col and col != 'TempF']
        temp_metric = temp_metrics[0] if len(temp_metrics) > 0 else None
    
    if temp_metric is None:
        ax2.text(0.5, 0.5, "No suitable temperature metric found", 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
    else:
        # Create scatter plot colored by season
        for season, group in merged_df.groupby('Season'):
            ax2.scatter(
                group[temp_metric], 
                group['Total_Octocorals'],
                alpha=0.7, 
                s=80, 
                color=season_colors[season],
                edgecolor='white',
                label=season
            )
        
        # Add regression line for each season
        for season, group in merged_df.groupby('Season'):
            if len(group) >= 10:  # Only add regression line if we have enough data
                X = group[temp_metric].values.reshape(-1, 1)
                y = group['Total_Octocorals'].values
                model = LinearRegression().fit(X, y)
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_pred = model.predict(x_range)
                
                # Plot the regression line
                ax2.plot(x_range, y_pred, color=season_colors[season], linewidth=2.5, linestyle='--')
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(X.flatten(), y)
                
                # Add correlation to legend
                ax2.scatter([], [], alpha=0, label=f"{season}: r={corr:.2f}, p={p_value:.3f}")
        
        # Style the plot
        ax2.set_title('Octocoral Density vs Temperature by Season', 
                     fontweight='bold', fontsize=18, pad=15,
                     color=COLORS['dark_blue'],
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        ax2.set_xlabel(f'Temperature ({temp_metric})', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend with better positioning and styling
        legend = ax2.legend(loc='upper right', frameon=True, fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor(COLORS['grid'])
    
    # Add overall title
    fig.suptitle('SEASONAL PATTERNS IN OCTOCORAL DENSITY AND TEMPERATURE RELATIONSHIP', 
                fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                y=0.98,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "octocoral_temperature_seasonal_effects.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Seasonal temperature effects analysis saved.")

# Main execution
def main():
    """Main execution function."""
    print("\n=== Octocoral Density and Temperature Correlation Analysis ===\n")
    
    # Load and preprocess data
    octo_df, temp_df, stations_df, species_cols = load_and_preprocess_data()
    
    # Prepare temperature data for analysis
    temp_metrics_df = prepare_temperature_data(temp_df, octo_df)
    
    # Analyze basic correlations
    corr_df, merged_df = analyze_basic_correlations(octo_df, temp_metrics_df, species_cols)
    
    # Visualize key correlations with detailed scatter plots
    visualize_key_correlations(merged_df, corr_df, species_cols)
    
    # Analyze temperature trends over time
    analyze_temperature_trends(temp_df)
    
    # Analyze regional variations in temperature effects
    analyze_regional_temperature_effects(merged_df, octo_df, temp_metrics_df)
    
    # Analyze habitat-based temperature effects
    analyze_habitat_temperature_effects(merged_df, octo_df, temp_metrics_df)
    
    # Analyze seasonal temperature effects
    analyze_seasonal_temperature_effects(merged_df, octo_df, temp_df)
    
    print("\n=== Analysis Complete ===\n")

# Execute main function if script is run directly
if __name__ == "__main__":
    main() 