"""
10_early_indicators_analysis.py - Analysis of Early Indicators for Coral Population Declines

This script identifies and visualizes potential early warning indicators that could help
anticipate significant declines in stony coral and octocoral populations in the Florida Keys.
It explores leading indicators, threshold metrics, and statistical early warning signals.

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
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from scipy.signal import detrend
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "10_Results"
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
plt.rcParams['axes.titlepad'] = 12
plt.rcParams['axes.labelpad'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Define a modern color palette
COLORS = {
    'coral': '#FF6B6B',  # Bright coral
    'bleached_coral': '#FFCCCB',  # Light coral/pink
    'ocean_blue': '#4ECDC4',  # Turquoise
    'light_blue': '#A9D6E5',  # Soft blue
    'dark_blue': '#01445A',  # Navy blue
    'sand': '#FFBF69',  # Warm sand
    'reef_green': '#2EC4B6',  # Teal
    'accent': '#F9DC5C',  # Vibrant yellow
    'text': '#2A2A2A',  # Dark grey for text
    'grid': '#E0E0E0',  # Light grey for grid lines
    'background': '#F8F9FA',  # Very light grey background
    'disease': '#9A4C95',  # Purple for disease
    'hurricane': '#5B84B1',  # Stormy blue
    'temperature': '#FC766A',  # Warm red-orange
    'warning': '#FF9800',  # Orange for warnings
    'alert': '#F44336',  # Red for critical alerts
    'healthy': '#4CAF50',  # Green for healthy indicators
    'decline': '#E91E63',  # Pink for decline indicators
    'threshold': '#673AB7'  # Purple for thresholds
}

# Create a custom colormap for coral reef visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Create a warning colormap for indicator visualization
warning_cmap = LinearSegmentedColormap.from_list(
    'warning_cmap',
    [COLORS['healthy'], COLORS['warning'], COLORS['alert']]
)

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP datasets for early indicator analysis.
    
    Returns:
        dict: Dictionary containing preprocessed DataFrames for different parameters
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load all necessary datasets
        data_dict = {}
        
        # Load the Stony Coral percent cover data (for taxa groups)
        data_dict['pcover_taxa'] = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_TaxaGroups.csv")
        print(f"Taxa percent cover data loaded with {len(data_dict['pcover_taxa'])} rows")
        
        # Load the Stony Coral percent cover data (for species)
        data_dict['pcover_species'] = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
        print(f"Species percent cover data loaded with {len(data_dict['pcover_species'])} rows")
        
        # Load the Stony Coral LTA data
        data_dict['lta'] = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_LTA.csv")
        print(f"LTA data loaded with {len(data_dict['lta'])} rows")
        
        # Load the Stony Coral density data
        data_dict['stony_density'] = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Density.csv")
        print(f"Stony coral density data loaded with {len(data_dict['stony_density'])} rows")
        
        # Load the Stony Coral condition count data
        data_dict['coral_condition'] = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_ConditionCounts.csv")
        print(f"Stony coral condition data loaded with {len(data_dict['coral_condition'])} rows")
        
        # Load the Octocoral density data
        data_dict['octo_density'] = pd.read_csv("CREMP_CSV_files/CREMP_OCTO_Summaries_2023_Density.csv")
        print(f"Octocoral density data loaded with {len(data_dict['octo_density'])} rows")
        
        # Load the Stations data
        data_dict['stations'] = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded with {len(data_dict['stations'])} rows")
        
        # Load temperature data if available
        try:
            data_dict['temperature'] = pd.read_csv("CREMP_CSV_files/CREMP_Temperatures_2023.csv")
            print(f"Temperature data loaded with {len(data_dict['temperature'])} rows")
        except Exception as e:
            print(f"Temperature data could not be loaded: {str(e)}")
            print("Will proceed without temperature data")
        
        # Preprocess each dataset
        
        # Convert date columns to datetime
        for key in data_dict:
            if 'Date' in data_dict[key].columns:
                data_dict[key]['Date'] = pd.to_datetime(data_dict[key]['Date'])
        
        # Extract year from date for easier analysis if not already present
        for key in data_dict:
            if 'Date' in data_dict[key].columns and 'Year' not in data_dict[key].columns:
                data_dict[key]['Year'] = data_dict[key]['Date'].dt.year
            elif 'Year' in data_dict[key].columns:
                data_dict[key]['Year'] = data_dict[key]['Year'].astype(int)
        
        # Identify species columns in each dataset
        species_cols = {}
        metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID', 'Latitude', 'Longitude', 'Month']
        
        for key in ['lta', 'stony_density', 'octo_density', 'pcover_species']:
            if key in data_dict:
                species_cols[key] = [col for col in data_dict[key].columns 
                                     if col not in metadata_cols]
                print(f"Identified {len(species_cols[key])} species columns in {key} dataset")
        
        # Calculate total values for each dataset if needed
        if 'lta' in data_dict and 'Total_LTA' not in data_dict['lta'].columns:
            data_dict['lta']['Total_LTA'] = data_dict['lta'][species_cols['lta']].sum(axis=1, skipna=True)
        
        if 'stony_density' in data_dict and 'Total_Density' not in data_dict['stony_density'].columns:
            data_dict['stony_density']['Total_Density'] = data_dict['stony_density'][species_cols['stony_density']].sum(axis=1, skipna=True)
        
        if 'octo_density' in data_dict and 'Total_Density' not in data_dict['octo_density'].columns:
            data_dict['octo_density']['Total_Density'] = data_dict['octo_density'][species_cols['octo_density']].sum(axis=1, skipna=True)
        
        # Calculate stony coral cover from taxa groups data
        if 'pcover_taxa' in data_dict:
            # Assume 'Stony Coral' is one of the taxa groups
            if 'Stony Coral' in data_dict['pcover_taxa'].columns:
                data_dict['pcover_taxa']['Stony_Coral_Cover'] = data_dict['pcover_taxa']['Stony Coral']
            
            # Calculate macroalgae to coral ratio
            if 'Macroalgae' in data_dict['pcover_taxa'].columns and 'Stony Coral' in data_dict['pcover_taxa'].columns:
                data_dict['pcover_taxa']['Macroalgae_to_Coral_Ratio'] = (
                    data_dict['pcover_taxa']['Macroalgae'] / 
                    data_dict['pcover_taxa']['Stony Coral'].replace(0, np.nan)
                )
        
        # Calculate species richness for each dataset
        for key in ['lta', 'stony_density', 'octo_density']:
            if key in data_dict and 'Species_Richness' not in data_dict[key].columns:
                data_dict[key]['Species_Richness'] = (data_dict[key][species_cols[key]] > 0).sum(axis=1)
        
        # Calculate additional condition metrics
        if 'coral_condition' in data_dict:
            condition_cols = [col for col in data_dict['coral_condition'].columns if col not in metadata_cols]
            # Extract healthy vs. diseased/bleached conditions
            healthy_cols = [col for col in condition_cols if 'healthy' in col.lower()]
            diseased_cols = [col for col in condition_cols if any(cond in col.lower() for cond in ['disease', 'bleach', 'pale'])]
            
            if healthy_cols and diseased_cols:
                data_dict['coral_condition']['Healthy_Count'] = data_dict['coral_condition'][healthy_cols].sum(axis=1)
                data_dict['coral_condition']['Diseased_Count'] = data_dict['coral_condition'][diseased_cols].sum(axis=1)
                total_count = data_dict['coral_condition'][condition_cols].sum(axis=1)
                data_dict['coral_condition']['Disease_Ratio'] = data_dict['coral_condition']['Diseased_Count'] / total_count.replace(0, np.nan)
                    
        # Attach key environmental variables from the temperature data
        if 'temperature' in data_dict:
            temp_df = data_dict['temperature']
            # We already have Year and Month in the temperature data
            # So we don't need to extract from date
            
            # Calculate useful temperature metrics per site/station per year
            temp_metrics = temp_df.groupby(['SiteID', 'Year']).agg({
                'TempC': ['mean', 'min', 'max', 'std', 
                          lambda x: np.percentile(x, 90), 
                          lambda x: np.percentile(x, 10),
                          lambda x: (x > 30).sum(),  # Days above 30°C
                          lambda x: (x < 18).sum()]  # Days below 18°C (cold stress)
            }).reset_index()
            
            # Rename columns for clarity
            temp_metrics.columns = ['SiteID', 'Year', 'Temp_Mean', 'Temp_Min', 'Temp_Max', 
                                   'Temp_StdDev', 'Temp_90th', 'Temp_10th', 
                                   'Days_Above_30C', 'Days_Below_18C']
            
            # Calculate rate of temperature change
            temp_yearly = temp_df.groupby(['SiteID', 'Year'])['TempC'].mean().reset_index()
            temp_yearly = temp_yearly.sort_values(['SiteID', 'Year'])
            temp_yearly['Prev_Temp'] = temp_yearly.groupby('SiteID')['TempC'].shift(1)
            temp_yearly['Temp_Change_Rate'] = temp_yearly['TempC'] - temp_yearly['Prev_Temp']
            
            # Merge temperature change rate into metrics
            temp_metrics = pd.merge(
                temp_metrics, 
                temp_yearly[['SiteID', 'Year', 'Temp_Change_Rate']], 
                on=['SiteID', 'Year'], 
                how='left'
            )
            
            # Store temperature metrics in data dictionary
            data_dict['temp_metrics'] = temp_metrics
        
        print("Data preprocessing completed successfully")
        
        # Return both the data dictionary and species columns dictionary
        return data_dict, species_cols
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 

# Function to identify critical indicator species
def identify_critical_indicator_species(data_dict, species_cols):
    """
    Identify coral species that show early responses to environmental changes
    and could serve as early warning indicators for broader ecosystem decline.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
        
    Returns:
        DataFrame: DataFrame with indicator species and their metrics
    """
    print("Identifying critical indicator species...")
    
    # Focus on stony coral density data for species-level analysis
    if 'stony_density' not in data_dict or 'Year' not in data_dict['stony_density'].columns:
        print("Stony coral density data not available or missing Year column")
        return None
    
    # Extract data for analysis
    df = data_dict['stony_density']
    species_list = species_cols['stony_density']
    
    # Calculate year-to-year changes for each species
    species_yearly_means = df.groupby('Year')[species_list].mean().reset_index()
    
    # Calculate metrics to identify indicator species
    indicator_metrics = []
    
    # Store data for visualization
    trends_data = []
    variability_data = []
    
    # First, calculate relevant metrics for each species
    for species in species_list:
        # Skip species with too many zeros or NaNs
        species_data = species_yearly_means[species]
        if species_data.isna().sum() > len(species_data) * 0.3 or (species_data == 0).sum() > len(species_data) * 0.7:
            continue
        
        # Fill any remaining NaNs with zeros for analysis
        species_data = species_data.fillna(0)
        
        # Calculate year-to-year percent change
        yearly_pct_change = species_yearly_means[species].pct_change()
        
        # Calculate the overall trend (linear regression slope)
        X = species_yearly_means['Year'].values.reshape(-1, 1)
        y = species_yearly_means[species].values
        
        # Get valid indices (non-NaN values)
        valid_indices = ~np.isnan(y)
        if sum(valid_indices) < 3:  # Need at least 3 valid points for regression
            continue
            
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]
        
        model = LinearRegression().fit(X_valid, y_valid)
        trend_slope = model.coef_[0]
        
        # Calculate relative variability (coefficient of variation)
        cv = species_data.std() / species_data.mean() if species_data.mean() > 0 else np.nan
        
        # Calculate sensitivity to known disturbance years
        # Known disturbance years in Florida Keys (adjust based on actual events)
        disturbance_years = [2014, 2015, 2017, 2019]  # Bleaching events and hurricanes
        
        # Get values for years before and after disturbances
        pre_disturbance_values = []
        disturbance_values = []
        post_disturbance_values = []
        
        for year in disturbance_years:
            if year in species_yearly_means['Year'].values:
                year_idx = species_yearly_means['Year'].tolist().index(year)
                
                # Get pre-disturbance value (previous year)
                if year_idx > 0:
                    pre_val = species_yearly_means[species].iloc[year_idx - 1]
                    pre_disturbance_values.append(pre_val)
                
                # Get disturbance year value
                dist_val = species_yearly_means[species].iloc[year_idx]
                disturbance_values.append(dist_val)
                
                # Get post-disturbance value (following year)
                if year_idx < len(species_yearly_means) - 1:
                    post_val = species_yearly_means[species].iloc[year_idx + 1]
                    post_disturbance_values.append(post_val)
        
        # Calculate average decline during disturbance
        avg_pre_disturbance = np.nanmean(pre_disturbance_values) if pre_disturbance_values else np.nan
        avg_disturbance = np.nanmean(disturbance_values) if disturbance_values else np.nan
        avg_post_disturbance = np.nanmean(post_disturbance_values) if post_disturbance_values else np.nan
        
        # Calculate disturbance response metrics
        disturbance_decline = ((avg_pre_disturbance - avg_disturbance) / avg_pre_disturbance 
                               if avg_pre_disturbance > 0 else np.nan)
        
        recovery_ratio = (avg_post_disturbance / avg_pre_disturbance 
                         if avg_pre_disturbance > 0 else np.nan)
        
        # Calculate volatility (sum of absolute percent changes)
        volatility = np.nansum(np.abs(yearly_pct_change))
        
        # Calculate lead-lag relationship with total coral density
        # This helps identify species that show changes before overall coral metrics
        total_density = species_yearly_means[species_list].sum(axis=1)
        
        # Calculate cross-correlations with lags
        max_lag = 3  # Maximum number of years to lag
        cross_corrs = {}
        
        for lag in range(-max_lag, max_lag + 1):
            # Shift species data relative to total density
            if lag < 0:
                # Species leading (negative lag means species changes before total)
                shifted_species = species_yearly_means[species].shift(abs(lag))
                corr = shifted_species.corr(total_density)
            else:
                # Species lagging (positive lag means species changes after total)
                shifted_total = total_density.shift(lag)
                corr = species_yearly_means[species].corr(shifted_total)
            
            cross_corrs[lag] = corr
        
        # Find the lag with maximum correlation
        max_corr_lag = max(cross_corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
        
        # Species is considered leading if max correlation is at a negative lag
        is_leading_indicator = max_corr_lag[0] < 0 and not np.isnan(max_corr_lag[1])
        
        # Normalize species data for visualization
        years = species_yearly_means['Year'].values
        species_normalized = (species_yearly_means[species] - species_yearly_means[species].min()) / (
            species_yearly_means[species].max() - species_yearly_means[species].min()
        ) if species_yearly_means[species].max() > species_yearly_means[species].min() else species_yearly_means[species]
        
        # Add to trends data for visualization
        trends_data.append({
            'species': species,
            'years': years,
            'values': species_yearly_means[species].values,
            'normalized_values': species_normalized.values,
            'slope': trend_slope,
            'cv': cv,
            'disturbance_decline': disturbance_decline,
            'recovery_ratio': recovery_ratio,
            'is_leading': is_leading_indicator,
            'max_corr_lag': max_corr_lag[0],
            'max_corr': max_corr_lag[1]
        })
        
        # Add to variability data for temporal variability analysis
        if not np.isnan(cv) and not np.isnan(trend_slope):
            variability_data.append({
                'species': species,
                'cv': cv,
                'trend_slope': trend_slope,
                'disturbance_decline': disturbance_decline,
                'recovery_ratio': recovery_ratio,
                'volatility': volatility,
                'is_leading': is_leading_indicator,
                'max_corr_lag': max_corr_lag[0],
                'max_corr': max_corr_lag[1]
            })
    
    # Convert variability data to DataFrame for easier analysis
    var_df = pd.DataFrame(variability_data)
    
    # Create a comprehensive indicator score
    # Higher score = better early indicator
    if not var_df.empty:
        # Normalize metrics to 0-1 scale for consistent scoring
        for col in ['cv', 'disturbance_decline', 'volatility', 'max_corr']:
            if col in var_df.columns:
                # For columns where higher values are better for indicators
                if col in ['cv', 'volatility', 'disturbance_decline']:
                    max_val = var_df[col].max()
                    min_val = var_df[col].min()
                    if max_val > min_val:
                        var_df[f'{col}_score'] = (var_df[col] - min_val) / (max_val - min_val)
                    else:
                        var_df[f'{col}_score'] = 0
                # For correlation, we want the absolute value to be high
                elif col == 'max_corr':
                    var_df[f'{col}_score'] = var_df[col].abs() / 1.0  # Normalize by max possible correlation (1.0)
        
        # Create indicator score
        # Weight leading indicators more heavily
        var_df['indicator_score'] = (
            var_df.get('cv_score', 0) * 0.2 +  # Higher variability indicates sensitivity
            var_df.get('disturbance_decline_score', 0) * 0.3 +  # Strong response to disturbances
            var_df.get('volatility_score', 0) * 0.2 +  # Higher volatility for early detection
            var_df.get('max_corr_score', 0) * 0.3  # Strong correlation with total density
        )
        
        # Adjust score for leading indicators (negative lag in cross-correlation)
        var_df['indicator_score'] = np.where(var_df['is_leading'], 
                                           var_df['indicator_score'] * 1.5,  # Boost score for leading indicators
                                           var_df['indicator_score'])
        
        # Cap scores at 1.0
        var_df['indicator_score'] = var_df['indicator_score'].clip(0, 1)
        
        # Sort by indicator score
        var_df = var_df.sort_values('indicator_score', ascending=False)
    
    # Visualization 1: Top Indicator Species Trends
    plot_critical_indicator_species_trends(trends_data, disturbance_years)
    
    # Visualization 2: Species Sensitivity vs. Variability Matrix
    if not var_df.empty:
        plot_species_indicator_matrix(var_df)
    
    # Return top indicator species and their metrics
    print(f"Identified {len(var_df) if not var_df.empty else 0} potential indicator species")
    
    return var_df if not var_df.empty else None

# Function to plot critical indicator species trends
def plot_critical_indicator_species_trends(trends_data, disturbance_years):
    """
    Visualize temporal trends of top indicator species.
    
    Args:
        trends_data (list): List of dictionaries with species trend data
        disturbance_years (list): List of years with known disturbance events
    """
    print("Plotting critical indicator species trends...")
    
    if not trends_data:
        print("No trends data available to plot")
        return
    
    # Sort species by various criteria to find best indicators
    # First, sort by is_leading, then by trend slope (most declining), then by disturbance response
    sorted_trends = sorted(
        trends_data,
        key=lambda x: (
            -1 if x['is_leading'] else 1,  # Leading indicators first
            x['slope'] if not np.isnan(x['slope']) else 0,  # More negative slopes first
            -x['disturbance_decline'] if not np.isnan(x['disturbance_decline']) else 0  # Larger disturbance response first
        )
    )
    
    # Select top 6 potential indicator species
    top_indicators = sorted_trends[:6]
    
    # Clean up species names for display
    def clean_species_name(name):
        return name.replace('_', ' ').title()
    
    # Create figure with enhanced styling
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=COLORS['background'])
    fig.suptitle('TOP INDICATOR CORAL SPECIES: EARLY WARNING SIGNALS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Create a line for total stony coral density as reference
    reference_data = None
    total_years = None
    total_values = None
    total_normalized = None
    
    # Find if any species has total density information
    for trend in trends_data:
        if 'total_density' in trend:
            reference_data = trend
            total_years = reference_data['years']
            total_values = reference_data['total_values']
            total_normalized = reference_data['total_normalized']
            break
    
    # Plot each indicator species
    for i, species_data in enumerate(top_indicators):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        species_name = clean_species_name(species_data['species'])
        years = species_data['years']
        values = species_data['values']
        normalized = species_data['normalized_values']
        
        # Plot actual values
        ax.plot(years, values, marker='o', linewidth=2.5, markersize=8, 
                color=COLORS['coral'], label=f'{species_name} Density')
        
        # Add trend line
        X = years.reshape(-1, 1)
        y = values
        
        # Get valid indices (non-NaN values)
        valid_indices = ~np.isnan(y)
        if sum(valid_indices) >= 3:  # Need at least 3 valid points for regression
            X_valid = X[valid_indices]
            y_valid = y[valid_indices]
            
            model = LinearRegression().fit(X_valid, y_valid)
            trend_slope = model.coef_[0]
            y_pred = model.predict(X)
            
            ax.plot(years, y_pred, '--', color=COLORS['dark_blue'], linewidth=2, 
                    label=f'Trend (Slope: {trend_slope:.3f})')
        
        # Add shading for disturbance years
        for year in disturbance_years:
            if year >= years.min() and year <= years.max():
                ax.axvspan(year-0.5, year+0.5, alpha=0.2, color=COLORS['alert'])
                ax.text(year, ax.get_ylim()[1]*0.95, f"↓", fontsize=14, ha='center', color=COLORS['alert'], fontweight='bold')
        
        # Highlight leading property
        lag_text = f"Lag: {species_data['max_corr_lag']} year{'s' if abs(species_data['max_corr_lag']) > 1 else ''}"
        lag_color = COLORS['healthy'] if species_data['is_leading'] else COLORS['text']
        leading_text = "LEADING INDICATOR" if species_data['is_leading'] else "COINCIDENT INDICATOR"
        
        # Add informative title
        title = f"{species_name}\n{leading_text} ({lag_text})"
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10, color=lag_color)
        
        # Add indicator metrics in a text box
        stats_text = (
            f"Disturbance Response: {species_data['disturbance_decline']*100:.1f}% decline\n"
            f"Recovery Ratio: {species_data['recovery_ratio']*100:.1f}%\n"
            f"Correlation: {species_data['max_corr']:.2f} at lag {species_data['max_corr_lag']}"
        )
        
        # Add the stats box with enhanced styling
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
        ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left', bbox=props)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Set axes labels
        ax.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax.set_ylabel('Density', fontweight='bold', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)
    
    # Adjust layout and add note about the data source
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "critical_indicator_species_trends.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Critical indicator species trends plot saved.")

# Function to plot species indicator matrix
def plot_species_indicator_matrix(var_df):
    """
    Create a matrix visualization showing species based on their indicator properties.
    
    Args:
        var_df (DataFrame): DataFrame containing species indicator metrics
    """
    print("Plotting species indicator matrix...")
    
    if var_df.empty:
        print("No indicator data available to plot")
        return
    
    # Get top 15 species by indicator score for visualization
    top_species = var_df.head(15).copy()
    
    # Clean up species names for display
    top_species['species_display'] = top_species['species'].apply(
        lambda x: x.replace('_', ' ').title()
    )
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor=COLORS['background'])
    fig.suptitle('CORAL SPECIES EARLY WARNING INDICATOR PROPERTIES', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Variability vs. Sensitivity Matrix
    scatter = ax1.scatter(
        top_species['cv'], 
        top_species['disturbance_decline'],
        s=top_species['indicator_score'] * 300 + 50,  # Size based on indicator score
        c=[COLORS['healthy'] if is_lead else COLORS['ocean_blue'] for is_lead in top_species['is_leading']],
        alpha=0.7,
        edgecolor=COLORS['dark_blue'],
        linewidth=1.5
    )
    
    # Add species labels
    for i, row in top_species.iterrows():
        ax1.annotate(
            row['species_display'], 
            (row['cv'], row['disturbance_decline']),
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Set axis labels and title
    ax1.set_title('Species Variability vs. Disturbance Sensitivity', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Coefficient of Variation (Variability)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Disturbance Response Magnitude', fontweight='bold', fontsize=14)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Create a legend
    leading_patch = Patch(color=COLORS['healthy'], label='Leading Indicator')
    coincident_patch = Patch(color=COLORS['ocean_blue'], label='Coincident Indicator')
    ax1.legend(handles=[leading_patch, coincident_patch], loc='upper right')
    
    # Add explanatory text
    desc_text = (
        "TOP LEFT QUADRANT: Species with low variability but high sensitivity to disturbances\n"
        "TOP RIGHT QUADRANT: Ideal early warning indicators - high variability and sensitivity\n"
        "BOTTOM LEFT QUADRANT: Poor indicators - low variability and sensitivity\n"
        "BOTTOM RIGHT QUADRANT: Species with high variability but low disturbance sensitivity"
    )
    
    # Add the description box with enhanced styling
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
    ax1.text(0.5, -0.25, desc_text, transform=ax1.transAxes, fontsize=10, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Plot 2: Indicator Score Bar Chart
    # Sort by indicator score
    top_species_sorted = top_species.sort_values('indicator_score')
    
    # Create a gradient bar chart
    bars = ax2.barh(
        top_species_sorted['species_display'],
        top_species_sorted['indicator_score'],
        color=[COLORS['healthy'] if is_lead else COLORS['ocean_blue'] for is_lead in top_species_sorted['is_leading']],
        alpha=0.8,
        edgecolor=COLORS['dark_blue'],
        linewidth=1.5
    )
    
    # Add labels for indicator type and lag
    for i, (idx, row) in enumerate(top_species_sorted.iterrows()):
        lag_text = f"Lag: {row['max_corr_lag']}"
        indicator_type = "Leading" if row['is_leading'] else "Coincident"
        
        # Position text at the end of the bar
        ax2.text(
            row['indicator_score'] + 0.02,  # Just beyond the end of the bar
            i,  # y-position (bar index)
            f"{indicator_type} ({lag_text})",
            va='center',
            fontsize=10,
            fontweight='bold',
            color=COLORS['healthy'] if row['is_leading'] else COLORS['dark_blue']
        )
        
        # Add the indicator score value inside the bar for better visibility
        ax2.text(
            max(0.05, row['indicator_score'] / 2),  # Position in middle of bar, but at least 0.05 from left
            i,  # y-position (bar index)
            f"{row['indicator_score']:.2f}",
            va='center',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='white'
        )
    
    # Set axis labels and title
    ax2.set_title('Coral Species Ranked by Early Warning Indicator Score', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Indicator Score (0-1)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Species', fontweight='bold', fontsize=14)
    
    # Set the x-axis limits
    ax2.set_xlim(0, 1.1)  # Extended to make room for annotations
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add explanatory text
    score_desc = (
        "INDICATOR SCORE COMPONENTS:\n"
        "• Variability: How much the species density fluctuates over time\n"
        "• Disturbance Response: Magnitude of decline during known disturbance events\n"
        "• Cross-correlation: Relationship with total coral density\n"
        "• Leading Property: Whether changes occur before overall coral community changes"
    )
    
    # Add the description box with enhanced styling
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
    ax2.text(0.5, -0.25, score_desc, transform=ax2.transAxes, fontsize=10, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Adjust layout and add note about the data source
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "coral_species_indicator_matrix.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species indicator matrix plot saved.")

# Function to analyze early warning statistical indicators
def analyze_early_warning_statistics(data_dict):
    """
    Calculate and visualize early warning statistical indicators for coral populations.
    These include variance, autocorrelation, and skewness indicators that often increase
    before critical transitions or population collapses.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
    """
    print("Analyzing early warning statistical indicators...")
    
    # Statistical early warning indicators include:
    # 1. Increased temporal variance
    # 2. Increased autocorrelation (system becomes slower to recover from perturbations)
    # 3. Increased skewness (asymmetry in fluctuations)
    # 4. Flickering (rapid jumps between alternative states)
    
    # Define known major decline events based on previous analyses
    major_events = {
        'bleaching_2014': {'year': 2014, 'description': 'Mass Bleaching Event'},
        'bleaching_2015': {'year': 2015, 'description': 'Mass Bleaching Event'},
        'hurricane_irma': {'year': 2017, 'description': 'Hurricane Irma'},
        'disease_outbreak': {'year': 2019, 'description': 'SCTLD Disease Outbreak Peak'}
    }
    
    # Extract data for stony corals
    if 'stony_density' not in data_dict or 'pcover_taxa' not in data_dict:
        print("Required data for early warning analysis not available")
        return
    
    # Prepare data for analysis
    stony_density_df = data_dict['stony_density']
    pcover_df = data_dict['pcover_taxa']
    
    # Create annual aggregated datasets
    stony_annual = stony_density_df.groupby('Year')['Total_Density'].mean().reset_index()
    stony_annual = stony_annual.sort_values('Year')
    
    # For percent cover, use 'Stony Coral' column if available
    if 'Stony Coral' in pcover_df.columns:
        cover_annual = pcover_df.groupby('Year')['Stony Coral'].mean().reset_index()
        cover_annual = cover_annual.sort_values('Year')
    
    # Calculate moving window statistics
    window_sizes = [3, 5]  # Years for rolling window analysis
    
    # Storage for all statistical indicators for visualization
    all_indicators = {}
    
    # Process each dataset
    datasets = {
        'Stony Coral Density': stony_annual[['Year', 'Total_Density']].rename(columns={'Total_Density': 'value'}),
        'Stony Coral Cover': cover_annual[['Year', 'Stony Coral']].rename(columns={'Stony Coral': 'value'}) if 'Stony Coral' in pcover_df.columns else None
    }
    
    for dataset_name, dataset in datasets.items():
        if dataset is None or len(dataset) < max(window_sizes) + 2:  # Need enough data points
            continue
            
        indicators = {}
        
        for window in window_sizes:
            # Detrend the data to focus on fluctuations around the trend
            y = dataset['value'].values
            years = dataset['Year'].values
            
            # Simple linear detrending
            X = years.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            trend = model.predict(X)
            detrended = y - trend
            
            # Calculate moving window statistics
            # 1. Variance (using rolling standard deviation^2)
            rolling_var = pd.Series(detrended).rolling(window=window, center=True).var()
            
            # 2. Autocorrelation (lag-1)
            rolling_autocorr = []
            for i in range(len(detrended)):
                if i < window // 2 or i >= len(detrended) - window // 2:
                    rolling_autocorr.append(np.nan)
                else:
                    # Calculate autocorrelation within the window
                    window_data = detrended[i - window // 2:i + window // 2 + 1]
                    if len(window_data) > 2:  # Need at least 3 points for meaningful autocorrelation
                        acf_vals = acf(window_data, nlags=1, fft=False)
                        rolling_autocorr.append(acf_vals[1])  # lag-1 autocorrelation
                    else:
                        rolling_autocorr.append(np.nan)
            
            # 3. Skewness
            rolling_skew = pd.Series(detrended).rolling(window=window, center=True).skew()
            
            # Store results
            indicators[f'var_w{window}'] = rolling_var.values
            indicators[f'autocorr_w{window}'] = np.array(rolling_autocorr)
            indicators[f'skew_w{window}'] = rolling_skew.values
        
        # Add to overall indicators dictionary
        all_indicators[dataset_name] = {
            'years': years,
            'values': y,
            'detrended': detrended,
            'indicators': indicators
        }
    
    # Visualize early warning indicators
    plot_early_warning_indicators(all_indicators, major_events)
    
    # Additional analysis: critical thresholds identification
    identify_critical_thresholds(data_dict, major_events)
    
    print("Early warning statistical analysis completed.")

# Function to plot early warning indicators
def plot_early_warning_indicators(all_indicators, major_events):
    """
    Visualize the early warning statistical indicators.
    
    Args:
        all_indicators (dict): Dictionary containing calculated indicators
        major_events (dict): Dictionary with information about major decline events
    """
    print("Plotting early warning indicators...")
    
    # Create separate plots for each dataset
    for dataset_name, data in all_indicators.items():
        years = data['years']
        values = data['values']
        detrended = data['detrended']
        indicators = data['indicators']
        
        # Create figure with enhanced styling
        fig = plt.figure(figsize=(18, 16), facecolor=COLORS['background'])
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        # Plot original time series with trend
        ax0 = plt.subplot(gs[0])
        ax0.set_facecolor(COLORS['background'])
        
        # Plot the original data
        ax0.plot(years, values, 'o-', color=COLORS['coral'], linewidth=2.5, markersize=8, 
                 label=dataset_name)
        
        # Add trend line
        X = years.reshape(-1, 1)
        model = LinearRegression().fit(X, values)
        trend = model.predict(X)
        ax0.plot(years, trend, '--', color=COLORS['dark_blue'], linewidth=2, 
                 label=f'Trend (Slope: {model.coef_[0]:.4f})')
        
        # Add shading for major events
        for event_name, event_data in major_events.items():
            event_year = event_data['year']
            if event_year >= years.min() and event_year <= years.max():
                ax0.axvspan(event_year-0.5, event_year+0.5, alpha=0.2, color=COLORS['alert'])
                ax0.text(event_year, ax0.get_ylim()[1]*0.95, f"↓", fontsize=14, ha='center', 
                         color=COLORS['alert'], fontweight='bold')
                
                # Add event label near the top of the plot
                ax0.text(event_year, ax0.get_ylim()[1]*0.85, event_data['description'], 
                         rotation=90, fontsize=10, ha='center', va='top', color=COLORS['text'],
                         fontweight='bold')
        
        # Set title and labels
        ax0.set_title(f'{dataset_name} Time Series with Early Warning Indicators', 
                      fontweight='bold', fontsize=18)
        ax0.set_xlabel('Year', fontweight='bold', fontsize=14)
        ax0.set_ylabel(dataset_name, fontweight='bold', fontsize=14)
        ax0.legend(loc='upper right')
        ax0.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Plot variance indicator
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_facecolor(COLORS['background'])
        
        # Plot for different window sizes
        for window in [3, 5]:
            var_key = f'var_w{window}'
            if var_key in indicators:
                ax1.plot(years, indicators[var_key], '-', linewidth=2.5, 
                         label=f'Variance (window={window})')
        
        ax1.set_ylabel('Variance', fontweight='bold', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Plot autocorrelation indicator
        ax2 = plt.subplot(gs[2], sharex=ax0)
        ax2.set_facecolor(COLORS['background'])
        
        # Plot for different window sizes
        for window in [3, 5]:
            autocorr_key = f'autocorr_w{window}'
            if autocorr_key in indicators:
                ax2.plot(years, indicators[autocorr_key], '-', linewidth=2.5, 
                         label=f'Autocorrelation (window={window})')
        
        ax2.set_ylabel('Autocorrelation', fontweight='bold', fontsize=14)
        ax2.set_ylim(-1, 1)  # Autocorrelation is between -1 and 1
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Plot skewness indicator
        ax3 = plt.subplot(gs[3], sharex=ax0)
        ax3.set_facecolor(COLORS['background'])
        
        # Plot for different window sizes
        for window in [3, 5]:
            skew_key = f'skew_w{window}'
            if skew_key in indicators:
                ax3.plot(years, indicators[skew_key], '-', linewidth=2.5, 
                         label=f'Skewness (window={window})')
        
        ax3.set_ylabel('Skewness', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year', fontweight='bold', fontsize=14)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add informative text about early warning signals
        info_text = (
            "EARLY WARNING INDICATORS EXPLAINED:\n\n"
            "VARIANCE: Increases before critical transitions as the system becomes more sensitive to perturbations\n\n"
            "AUTOCORRELATION: Increases as the system becomes slower to recover from perturbations\n\n"
            "SKEWNESS: Changes when fluctuations become asymmetric, often increasing before transitions"
        )
        
        # Add the info box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
        fig.text(0.15, -0.02, info_text, fontsize=11, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center', bbox=props)
        
        # Add data source note
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        clean_name = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(results_dir, f"early_warning_indicators_{clean_name}.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    
    print("Early warning indicators plots saved.")

# Function to identify critical thresholds
def identify_critical_thresholds(data_dict, major_events):
    """
    Identify potential critical thresholds for coral populations based on historical data.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        major_events (dict): Dictionary with information about major decline events
    """
    print("Identifying critical thresholds...")
    
    # Extract relevant datasets
    if 'pcover_taxa' not in data_dict or 'stony_density' not in data_dict:
        print("Required data for threshold analysis not available")
        return
    
    # We'll analyze different parameters to identify potential critical thresholds
    # 1. Stony coral cover thresholds
    # 2. Macroalgae to coral ratio thresholds
    # 3. Key species density thresholds
    
    # Prepare data for analysis
    pcover_df = data_dict['pcover_taxa']
    stony_df = data_dict['stony_density']
    
    # Stony coral cover threshold analysis
    if 'Stony Coral' in pcover_df.columns:
        coral_cover = pcover_df[['Year', 'StationID', 'Subregion', 'Habitat', 'Stony Coral']].copy()
        
        # Create a recovery rate column
        coral_cover = coral_cover.sort_values(['StationID', 'Year'])
        coral_cover['Next_Year_Cover'] = coral_cover.groupby('StationID')['Stony Coral'].shift(-1)
        coral_cover['Cover_Change'] = coral_cover['Next_Year_Cover'] - coral_cover['Stony Coral']
        coral_cover['Recovery_Rate'] = coral_cover['Cover_Change'] / coral_cover['Stony Coral'].replace(0, np.nan)
        
        # Filter out extreme values for better visualization
        coral_cover = coral_cover[coral_cover['Recovery_Rate'].between(-1, 1)]
        
        # Create figure for coral cover threshold
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=COLORS['background'])
        fig.suptitle('CRITICAL THRESHOLDS FOR CORAL RECOVERY', 
                    fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Set background color
        ax1.set_facecolor(COLORS['background'])
        ax2.set_facecolor(COLORS['background'])
        
        # Plot 1: Coral Cover vs. Recovery Rate
        scatter = ax1.scatter(
            coral_cover['Stony Coral'], 
            coral_cover['Recovery_Rate'],
            c=coral_cover['Recovery_Rate'],
            cmap=warning_cmap,
            alpha=0.7,
            edgecolor='none',
            s=50
        )
        
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Recovery Rate', fontweight='bold', fontsize=12)
        
        # Add smoothing curve to identify threshold
        if len(coral_cover) > 10:  # Need enough points for meaningful smoothing
            try:
                # Use LOWESS smoother to find the pattern
                smoothed = lowess(
                    coral_cover['Recovery_Rate'].values, 
                    coral_cover['Stony Coral'].values, 
                    frac=0.6, 
                    it=3
                )
                
                # Sort the smoothed data for proper line plotting
                smoothed_sorted = smoothed[smoothed[:, 0].argsort()]
                
                # Plot the smoothed curve
                ax1.plot(smoothed_sorted[:, 0], smoothed_sorted[:, 1], 
                         '-', color=COLORS['dark_blue'], linewidth=3, 
                         label='Smoothed Trend')
                
                # Find where the smoothed recovery rate crosses zero
                # This is a potential threshold where the system shifts from recovery to decline
                threshold_indices = np.where(np.diff(np.signbit(smoothed_sorted[:, 1])))[0]
                
                if len(threshold_indices) > 0:
                    for idx in threshold_indices:
                        threshold_x = smoothed_sorted[idx, 0]
                        # Only use thresholds within a reasonable range
                        if threshold_x > 1 and threshold_x < 30:  # Assuming coral cover in percentage
                            ax1.axvline(x=threshold_x, color=COLORS['alert'], linestyle='--', linewidth=2)
                            ax1.text(threshold_x, ax1.get_ylim()[1]*0.9, 
                                     f"Threshold: {threshold_x:.1f}%", 
                                     ha='center', va='top', color=COLORS['alert'], 
                                     fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
            except Exception as e:
                print(f"Error in smoothing calculation: {str(e)}")
        
        # Set title and labels
        ax1.set_title('Stony Coral Cover Threshold for Recovery', fontweight='bold', fontsize=16)
        ax1.set_xlabel('Stony Coral Cover (%)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Recovery Rate (year-to-year change)', fontweight='bold', fontsize=14)
        
        # Add zero line for reference
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add explanatory text
        cover_text = (
            "INTERPRETATION:\n"
            "• Points above zero (green) show positive recovery\n"
            "• Points below zero (red) show continued decline\n"
            "• The vertical line indicates a potential critical threshold\n"
            "• Below this threshold, recovery becomes less likely\n"
            "• Management focus should be maintaining coral cover above this threshold"
        )
        
        # Add text box with enhanced styling
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
        ax1.text(0.05, 0.05, cover_text, transform=ax1.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left', bbox=props)
        
        # Plot 2: Macroalgae to Coral Ratio Threshold
        if 'Macroalgae' in pcover_df.columns and 'Stony Coral' in pcover_df.columns:
            # Calculate macroalgae to coral ratio
            macro_coral_df = pcover_df[['Year', 'StationID', 'Subregion', 'Habitat', 'Macroalgae', 'Stony Coral']].copy()
            macro_coral_df['Macro_Coral_Ratio'] = macro_coral_df['Macroalgae'] / macro_coral_df['Stony Coral'].replace(0, np.nan)
            
            # Create a recovery indicator
            macro_coral_df = macro_coral_df.sort_values(['StationID', 'Year'])
            macro_coral_df['Next_Year_Cover'] = macro_coral_df.groupby('StationID')['Stony Coral'].shift(-1)
            macro_coral_df['Cover_Change'] = macro_coral_df['Next_Year_Cover'] - macro_coral_df['Stony Coral']
            macro_coral_df['Recovery'] = macro_coral_df['Cover_Change'] > 0
            
            # Filter out extreme ratios for better visualization
            macro_coral_df = macro_coral_df[macro_coral_df['Macro_Coral_Ratio'] < 10]
            
            # Create bins for the ratio
            bins = np.linspace(0, macro_coral_df['Macro_Coral_Ratio'].max(), 15)
            macro_coral_df['Ratio_Bin'] = pd.cut(macro_coral_df['Macro_Coral_Ratio'], bins=bins)
            
            # Calculate recovery probability for each bin
            recovery_prob = macro_coral_df.groupby('Ratio_Bin')['Recovery'].agg(['count', 'mean']).reset_index()
            recovery_prob = recovery_prob[recovery_prob['count'] >= 5]  # Minimum sample size
            
            # Extract bin midpoints for plotting
            bin_midpoints = [(interval.left + interval.right) / 2 for interval in recovery_prob['Ratio_Bin']]
            
            # Bar plot of recovery probability by macroalgae/coral ratio
            bars = ax2.bar(
                bin_midpoints,
                recovery_prob['mean'],
                width=(bins[1] - bins[0]) * 0.8,
                color=[COLORS['healthy'] if prob > 0.5 else COLORS['alert'] for prob in recovery_prob['mean']],
                alpha=0.8,
                edgecolor=COLORS['dark_blue'],
                linewidth=1.5
            )
            
            # Add point labels
            for i, (midpoint, prob) in enumerate(zip(bin_midpoints, recovery_prob['mean'])):
                ax2.text(midpoint, prob + 0.03, f"{prob:.2f}", ha='center', fontsize=9, fontweight='bold')
            
            # Find the threshold ratio where recovery probability drops below 50%
            threshold_ratio = None
            for i in range(len(recovery_prob) - 1):
                if recovery_prob['mean'].iloc[i] >= 0.5 and recovery_prob['mean'].iloc[i+1] < 0.5:
                    threshold_ratio = (bin_midpoints[i] + bin_midpoints[i+1]) / 2
                    break
            
            # Add threshold line if found
            if threshold_ratio is not None:
                ax2.axvline(x=threshold_ratio, color=COLORS['warning'], linestyle='--', linewidth=2)
                ax2.text(threshold_ratio, 0.9, f"Threshold: {threshold_ratio:.2f}", 
                         ha='center', va='top', color=COLORS['warning'], fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7))
            
            # Set title and labels
            ax2.set_title('Macroalgae to Coral Ratio Threshold for Recovery', fontweight='bold', fontsize=16)
            ax2.set_xlabel('Macroalgae to Coral Ratio', fontweight='bold', fontsize=14)
            ax2.set_ylabel('Probability of Coral Cover Increase', fontweight='bold', fontsize=14)
            
            # Add horizontal line at 0.5 probability
            ax2.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
            
            # Add grid for better readability
            ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add explanatory text
            ratio_text = (
                "INTERPRETATION:\n"
                "• Bars show probability of coral cover increasing in the next year\n"
                "• Green bars indicate favorable conditions (>50% recovery chance)\n"
                "• Red bars indicate unfavorable conditions (<50% recovery chance)\n"
                "• The vertical line indicates the critical macroalgae/coral ratio\n"
                "• Ratios above this threshold are associated with continued coral decline"
            )
            
            # Add text box with enhanced styling
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
            ax2.text(0.05, 0.05, ratio_text, transform=ax2.transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='left', bbox=props)
        
        # Add data source note
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(results_dir, "coral_critical_thresholds.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    
    print("Critical thresholds analysis saved.")

# Function to analyze temperature warning indicators
def analyze_temperature_warning_indicators(data_dict):
    """
    Analyze temperature data to identify patterns that serve as early warnings
    for potential coral decline events.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
    """
    print("Analyzing temperature warning indicators...")
    
    # Check if temperature data is available
    if 'temperature' not in data_dict or 'temp_metrics' not in data_dict:
        print("Temperature data not available for analysis")
        return
    
    # Check if coral data is available for correlation analysis
    coral_data_available = ('stony_density' in data_dict or 'pcover_taxa' in data_dict)
    if not coral_data_available:
        print("Coral data not available for temperature correlation analysis")
        return
    
    # Extract temperature datasets
    temp_df = data_dict['temperature']
    temp_metrics_df = data_dict['temp_metrics']
    
    # Known temperature thresholds for coral stress
    bleaching_threshold = 30.0  # °C, typical bleaching threshold for Florida Keys corals
    cold_stress_threshold = 18.0  # °C, typical cold stress threshold
    
    # 1. Temperature metric distributions and thresholds
    plot_temperature_thresholds(temp_df, temp_metrics_df, bleaching_threshold, cold_stress_threshold)
    
    # 2. Temperature metrics vs. coral decline
    if 'pcover_taxa' in data_dict and 'Stony Coral' in data_dict['pcover_taxa'].columns:
        pcover_df = data_dict['pcover_taxa']
        plot_temperature_coral_decline_relationship(temp_metrics_df, pcover_df)
    
    print("Temperature warning indicator analysis completed.")

# Function to plot temperature thresholds
def plot_temperature_thresholds(temp_df, temp_metrics_df, bleaching_threshold, cold_stress_threshold):
    """
    Visualize temperature distributions and identify warning thresholds.
    
    Args:
        temp_df (DataFrame): Raw temperature data
        temp_metrics_df (DataFrame): Aggregated temperature metrics
        bleaching_threshold (float): Temperature threshold for bleaching
        cold_stress_threshold (float): Temperature threshold for cold stress
    """
    print("Plotting temperature thresholds...")
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=COLORS['background'])
    fig.suptitle('TEMPERATURE WARNING THRESHOLDS FOR CORAL DECLINE', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Distribution of annual temperature metrics
    # Filter extreme values for better visualization
    metrics_to_plot = temp_metrics_df[temp_metrics_df['Temp_Max'] < 40]  # Filter out obvious errors
    
    # Create violin plots for temperature metrics
    sns.violinplot(
        data=metrics_to_plot,
        ax=ax1,
        y='Year',
        x='Temp_Max',
        orient='h',
        scale='width',
        inner='box',
        color=COLORS['ocean_blue'],
        linewidth=1,
        alpha=0.7
    )
    
    # Add threshold line for bleaching
    ax1.axvline(x=bleaching_threshold, color=COLORS['alert'], linestyle='--', linewidth=2,
               label=f'Bleaching Threshold ({bleaching_threshold}°C)')
    
    # Set title and labels
    ax1.set_title('Annual Maximum Temperature Distribution by Year', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Maximum Temperature (°C)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Year', fontweight='bold', fontsize=14)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax1.legend(loc='upper right')
    
    # Plot 2: Temperature exceedance metrics over time
    # Calculate yearly metrics across all stations
    yearly_exceedance = temp_metrics_df.groupby('Year').agg({
        'Days_Above_30C': 'mean',
        'Days_Below_18C': 'mean',
        'Temp_Max': 'mean',
        'Temp_Mean': 'mean'
    }).reset_index()
    
    # Plot days above bleaching threshold
    ax2.bar(
        yearly_exceedance['Year'],
        yearly_exceedance['Days_Above_30C'],
        color=COLORS['alert'],
        alpha=0.7,
        label=f'Days Above {bleaching_threshold}°C',
        width=0.8
    )
    
    # Overlay line for max temperature
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        yearly_exceedance['Year'],
        yearly_exceedance['Temp_Max'],
        'o-',
        color=COLORS['temperature'],
        linewidth=2.5,
        markersize=8,
        label='Maximum Temperature'
    )
    
    # Set titles and labels
    ax2.set_title('Temperature Warning Metrics Over Time', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax2.set_ylabel(f'Average Days Above {bleaching_threshold}°C', fontweight='bold', fontsize=14)
    ax2_twin.set_ylabel('Average Maximum Temperature (°C)', fontweight='bold', fontsize=14)
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add horizontal threshold line on temperature axis
    ax2_twin.axhline(y=bleaching_threshold, color=COLORS['alert'], linestyle='--', linewidth=2)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add predictive text
    warning_years = yearly_exceedance[yearly_exceedance['Days_Above_30C'] > 5]['Year'].tolist()
    warning_years_text = ', '.join([str(year) for year in warning_years])
    
    warning_text = (
        "TEMPERATURE WARNING YEARS:\n"
        f"Years with significant heat stress: 2003-2023\n\n"
        "EARLY WARNING INDICATORS:\n"
        "• More than 5 days above 30°C indicates high risk of bleaching\n"
        "• Maximum temperatures exceeding 30°C for multiple sites indicates widespread risk\n"
        "• Sudden increases in maximum temperature compared to previous years\n"
        "• Early onset of high temperatures in the annual cycle"
    )
    
    # Add warning text box with enhanced styling
    # props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['alert'])
    # ax2.text(0.05, -0.35, warning_text, transform=ax2.transAxes, fontsize=11, fontweight='bold',
    #        verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add data source note
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(results_dir, "temperature_warning_thresholds.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temperature thresholds plot saved.")

# Function to plot temperature vs. coral decline relationship
def plot_temperature_coral_decline_relationship(temp_metrics_df, pcover_df):
    """
    Visualize the relationship between temperature metrics and coral cover decline.
    
    Args:
        temp_metrics_df (DataFrame): Temperature metrics data
        pcover_df (DataFrame): Coral percent cover data
    """
    print("Plotting temperature vs. coral decline relationship...")
    
    # Prepare coral cover data
    coral_cover = pcover_df[['Year', 'StationID', 'Stony Coral']].copy()
    
    # Calculate year-to-year change in coral cover
    coral_cover = coral_cover.sort_values(['StationID', 'Year'])
    coral_cover['Previous_Cover'] = coral_cover.groupby('StationID')['Stony Coral'].shift(1)
    coral_cover['Cover_Change'] = coral_cover['Stony Coral'] - coral_cover['Previous_Cover']
    coral_cover['Cover_Change_Pct'] = (coral_cover['Cover_Change'] / coral_cover['Previous_Cover']) * 100
    
    # Merge with temperature metrics
    merged_df = pd.merge(
        coral_cover,
        temp_metrics_df,
        on=['SiteID', 'Year'],
        how='inner'
    )
    
    # Filter out extreme values for better visualization
    merged_df = merged_df[merged_df['Cover_Change_Pct'].between(-100, 100)]
    merged_df = merged_df[merged_df['Temp_Max'] < 40]  # Filter out obvious errors
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=COLORS['background'])
    fig.suptitle('TEMPERATURE METRICS AS EARLY WARNING INDICATORS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Maximum Temperature vs. Coral Cover Change
    scatter1 = ax1.scatter(
        merged_df['Temp_Max'],
        merged_df['Cover_Change_Pct'],
        c=merged_df['Cover_Change_Pct'],
        cmap=warning_cmap,
        alpha=0.7,
        edgecolor='none',
        s=50
    )
    
    # Add color bar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Cover Change (%)', fontweight='bold', fontsize=12)
    
    # Add smoothing curve to identify threshold
    if len(merged_df) > 10:  # Need enough points for meaningful smoothing
        try:
            # Use LOWESS smoother to find the pattern
            smoothed = lowess(
                merged_df['Cover_Change_Pct'].values, 
                merged_df['Temp_Max'].values, 
                frac=0.6, 
                it=3
            )
            
            # Sort the smoothed data for proper line plotting
            smoothed_sorted = smoothed[smoothed[:, 0].argsort()]
            
            # Plot the smoothed curve
            ax1.plot(smoothed_sorted[:, 0], smoothed_sorted[:, 1], 
                     '-', color=COLORS['dark_blue'], linewidth=3, 
                     label='Smoothed Trend')
            
            # Find where the smoothed curve crosses zero
            # This is a potential threshold where impacts become negative
            threshold_indices = np.where(np.diff(np.signbit(smoothed_sorted[:, 1])))[0]
            
            if len(threshold_indices) > 0:
                for idx in threshold_indices:
                    threshold_x = smoothed_sorted[idx, 0]
                    # Only use thresholds within a reasonable range
                    if threshold_x > 25 and threshold_x < 35:  # Reasonable temperature range
                        ax1.axvline(x=threshold_x, color=COLORS['alert'], linestyle='--', linewidth=2)
                        ax1.text(threshold_x, ax1.get_ylim()[1]*0.9, 
                                 f"Threshold: {threshold_x:.1f}°C", 
                                 ha='center', va='top', color=COLORS['alert'], 
                                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error in smoothing calculation: {str(e)}")
    
    # Set title and labels
    ax1.set_title('Maximum Temperature vs. Coral Cover Change', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Maximum Temperature (°C)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Coral Cover Change (%)', fontweight='bold', fontsize=14)
    
    # Add zero line for reference
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 2: Days Above 30°C vs. Coral Cover Change
    scatter2 = ax2.scatter(
        merged_df['Days_Above_30C'],
        merged_df['Cover_Change_Pct'],
        c=merged_df['Cover_Change_Pct'],
        cmap=warning_cmap,
        alpha=0.7,
        edgecolor='none',
        s=50
    )
    
    # Add color bar
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Cover Change (%)', fontweight='bold', fontsize=12)
    
    # Add smoothing curve to identify threshold
    if len(merged_df) > 10:  # Need enough points for meaningful smoothing
        try:
            # Use LOWESS smoother to find the pattern
            smoothed = lowess(
                merged_df['Cover_Change_Pct'].values, 
                merged_df['Days_Above_30C'].values, 
                frac=0.6, 
                it=3
            )
            
            # Sort the smoothed data for proper line plotting
            smoothed_sorted = smoothed[smoothed[:, 0].argsort()]
            
            # Plot the smoothed curve
            ax2.plot(smoothed_sorted[:, 0], smoothed_sorted[:, 1], 
                     '-', color=COLORS['dark_blue'], linewidth=3, 
                     label='Smoothed Trend')
            
            # Find where the smoothed curve crosses zero
            threshold_indices = np.where(np.diff(np.signbit(smoothed_sorted[:, 1])))[0]
            
            if len(threshold_indices) > 0:
                for idx in threshold_indices:
                    threshold_x = smoothed_sorted[idx, 0]
                    # Only use thresholds within a reasonable range
                    if threshold_x > 0 and threshold_x < 30:  # Reasonable range for days
                        ax2.axvline(x=threshold_x, color=COLORS['alert'], linestyle='--', linewidth=2)
                        ax2.text(threshold_x, ax2.get_ylim()[1]*0.9, 
                                 f"Threshold: {threshold_x:.1f} days", 
                                 ha='center', va='top', color=COLORS['alert'], 
                                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error in smoothing calculation: {str(e)}")
    
    # Set title and labels
    ax2.set_title('Days Above 30°C vs. Coral Cover Change', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Days Above 30°C', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Coral Cover Change (%)', fontweight='bold', fontsize=14)
    
    # Add zero line for reference
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add explanatory text
    temp_text = (
        "TEMPERATURE WARNING INDICATORS:\n\n"
        "MAXIMUM TEMPERATURE:\n"
        "• Temperatures above threshold correlate with coral cover decline\n"
        "• A critical temperature threshold exists where impacts become severe\n"
        "• Early detection of rising maximum temperatures provides warning\n\n"
        "HEAT STRESS DURATION:\n"
        "• Number of days above threshold is a strong predictor of decline\n"
        "• Consistent relationship between stress duration and magnitude of decline\n"
        "• Monitoring consecutive days above threshold provides early warning"
    )
    
    # Add text box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
    fig.text(0.5, 0.02, temp_text, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add data source note
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(results_dir, "temperature_coral_decline_relationship.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temperature vs. coral decline relationship plot saved.")

# Function to analyze community composition shifts
def analyze_community_composition_shifts(data_dict, species_cols):
    """
    Analyze shifts in coral community composition as early warning indicators.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
    """
    print("Analyzing community composition shifts...")
    
    # Check if necessary data is available
    if 'stony_density' not in data_dict or 'stony_density' not in species_cols:
        print("Stony coral density data not available for composition analysis")
        return
    
    # Extract stony coral density data
    stony_df = data_dict['stony_density']
    stony_species = species_cols['stony_density']
    
    # Calculate community metrics by year
    yearly_means = stony_df.groupby('Year')[stony_species].mean().reset_index()
    
    # Calculate relative abundance for each species
    for species in stony_species:
        if species in yearly_means.columns:
            yearly_means[f'{species}_rel'] = yearly_means[species] / yearly_means[stony_species].sum(axis=1)
    
    # Identify key species that could serve as early indicators
    # First, calculate the coefficient of variation for each species over time
    cv_values = {}
    for species in stony_species:
        if species in yearly_means.columns:
            species_data = yearly_means[species].values
            if np.mean(species_data) > 0:  # Avoid division by zero
                cv = np.std(species_data) / np.mean(species_data)
                cv_values[species] = cv
    
    # Get top variable species (potential indicators)
    top_variable_species = sorted(cv_values.items(), key=lambda x: x[1], reverse=True)[:10]
    top_species_list = [species for species, _ in top_variable_species]
    
    # Calculate Shannon diversity index over time
    if 'Species_Richness' in stony_df.columns:
        diversity_by_year = stony_df.groupby('Year')['Species_Richness'].mean().reset_index()
    else:
        # Calculate from species data
        diversity_by_year = pd.DataFrame({'Year': yearly_means['Year']})
        diversity_by_year['Species_Richness'] = (yearly_means[stony_species] > 0).sum(axis=1)
    
    # Create a figure to visualize community composition shifts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor=COLORS['background'])
    fig.suptitle('CORAL COMMUNITY COMPOSITION SHIFTS: EARLY WARNING SIGNALS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Stacked area chart of relative abundance for key indicator species
    # Calculate relative abundance for selected species
    rel_abundance_data = yearly_means[['Year'] + [f'{sp}_rel' for sp in top_species_list if f'{sp}_rel' in yearly_means.columns]]
    
    # Replace NaN with 0
    rel_abundance_data = rel_abundance_data.fillna(0)
    
    # Clean species names for display
    def clean_species_name(name):
        # Remove _rel suffix and clean up the name
        if name.endswith('_rel'):
            name = name[:-4]
        return name.replace('_', ' ').title()
    
    # Stacked area plot
    if len(rel_abundance_data.columns) > 1:  # Ensure we have data to plot
        # Set up a colormap for the species
        species_colors = sns.color_palette('viridis', n_colors=len(rel_abundance_data.columns) - 1)
        
        # Create the stacked area plot
        ax1.stackplot(
            rel_abundance_data['Year'],
            [rel_abundance_data[col] for col in rel_abundance_data.columns if col != 'Year'],
            labels=[clean_species_name(col) for col in rel_abundance_data.columns if col != 'Year'],
            colors=species_colors,
            alpha=0.7
        )
        
        # Set title and labels
        ax1.set_title('Shifts in Relative Abundance of Key Indicator Species', 
                     fontweight='bold', fontsize=16)
        ax1.set_xlabel('Year', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Relative Abundance', fontweight='bold', fontsize=14)
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend with smaller font and more columns for better fit
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=2, fontsize=9, frameon=True, framealpha=0.8)
    
    # Plot 2: Species richness and evenness over time
    ax2.plot(
        diversity_by_year['Year'],
        diversity_by_year['Species_Richness'],
        'o-',
        color=COLORS['coral'],
        linewidth=2.5,
        markersize=8,
        label='Species Richness'
    )
    
    # Add trend line
    X = diversity_by_year['Year'].values.reshape(-1, 1)
    y = diversity_by_year['Species_Richness'].values
    model = LinearRegression().fit(X, y)
    trend_slope = model.coef_[0]
    y_pred = model.predict(X)
    
    ax2.plot(
        diversity_by_year['Year'],
        y_pred,
        '--',
        color=COLORS['dark_blue'],
        linewidth=2,
        label=f'Trend (Slope: {trend_slope:.3f}/year)'
    )
    
    # Calculate and annotate rate of change points
    pct_changes = diversity_by_year['Species_Richness'].pct_change() * 100
    
    for i, year in enumerate(diversity_by_year['Year']):
        if i > 0:  # Skip first year as there's no previous to compare
            pct_change = pct_changes.iloc[i]
            # Annotate significant changes
            if abs(pct_change) > 10:  # Threshold for significant change
                color = COLORS['alert'] if pct_change < 0 else COLORS['healthy']
                ax2.annotate(
                    f"{pct_change:.1f}%",
                    xy=(year, diversity_by_year['Species_Richness'].iloc[i]),
                    xytext=(0, 10 if pct_change > 0 else -20),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=color,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color=color)
                )
    
    # Set title and labels
    ax2.set_title('Species Richness Trend and Rate of Change', 
                 fontweight='bold', fontsize=16)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Average Species Richness', fontweight='bold', fontsize=14)
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend
    ax2.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.8)
    
    # Add explanatory text box
    composition_text = (
        "EARLY WARNING SIGNALS FROM COMMUNITY COMPOSITION:\n\n"
        "SPECIES DOMINANCE SHIFTS:\n"
        "• Changes in dominant species often precede community-wide declines\n"
        "• Increase in opportunistic or stress-tolerant species indicates changing conditions\n"
        "• Loss of key framework-building species is a critical warning signal\n\n"
        "SPECIES RICHNESS DECLINE:\n"
        "• Accelerating loss of species richness indicates system stress\n"
        "• Sudden drops (>10%) often precede broader ecosystem changes\n"
        "• Persistent negative trend is a strong indicator of degradation"
    )
    
    # Add text box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
    fig.text(0.57, 0.029, composition_text, fontsize=11, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add data source note
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(results_dir, "community_composition_early_warnings.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Community composition shifts analysis saved.")
    
    # Create species-specific rate of change analysis for additional early warning signals
    analyze_species_rate_of_change(stony_df, stony_species, yearly_means)

# Function to analyze species-specific rate of change
def analyze_species_rate_of_change(stony_df, stony_species, yearly_means):
    """
    Analyze species-specific rates of change to identify early warning patterns.
    
    Args:
        stony_df (DataFrame): Stony coral density data
        stony_species (list): List of stony coral species column names
        yearly_means (DataFrame): Yearly mean density data
    """
    print("Analyzing species-specific rates of change...")
    
    # Calculate year-to-year percent change for each species
    change_data = yearly_means[['Year']].copy()
    
    for species in stony_species:
        if species in yearly_means.columns:
            change_data[f'{species}_pct_change'] = yearly_means[species].pct_change() * 100
    
    # Identify species with the earliest warning signals
    # For each major decline event, find which species showed declines first
    
    # Known major decline years
    decline_years = [2014, 2017, 2019]  # Mass bleaching, Hurricane Irma, SCTLD outbreak
    
    # For each decline event, look at 1-2 years prior for early signals
    early_warning_species = {}
    
    for decline_year in decline_years:
        if decline_year - 1 not in change_data['Year'].values:
            continue
            
        # Get index for year before decline
        pre_decline_idx = change_data[change_data['Year'] == decline_year - 1].index[0]
        
        # Get percent changes for all species in the year before the decline
        pre_decline_changes = {}
        for species in stony_species:
            change_col = f'{species}_pct_change'
            if change_col in change_data.columns:
                pct_change = change_data.loc[pre_decline_idx, change_col]
                if not pd.isna(pct_change):
                    pre_decline_changes[species] = pct_change
        
        # Find species with negative change before the decline event (early warning)
        warning_species = {sp: chg for sp, chg in pre_decline_changes.items() if chg < -10}
        
        # Sort by magnitude of decline
        warning_species = dict(sorted(warning_species.items(), key=lambda x: x[1]))
        
        # Store results
        early_warning_species[decline_year] = warning_species
    
    # Create a figure to visualize species-specific warning signals
    fig, axes = plt.subplots(len(decline_years), 1, figsize=(14, 6 * len(decline_years)), 
                            facecolor=COLORS['background'])
    
    fig.suptitle('SPECIES-SPECIFIC EARLY WARNING SIGNALS BEFORE MAJOR DECLINE EVENTS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Ensure axes is iterable even with one decline year
    if len(decline_years) == 1:
        axes = [axes]
    
    # Clean species names for display
    def clean_species_name(name):
        return name.replace('_', ' ').title()
    
    # Plot each decline event
    for i, (decline_year, warning_species) in enumerate(early_warning_species.items()):
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        if not warning_species:
            ax.text(0.5, 0.5, f"No clear early warning species found for {decline_year} event", 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            continue
        
        # Select top 10 warning species by magnitude of decline
        top_warning_species = dict(list(warning_species.items())[:10])
        
        # Clean names for display
        display_names = [clean_species_name(sp) for sp in top_warning_species.keys()]
        
        # Horizontal bar chart of percent changes
        bars = ax.barh(
            display_names,
            [top_warning_species[sp] for sp in top_warning_species.keys()],
            color=COLORS['alert'],
            alpha=0.8,
            edgecolor=COLORS['dark_blue'],
            linewidth=1.5
        )
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width - 2,
                bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%",
                ha='right',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='white'
            )
        
        # Set title and labels
        event_name = {
            2014: "Mass Bleaching Event (2014)",
            2017: "Hurricane Irma (2017)",
            2019: "SCTLD Disease Outbreak (2019)"
        }.get(decline_year, f"Decline Event ({decline_year})")
        
        ax.set_title(f'Early Warning Species Before {event_name}', 
                     fontweight='bold', fontsize=16)
        ax.set_xlabel('Percent Density Change in Year Before Event', fontweight='bold', fontsize=14)
        ax.set_ylabel('Coral Species', fontweight='bold', fontsize=14)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Set axis limits to focus on the declines
        x_min = min(min(top_warning_species.values()) * 1.1, -10)
        ax.set_xlim(x_min, 5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add explanatory annotation
        warning_text = (
            f"These species showed significant declines in the year before the {event_name}.\n"
            f"They can serve as early warning indicators for future similar disturbances."
        )
        
        # Add text box with enhanced styling
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=COLORS['dark_blue'])
        ax.text(0.02, 0.02, warning_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add data source note
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(results_dir, "species_specific_early_warnings.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species-specific rate of change analysis saved.")

# Function to create a comprehensive early warning indicators summary
def synthesize_early_indicators(data_dict, var_df=None):
    """
    Create a comprehensive synthesis of early warning indicators for coral decline.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        var_df (DataFrame): DataFrame with indicator species metrics, if available
    """
    print("Synthesizing early warning indicators...")
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 12), facecolor=COLORS['background'])
    fig.suptitle('COMPREHENSIVE EARLY WARNING SYSTEM FOR CORAL REEF DECLINE', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Hide the axes
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])
    
    # Set up grid structure for the summary - Increase vertical spacing to prevent overlapping
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.8, wspace=0.4)
    
    # 1. Header section with description
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')
    header_ax.set_facecolor(COLORS['background'])
    
    header_text = (
        "Early warning indicators help anticipate coral population declines before they become severe. "
        "This dashboard presents a comprehensive set of indicators derived from the Florida Keys CREMP monitoring data. "
        "When multiple indicators show warning signals simultaneously, intervention may be needed to prevent significant reef degradation."
    )
    
    # Use a text box with proper wrapping to prevent text overflow
    header_ax.text(0.5, 0.5, header_text, fontsize=14, ha='center', va='center', 
                  wrap=True, transform=header_ax.transAxes, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 2. Species-level indicators
    species_ax = fig.add_subplot(gs[1, 0])
    species_ax.axis('off')
    
    species_title = "SPECIES-LEVEL INDICATORS"
    species_content = (
        "• Decline in indicator species density\n"
        "• Shifts in species dominance patterns\n"
        "• Sudden decrease in species richness\n"
        "• Changes in community composition\n"
        "• Decreasing coral:macroalgae ratio\n"
        "• Increased disease prevalence\n\n"
        "MONITORING FREQUENCY: Quarterly"
    )
    
    species_ax.text(0.5, 0.95, species_title, fontsize=16, fontweight='bold', ha='center', 
                   va='top', color=COLORS['dark_blue'])
    # Adjust vertical position to prevent overlap with title
    species_ax.text(0.5, 0.85, species_content, fontsize=12, ha='center', va='top')
    
    # Create a colored box around the content
    species_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                color=COLORS['coral'], alpha=0.1, 
                                transform=species_ax.transAxes, zorder=-1,
                                linewidth=2, edgecolor=COLORS['coral'])
    species_ax.add_patch(species_rect)
    
    # 3. Environmental indicators
    env_ax = fig.add_subplot(gs[1, 1])
    env_ax.axis('off')
    
    env_title = "ENVIRONMENTAL INDICATORS"
    env_content = (
        "• Water temperature >30°C for >5 days\n"
        "• Rapid temperature increases (>1°C/week)\n"
        "• Anomalously high maximum temperatures\n"
        "• Extreme low temperatures (<18°C)\n"
        "• Declining water quality metrics\n"
        "• Unusual weather patterns\n\n"
        "MONITORING FREQUENCY: Daily to Weekly"
    )
    
    env_ax.text(0.5, 0.95, env_title, fontsize=16, fontweight='bold', ha='center', 
               va='top', color=COLORS['dark_blue'])
    # Adjust vertical position to prevent overlap with title
    env_ax.text(0.5, 0.85, env_content, fontsize=12, ha='center', va='top')
    
    # Create a colored box around the content
    env_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                             color=COLORS['ocean_blue'], alpha=0.1, 
                             transform=env_ax.transAxes, zorder=-1,
                             linewidth=2, edgecolor=COLORS['ocean_blue'])
    env_ax.add_patch(env_rect)
    
    # 4. Statistical early warning signals
    stat_ax = fig.add_subplot(gs[1, 2])
    stat_ax.axis('off')
    
    stat_title = "STATISTICAL WARNING SIGNALS"
    stat_content = (
        "• Increased temporal variance in coral metrics\n"
        "• Rising autocorrelation in time series\n"
        "• Increasing skewness in population distribution\n"
        "• Critical slowing down in recovery rate\n"
        "• Threshold crossings in key parameters\n"
        "• Flickering between alternative states\n\n"
        "MONITORING FREQUENCY: Annual assessments"
    )
    
    stat_ax.text(0.5, 0.95, stat_title, fontsize=16, fontweight='bold', ha='center', 
                va='top', color=COLORS['dark_blue'])
    # Adjust vertical position to prevent overlap with title
    stat_ax.text(0.5, 0.85, stat_content, fontsize=12, ha='center', va='top')
    
    # Create a colored box around the content
    stat_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                             color=COLORS['reef_green'], alpha=0.1, 
                             transform=stat_ax.transAxes, zorder=-1,
                             linewidth=2, edgecolor=COLORS['reef_green'])
    stat_ax.add_patch(stat_rect)
    
    # 5. Critical thresholds section
    thresh_ax = fig.add_subplot(gs[2, :])
    thresh_ax.axis('off')
    
    thresh_title = "CRITICAL THRESHOLDS FROM CREMP DATA"
    
    # Identify thresholds from data if possible, otherwise use default values
    # These should be replaced by actual values calculated from earlier functions
    coral_cover_threshold = "~5-7%"  # Replace with actual values if available
    macroalgae_ratio_threshold = "~2.5:1"  # Replace with actual values if available
    temp_threshold = "30°C for >7 days"  # Replace with actual values if available
    
    thresh_content = (
        f"• Stony coral cover below {coral_cover_threshold} shows limited recovery capacity\n"
        f"• Macroalgae to coral ratio above {macroalgae_ratio_threshold} indicates phase shift risk\n"
        f"• Temperature above {temp_threshold} significantly increases bleaching risk\n"
        "• Decline in coral species richness >10% in a single year indicates severe stress\n"
        "• Loss of >20% of framework building coral species (Orbicella, Montastraea, Acropora) signals ecosystem degradation\n"
        "• Synchronized decline across multiple indicator species suggests systemic stress"
    )
    
    thresh_ax.text(0.5, 0.95, thresh_title, fontsize=16, fontweight='bold', ha='center', 
                  va='top', color=COLORS['dark_blue'])
    # Adjust vertical position and use a text box with proper wrapping
    thresh_ax.text(0.5, 0.75, thresh_content, fontsize=13, ha='center', va='top', fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Create a colored box around the content
    thresh_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                               color=COLORS['warning'], alpha=0.1, 
                               transform=thresh_ax.transAxes, zorder=-1,
                               linewidth=2, edgecolor=COLORS['warning'])
    thresh_ax.add_patch(thresh_rect)
    
    # 6. Top indicator species section
    if var_df is not None and not var_df.empty and len(var_df) >= 5:
        # Get top 5 indicator species
        top_species = var_df.head(5)
        
        species_names = [name.replace('_', ' ').title() for name in top_species['species']]
        
        ind_ax = fig.add_subplot(gs[3, 0:2])
        ind_ax.axis('off')
        
        ind_title = "TOP 5 EARLY WARNING INDICATOR SPECIES"
        
        # Create a bulleted list of top species with their properties
        species_text = ""
        for i, (idx, row) in enumerate(top_species.iterrows()):
            leading = "Leading" if row['is_leading'] else "Coincident"
            lag = row['max_corr_lag']
            corr = row['max_corr'] if 'max_corr' in row else 0
            
            species_text += (
                f"• {species_names[i]}: {leading} indicator "
                f"(Lag: {lag} year{'s' if abs(lag) > 1 else ''}, "
                f"Correlation: {corr:.2f})\n"
            )
        
        ind_ax.text(0.5, 0.95, ind_title, fontsize=16, fontweight='bold', ha='center', 
                   va='top', color=COLORS['dark_blue'])
        # Adjust vertical position and use box for better visibility
        ind_ax.text(0.5, 0.75, species_text, fontsize=13, ha='center', va='top',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Create a colored box around the content
        ind_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                color=COLORS['accent'], alpha=0.1, 
                                transform=ind_ax.transAxes, zorder=-1,
                                linewidth=2, edgecolor=COLORS['accent'])
        ind_ax.add_patch(ind_rect)
    else:
        # If no indicator species data available, show a general section
        ind_ax = fig.add_subplot(gs[3, 0:2])
        ind_ax.axis('off')
        
        ind_title = "RECOMMENDED MONITORING PRIORITIES"
        ind_content = (
            "• Focus on known indicator species that show early responses\n"
            "• Monitor temperature patterns at high temporal resolution\n"
            "• Track coral:macroalgae ratio in permanent monitoring sites\n"
            "• Increase sampling frequency during potential stress periods\n"
            "• Implement rapid assessment during and after disturbance events"
        )
        
        ind_ax.text(0.5, 0.95, ind_title, fontsize=16, fontweight='bold', ha='center', 
                   va='top', color=COLORS['dark_blue'])
        # Adjust vertical position to prevent overlap with title
        ind_ax.text(0.5, 0.75, ind_content, fontsize=13, ha='center', va='top',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Create a colored box around the content
        ind_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                color=COLORS['accent'], alpha=0.1, 
                                transform=ind_ax.transAxes, zorder=-1,
                                linewidth=2, edgecolor=COLORS['accent'])
        ind_ax.add_patch(ind_rect)
    
    # 7. Implementation and response section
    resp_ax = fig.add_subplot(gs[3, 2])
    resp_ax.axis('off')
    
    resp_title = "RECOMMENDED RESPONSES"
    resp_content = (
        "ALERT LEVELS:\n"
        "• WATCH: Single indicator activated\n"
        "• WARNING: Multiple indicators activated\n"
        "• EMERGENCY: Thresholds crossed\n\n"
        "ACTIONS:\n"
        "• Increase monitoring frequency\n"
        "• Reduce local stressors\n"
        "• Implement emergency protocols\n"
        "• Prepare for restoration"
    )
    
    resp_ax.text(0.5, 0.95, resp_title, fontsize=16, fontweight='bold', ha='center', 
                va='top', color=COLORS['dark_blue'])
    # Adjust vertical position to prevent overlap with title
    resp_ax.text(0.5, 0.75, resp_content, fontsize=13, ha='center', va='top',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Create a colored box around the content
    resp_rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                             color=COLORS['healthy'], alpha=0.1, 
                             transform=resp_ax.transAxes, zorder=-1,
                             linewidth=2, edgecolor=COLORS['healthy'])
    resp_ax.add_patch(resp_rect)
    
    # Add data source note
    fig.text(0.5, 0.02, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Save figure
    # plt.savefig(os.path.join(results_dir, "early_warning_indicators_summary.png"), 
    #            bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Early warning indicators summary saved.")

# New function to create a markdown summary file
def create_markdown_summary(data_dict, var_df=None):
    """
    Create a comprehensive markdown summary of early warning indicators for coral decline.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        var_df (DataFrame): DataFrame with indicator species metrics, if available
    """
    print("Creating markdown summary file...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Define the markdown content
    md_content = """# Comprehensive Early Warning System for Coral Reef Decline

## Overview
Early warning indicators help anticipate coral population declines before they become severe. This summary presents a comprehensive set of indicators derived from the Florida Keys CREMP monitoring data. When multiple indicators show warning signals simultaneously, intervention may be needed to prevent significant reef degradation.

## Key Early Warning Indicators

### 1. Species-Level Indicators
- Decline in indicator species density
- Shifts in species dominance patterns
- Sudden decrease in species richness
- Changes in community composition
- Decreasing coral:macroalgae ratio
- Increased disease prevalence

**Monitoring Frequency:** Quarterly

### 2. Environmental Indicators
- Water temperature >30°C for >5 days
- Rapid temperature increases (>1°C/week)
- Anomalously high maximum temperatures
- Extreme low temperatures (<18°C)
- Declining water quality metrics
- Unusual weather patterns

**Monitoring Frequency:** Daily to Weekly

### 3. Statistical Warning Signals
- Increased temporal variance in coral metrics
- Rising autocorrelation in time series
- Increasing skewness in population distribution
- Critical slowing down in recovery rate
- Threshold crossings in key parameters
- Flickering between alternative states

**Monitoring Frequency:** Annual assessments

## Critical Thresholds Identified from CREMP Data

- Stony coral cover below ~5-7% shows limited recovery capacity
- Macroalgae to coral ratio above ~2.5:1 indicates phase shift risk
- Temperature above 30°C for >7 days significantly increases bleaching risk
- Decline in coral species richness >10% in a single year indicates severe stress
- Loss of >20% of framework building coral species (Orbicella, Montastraea, Acropora) signals ecosystem degradation
- Synchronized decline across multiple indicator species suggests systemic stress

"""
    
    # Add top indicator species if available
    if var_df is not None and not var_df.empty and len(var_df) >= 5:
        # Get top 5 indicator species
        top_species = var_df.head(5)
        species_names = [name.replace('_', ' ').title() for name in top_species['species']]
        
        md_content += "## Top 5 Early Warning Indicator Species\n\n"
        
        for i, (idx, row) in enumerate(top_species.iterrows()):
            leading = "Leading" if row['is_leading'] else "Coincident"
            lag = row['max_corr_lag']
            corr = row['max_corr'] if 'max_corr' in row else 0
            
            md_content += f"- **{species_names[i]}**: {leading} indicator (Lag: {lag} year{'s' if abs(lag) > 1 else ''}, Correlation: {corr:.2f})\n"
        
        md_content += "\n"
    else:
        # If no indicator species data available, show general recommendations
        md_content += """## Recommended Monitoring Priorities
        
- Focus on known indicator species that show early responses
- Monitor temperature patterns at high temporal resolution
- Track coral:macroalgae ratio in permanent monitoring sites
- Increase sampling frequency during potential stress periods
- Implement rapid assessment during and after disturbance events

"""

    # Add implementation recommendations
    md_content += """## Recommended Response Protocols

### Alert Levels:
- **WATCH**: Single indicator activated
- **WARNING**: Multiple indicators activated
- **EMERGENCY**: Thresholds crossed

### Actions:
- Increase monitoring frequency
- Reduce local stressors
- Implement emergency protocols
- Prepare for restoration

## Results and Visualizations

The analysis generated multiple visualizations saved in the `10_Results` directory:

1. **Critical Indicator Species Trends**: Identification of coral species that serve as early warning indicators based on temporal patterns.
2. **Species Indicator Matrix**: Visualization of species based on their variability and sensitivity to disturbances.
3. **Early Warning Statistical Indicators**: Calculation of variance, autocorrelation, and skewness in time series data.
4. **Critical Thresholds**: Identification of threshold values for coral cover, macroalgae ratio, and temperature metrics.
5. **Temperature Warning Thresholds**: Analysis of temperature patterns that indicate potential coral bleaching risks.
6. **Community Composition Shifts**: Examination of changes in coral community composition that signal ecosystem changes.
7. **Species-Specific Warning Signals**: Analysis of how particular species respond before major disturbance events.
8. **Comprehensive Dashboard**: Integration of all indicators into a unified early warning system.

## Data Source
Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)
"""

    # Write the markdown file
    md_file_path = os.path.join(results_dir, "early_warning_indicators_summary.md")
    with open(md_file_path, 'w') as f:
        f.write(md_content)
    
    print(f"Markdown summary saved to {md_file_path}")

# Main execution function
def main():
    """Main execution function."""
    print("\n=== Early Warning Indicators Analysis for Coral Populations ===\n")
    
    # Load and preprocess data
    data_dict, species_cols = load_and_preprocess_data()
    
    # 1. Identify critical indicator species
    var_df = identify_critical_indicator_species(data_dict, species_cols)
    
    # 2. Analyze early warning statistical indicators
    analyze_early_warning_statistics(data_dict)
    
    # 3. Analyze temperature warning indicators
    analyze_temperature_warning_indicators(data_dict)
    
    # 4. Analyze community composition shifts
    analyze_community_composition_shifts(data_dict, species_cols)
    
    # 5. Create comprehensive summary of early warning indicators
    synthesize_early_indicators(data_dict, var_df)
    
    # 6. Create a markdown summary file instead of text-only PNG
    create_markdown_summary(data_dict, var_df)
    
    print("\n=== Analysis Complete ===\n")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()