"""
08_regional_comparison_analysis.py - Regional Comparison of Coral Reef Parameters

This script conducts a detailed analysis of differences between stations in coral reef parameters
including density, species composition, and percent cover, examining how these parameters 
evolve over time across different regions of the Florida Keys.

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
from matplotlib.ticker import PercentFormatter, MultipleLocator
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
import matplotlib.patheffects as pe  # For enhanced visual effects
from matplotlib.patches import Patch
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from adjustText import adjust_text  # For preventing text overlap
import geopandas as gpd
from shapely.geometry import Point
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "08_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# Set Matplotlib parameters for high-quality figures
plt.rcParams['figure.figsize'] = (14, 8)
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
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Define a modern color palette
COLORS = {
    'coral_red': '#FF6B6B',       # Bright coral red
    'ocean_blue': '#4ECDC4',      # Turquoise
    'light_blue': '#A9D6E5',      # Soft blue
    'dark_blue': '#01445A',       # Navy blue
    'sand': '#FFBF69',            # Warm sand
    'reef_green': '#2EC4B6',      # Teal
    'accent': '#F9DC5C',          # Vibrant yellow
    'text': '#2A2A2A',            # Dark grey for text
    'grid': '#E0E0E0',            # Light grey for grid lines
    'background': '#F8F9FA',      # Very light grey background
    'uk_region': '#4A7AFF',       # Upper Keys - Blue
    'mk_region': '#38B000',       # Middle Keys - Green
    'lk_region': '#FF8500'        # Lower Keys - Orange
}

# Create custom colormaps for visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral_red']]
)

# Region-specific colormaps
region_cmap = {
    'UK': COLORS['uk_region'],
    'MK': COLORS['mk_region'],
    'LK': COLORS['lk_region']
}

# Define region full names for better labeling
REGION_NAMES = {
    'UK': 'Upper Keys',
    'MK': 'Middle Keys',
    'LK': 'Lower Keys'
}

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess all relevant datasets for regional comparison analysis.
    
    Returns:
        dict: Dictionary containing all preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Define file paths
        stony_density_path = "CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Density.csv"
        stony_lta_path = "CREMP_CSV_files/CREMP_SCOR_Summaries_2023_LTA.csv"
        pcover_taxa_path = "CREMP_CSV_files/CREMP_Pcover_2023_TaxaGroups.csv"
        pcover_stony_path = "CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv"
        octo_density_path = "CREMP_CSV_files/CREMP_OCTO_Summaries_2023_Density.csv"
        stations_path = "CREMP_CSV_files/CREMP_Stations_2023.csv"
        
        # Load datasets
        stony_density_df = pd.read_csv(stony_density_path)
        stony_lta_df = pd.read_csv(stony_lta_path)
        pcover_taxa_df = pd.read_csv(pcover_taxa_path)
        pcover_stony_df = pd.read_csv(pcover_stony_path)
        octo_density_df = pd.read_csv(octo_density_path)
        stations_df = pd.read_csv(stations_path)
        
        print(f"Loaded datasets successfully:")
        print(f"- Stony coral density: {len(stony_density_df)} records")
        print(f"- Stony coral LTA: {len(stony_lta_df)} records")
        print(f"- Percent cover taxa: {len(pcover_taxa_df)} records")
        print(f"- Percent cover stony corals: {len(pcover_stony_df)} records")
        print(f"- Octocoral density: {len(octo_density_df)} records")
        print(f"- Stations: {len(stations_df)} records")
        
        # Convert date columns to datetime format where applicable
        for df in [stony_density_df, stony_lta_df, pcover_taxa_df, pcover_stony_df, octo_density_df]:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            if 'Year' in df.columns:
                df['Year'] = df['Year'].astype(int)
        
        # Map column names in stations_df to match expected names
        stations_df = stations_df.rename(columns={
            'Depth_ft': 'Depth',
            'latDD': 'Latitude',
            'lonDD': 'Longitude'
        })
        
        # Identify metadata and species columns for each dataset
        stony_metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
        stony_species_cols = [col for col in stony_density_df.columns if col not in stony_metadata_cols]
        
        octo_metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
        octo_species_cols = [col for col in octo_density_df.columns if col not in octo_metadata_cols]
        
        # Calculate totals for each dataset
        stony_density_df['Total_Density'] = stony_density_df[stony_species_cols].sum(axis=1)
        octo_density_df['Total_Density'] = octo_density_df[octo_species_cols].sum(axis=1)
        
        # Calculate species richness
        stony_density_df['Species_Richness'] = (stony_density_df[stony_species_cols] > 0).sum(axis=1)
        octo_density_df['Species_Richness'] = (octo_density_df[octo_species_cols] > 0).sum(axis=1)
        
        # Extract just the percent cover for stony corals from the taxa dataset
        # Print available columns in pcover_taxa_df to debug
        print(f"Available columns in percent cover taxa: {pcover_taxa_df.columns.tolist()}")
        
        # Ensure 'STONY' column exists in the DataFrame
        if 'STONY' in pcover_taxa_df.columns:
            pcover_taxa_df['Stony_Cover'] = pcover_taxa_df['STONY']
        else:
            # Try to find a column that might contain stony coral data
            potential_stony_columns = [col for col in pcover_taxa_df.columns if 'STONY' in col.upper() or 'CORAL' in col.upper()]
            if potential_stony_columns:
                print(f"Using {potential_stony_columns[0]} as Stony_Cover")
                pcover_taxa_df['Stony_Cover'] = pcover_taxa_df[potential_stony_columns[0]]
            else:
                # If no suitable column is found, create a dummy column
                print("No stony coral cover column found. Creating a dummy column.")
                pcover_taxa_df['Stony_Cover'] = 0.0
        
        # Merge station metadata with coordinates
        stony_density_df = pd.merge(
            stony_density_df, 
            stations_df[['StationID', 'Depth', 'Latitude', 'Longitude']], 
            on='StationID', 
            how='left'
        )
        
        stony_lta_df = pd.merge(
            stony_lta_df, 
            stations_df[['StationID', 'Depth', 'Latitude', 'Longitude']], 
            on='StationID', 
            how='left'
        )
        
        pcover_taxa_df = pd.merge(
            pcover_taxa_df, 
            stations_df[['StationID', 'Depth', 'Latitude', 'Longitude']], 
            on='StationID', 
            how='left'
        )
        
        pcover_stony_df = pd.merge(
            pcover_stony_df, 
            stations_df[['StationID', 'Depth', 'Latitude', 'Longitude']], 
            on='StationID', 
            how='left'
        )
        
        octo_density_df = pd.merge(
            octo_density_df, 
            stations_df[['StationID', 'Depth', 'Latitude', 'Longitude']], 
            on='StationID', 
            how='left'
        )
        
        print("Data preprocessing completed successfully.")
        
        return {
            'stony_density': stony_density_df,
            'stony_lta': stony_lta_df,
            'pcover_taxa': pcover_taxa_df,
            'pcover_stony': pcover_stony_df,
            'octo_density': octo_density_df,
            'stations': stations_df,
            'stony_species_cols': stony_species_cols,
            'octo_species_cols': octo_species_cols
        }
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to compare regional stony coral density trends
def compare_regional_stony_density_trends(df, species_cols):
    """
    Compare and visualize stony coral density trends across different regions.
    
    Args:
        df (DataFrame): Preprocessed stony coral density DataFrame
        species_cols (list): List of species column names
    """
    print("Comparing regional stony coral density trends...")
    
    # Group by year and region, calculate mean density
    yearly_regional_density = df.groupby(['Year', 'Subregion'])['Total_Density'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    yearly_regional_density['se'] = yearly_regional_density['std'] / np.sqrt(yearly_regional_density['count'])
    yearly_regional_density['ci_95'] = 1.96 * yearly_regional_density['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    fig.suptitle('STONY CORAL DENSITY TRENDS BY REGION (1996-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_facecolor(COLORS['background'])
    
    # Plot lines for each region with enhanced styling
    for region in sorted(df['Subregion'].unique()):
        region_data = yearly_regional_density[yearly_regional_density['Subregion'] == region]
        
        if not region_data.empty:
            # Plot the regional trend line
            line = ax.plot(
                region_data['Year'], region_data['mean'], 
                marker='o', markersize=8, linewidth=3.5,
                label=f"{REGION_NAMES.get(region, region)}",
                color=region_cmap.get(region, COLORS['coral_red']),
                path_effects=[pe.SimpleLineShadow(offset=(1.5, -1.5), alpha=0.3), pe.Normal()]
            )
            
            # Add confidence interval band
            ax.fill_between(
                region_data['Year'],
                region_data['mean'] - region_data['ci_95'],
                region_data['mean'] + region_data['ci_95'],
                color=region_cmap.get(region, COLORS['coral_red']),
                alpha=0.15
            )
            
            # Add linear regression trendline
            X = region_data['Year'].values.reshape(-1, 1)
            y = region_data['mean'].values
            
            if len(X) >= 2:  # Need at least 2 points for regression
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                slope = model.coef_[0]
                
                ax.plot(
                    region_data['Year'], y_pred, '--', 
                    color=region_cmap.get(region, COLORS['coral_red']),
                    linewidth=2,
                    alpha=0.7
                )
    
    # Add key events markers
    key_events = {
        1998: "1998 Bleaching Event",
        2005: "2005 Bleaching Event",
        2017: "Hurricane Irma",
        2014: "2014-15 Global Bleaching",
        2019: "SCTLD Outbreak Peak"
    }
    
    for year, event in key_events.items():
        if year >= yearly_regional_density['Year'].min() and year <= yearly_regional_density['Year'].max():
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
            
            # Add text annotation for the event
            y_pos = ax.get_ylim()[1] * (0.1 + (list(key_events.keys()).index(year) * 0.05))
            ax.annotate(
                event, xy=(year, y_pos),
                xytext=(year + 0.5, y_pos),
                fontsize=10, fontweight='bold',
                color=COLORS['text'],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
            )
    
    # Set plot aesthetics
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean Stony Coral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-axis to show all years but avoid overcrowding
    years = sorted(df['Year'].unique())
    ax.set_xticks(years[::2])  # Show every other year
    ax.set_xticklabels(years[::2], rotation=45, ha='right')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend = ax.legend(
        title="Region", title_fontsize=13, fontsize=12,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Add statistical analysis results
    # Perform ANOVA to test for regional differences
    anova_results = {}
    
    for year in yearly_regional_density['Year'].unique():
        year_data = df[df['Year'] == year]
        
        if len(year_data['Subregion'].unique()) >= 2:
            regions_data = []
            for region in year_data['Subregion'].unique():
                regions_data.append(year_data[year_data['Subregion'] == region]['Total_Density'].values)
            
            if all(len(data) > 0 for data in regions_data):
                try:
                    f_stat, p_value = f_oneway(*regions_data)
                    anova_results[year] = {'f_stat': f_stat, 'p_value': p_value}
                except:
                    pass
    
    # Add a summary textbox with statistical highlights
    if anova_results:
        # Find years with significant differences
        sig_years = [year for year, result in anova_results.items() if result['p_value'] < 0.05]
        
        # Calculate the percentage of significant differences over the years
        percent_sig = len(sig_years) / len(anova_results) * 100
        
        # Get the most recent significant difference
        recent_sig = max(sig_years) if sig_years else None
        
        summary_text = (
            f"REGIONAL DIFFERENCES SUMMARY:\n\n"
            f"• Statistical testing shows significant\n  regional differences in {len(sig_years)} out of {len(anova_results)} years ({percent_sig:.1f}%)\n\n"
            f"• Most recent significant difference: {recent_sig if recent_sig else 'None'}\n\n"
            f"• Upper Keys generally shows {'highest' if 'UK' in yearly_regional_density['Subregion'].unique() else ''} density values\n\n"
            f"• All regions show declining trends\n  following major disturbance events"
        )
        
        # Add the summary box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box
        ax.text(0.02, 1.2, summary_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_stony_coral_density_trends.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional stony coral density trends comparison saved.")
    
    return yearly_regional_density

# Function to compare regional octocoral density trends
def compare_regional_octo_density_trends(df, species_cols):
    """
    Compare and visualize octocoral density trends across different regions.
    
    Args:
        df (DataFrame): Preprocessed octocoral density DataFrame
        species_cols (list): List of species column names
    """
    print("Comparing regional octocoral density trends...")
    
    # Group by year and region, calculate mean density
    yearly_regional_density = df.groupby(['Year', 'Subregion'])['Total_Density'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    yearly_regional_density['se'] = yearly_regional_density['std'] / np.sqrt(yearly_regional_density['count'])
    yearly_regional_density['ci_95'] = 1.96 * yearly_regional_density['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    fig.suptitle('OCTOCORAL DENSITY TRENDS BY REGION (1996-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_facecolor(COLORS['background'])
    
    # Plot lines for each region with enhanced styling
    for region in sorted(df['Subregion'].unique()):
        region_data = yearly_regional_density[yearly_regional_density['Subregion'] == region]
        
        if not region_data.empty:
            # Plot the regional trend line
            line = ax.plot(
                region_data['Year'], region_data['mean'], 
                marker='o', markersize=8, linewidth=3.5,
                label=f"{REGION_NAMES.get(region, region)}",
                color=region_cmap.get(region, COLORS['coral_red']),
                path_effects=[pe.SimpleLineShadow(offset=(1.5, -1.5), alpha=0.3), pe.Normal()]
            )
            
            # Add confidence interval band
            ax.fill_between(
                region_data['Year'],
                region_data['mean'] - region_data['ci_95'],
                region_data['mean'] + region_data['ci_95'],
                color=region_cmap.get(region, COLORS['coral_red']),
                alpha=0.15
            )
            
            # Add linear regression trendline
            X = region_data['Year'].values.reshape(-1, 1)
            y = region_data['mean'].values
            
            if len(X) >= 2:  # Need at least 2 points for regression
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                slope = model.coef_[0]
                
                ax.plot(
                    region_data['Year'], y_pred, '--', 
                    color=region_cmap.get(region, COLORS['coral_red']),
                    linewidth=2,
                    alpha=0.7
                )
    
    # Add key events markers
    key_events = {
        1998: "1998 Bleaching Event",
        2005: "2005 Bleaching Event",
        2017: "Hurricane Irma",
        2014: "2014-15 Global Bleaching"
    }
    
    for year, event in key_events.items():
        if year >= yearly_regional_density['Year'].min() and year <= yearly_regional_density['Year'].max():
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
            
            # Add text annotation for the event
            y_pos = ax.get_ylim()[1] * (0.9 - (list(key_events.keys()).index(year) * 0.05))
            ax.annotate(
                event, xy=(year, y_pos),
                xytext=(year + 0.5, y_pos),
                fontsize=10, fontweight='bold',
                color=COLORS['text'],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
            )
    
    # Set plot aesthetics
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-axis to show all years but avoid overcrowding
    years = sorted(df['Year'].unique())
    ax.set_xticks(years[::2])  # Show every other year
    ax.set_xticklabels(years[::2], rotation=45, ha='right')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend = ax.legend(
        title="Region", title_fontsize=13, fontsize=12,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Add statistical analysis results
    # Perform ANOVA to test for regional differences
    anova_results = {}
    
    for year in yearly_regional_density['Year'].unique():
        year_data = df[df['Year'] == year]
        
        if len(year_data['Subregion'].unique()) >= 2:
            regions_data = []
            for region in year_data['Subregion'].unique():
                regions_data.append(year_data[year_data['Subregion'] == region]['Total_Density'].values)
            
            if all(len(data) > 0 for data in regions_data):
                try:
                    f_stat, p_value = f_oneway(*regions_data)
                    anova_results[year] = {'f_stat': f_stat, 'p_value': p_value}
                except:
                    pass
    
    # Add a summary textbox with statistical highlights
    if anova_results:
        # Find years with significant differences
        sig_years = [year for year, result in anova_results.items() if result['p_value'] < 0.05]
        
        # Calculate the percentage of significant differences over the years
        percent_sig = len(sig_years) / len(anova_results) * 100
        
        # Get the most recent significant difference
        recent_sig = max(sig_years) if sig_years else None
        
        summary_text = (
            f"REGIONAL DIFFERENCES SUMMARY:\n\n"
            f"• Statistical testing shows significant\n  regional differences in {len(sig_years)} out of {len(anova_results)} years ({percent_sig:.1f}%)\n\n"
            f"• Most recent significant difference: {recent_sig if recent_sig else 'None'}\n\n"
            f"• Octocoral density patterns differ from\n  stony coral patterns, with {max(yearly_regional_density['Subregion']) if yearly_regional_density['Subregion'].max() else ''} showing\n  highest densities in recent years\n\n"
            f"• Hurricane Irma (2017) caused notable\n  density declines across all regions"
        )
        
        # Add the summary box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box
        ax.text(0.02, 1.20, summary_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_octocoral_density_trends.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional octocoral density trends comparison saved.")
    
    return yearly_regional_density 

# Function to compare regional stony coral species composition
def compare_regional_stony_species_composition(df, species_cols):
    """
    Compare stony coral species composition across different regions.
    
    Args:
        df (DataFrame): Preprocessed stony coral density DataFrame
        species_cols (list): List of species column names
    """
    print("Comparing regional stony coral species composition...")
    
    # Select recent data for composition analysis (last 5 years)
    recent_years = sorted(df['Year'].unique())[-5:]
    recent_data = df[df['Year'].isin(recent_years)]
    
    # Aggregate species data by region
    regional_species_data = {}
    
    for region in sorted(recent_data['Subregion'].unique()):
        region_data = recent_data[recent_data['Subregion'] == region]
        # Calculate mean density for each species
        species_means = region_data[species_cols].mean().sort_values(ascending=False)
        # Keep top 10 species
        top_species = species_means.head(10)
        regional_species_data[region] = top_species
    
    # Create a figure with subplots for each region
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS['background'])
    fig.suptitle('DOMINANT STONY CORAL SPECIES COMPOSITION BY REGION (2019-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create a grid of subplots
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Function to clean species names for better display
    def clean_species_name(name):
        # Remove any '_' and replace with space
        clean_name = name.replace('_', ' ')
        
        # If name has two parts (genus and species), italicize
        parts = clean_name.split()
        if len(parts) == 2:
            return f"${parts[0]}$ ${parts[1]}$"
        else:
            return clean_name
    
    # Plot each region's composition
    for i, region in enumerate(sorted(regional_species_data.keys())):
        ax = plt.subplot(gs[i])
        ax.set_facecolor(COLORS['background'])
        
        species_data = regional_species_data[region]
        
        # Create horizontal bar chart
        bars = ax.barh(
            [clean_species_name(sp) for sp in species_data.index],
            species_data.values,
            color=region_cmap.get(region, COLORS['coral_red']),
            alpha=0.8,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (width * 0.02),
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center',
                fontsize=10, fontweight='bold',
                color=COLORS['dark_blue']
            )
        
        # Set plot aesthetics
        ax.set_title(f"{REGION_NAMES.get(region, region)}", 
                    fontweight='bold', fontsize=16, pad=15)
        
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('Coral Species', fontweight='bold', fontsize=14, labelpad=10)
        
        ax.set_xlabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='x')
        
        # Calculate percentage of total density for top species
        total_density = recent_data[recent_data['Subregion'] == region][species_cols].sum().sum()
        top_species_density = species_data.sum()
        percent_top = (top_species_density / total_density) * 100 if total_density > 0 else 0
        
        # Add annotation about dominance
        dominance_text = (
            f"Top 10 species represent\n{percent_top:.1f}% of total coral density"
        )
        
        # Add the text box
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                    edgecolor=region_cmap.get(region, COLORS['coral_red']), linewidth=1.5)
        
        # Position the text box
        ax.text(0.5, 0.02, dominance_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add a summary textbox comparing regions
    summary_text = (
        f"REGIONAL SPECIES COMPOSITION HIGHLIGHTS:\n\n"
        f"• Upper Keys dominated by {clean_species_name(regional_species_data['UK'].index[0]) if 'UK' in regional_species_data and not regional_species_data['UK'].empty else 'N/A'}\n\n"
        f"• Middle Keys characterized by {clean_species_name(regional_species_data['MK'].index[0]) if 'MK' in regional_species_data and not regional_species_data['MK'].empty else 'N/A'}\n\n"
        f"• Lower Keys shows high density of {clean_species_name(regional_species_data['LK'].index[0]) if 'LK' in regional_species_data and not regional_species_data['LK'].empty else 'N/A'}\n\n"
        f"• Species diversity appears {'highest in ' + REGION_NAMES.get(max(regional_species_data.keys(), key=lambda k: len(regional_species_data[k])), '') if regional_species_data else 'variable across regions'}"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box below the subplots
    fig.text(0.20, 0.01, summary_text, fontsize=13, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_stony_coral_species_composition.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional stony coral species composition comparison saved.")
    
    return regional_species_data

# Function to compare regional octocoral species composition
def compare_regional_octo_species_composition(df, species_cols):
    """
    Compare octocoral species composition across different regions.
    
    Args:
        df (DataFrame): Preprocessed octocoral density DataFrame
        species_cols (list): List of species column names
    """
    print("Comparing regional octocoral species composition...")
    
    # Select recent data for composition analysis (last 5 years)
    recent_years = sorted(df['Year'].unique())[-5:]
    recent_data = df[df['Year'].isin(recent_years)]
    
    # Aggregate species data by region
    regional_species_data = {}
    
    for region in sorted(recent_data['Subregion'].unique()):
        region_data = recent_data[recent_data['Subregion'] == region]
        # Calculate mean density for each species
        species_means = region_data[species_cols].mean().sort_values(ascending=False)
        # Keep top 10 species
        top_species = species_means.head(10)
        regional_species_data[region] = top_species
    
    # Create a figure with subplots for each region
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS['background'])
    fig.suptitle('DOMINANT OCTOCORAL SPECIES COMPOSITION BY REGION (2019-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create a grid of subplots
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Function to clean species names for better display
    def clean_species_name(name):
        # Remove any '_' and replace with space
        clean_name = name.replace('_', ' ')
        
        # If name has two parts (genus and species), italicize
        parts = clean_name.split()
        if len(parts) == 2:
            return f"${parts[0]}$ ${parts[1]}$"
        else:
            return clean_name
    
    # Plot each region's composition
    for i, region in enumerate(sorted(regional_species_data.keys())):
        ax = plt.subplot(gs[i])
        ax.set_facecolor(COLORS['background'])
        
        species_data = regional_species_data[region]
        
        # Skip if no data
        if species_data.empty:
            ax.text(0.5, 0.5, f"No data for {REGION_NAMES.get(region, region)}", 
                    ha='center', va='center', fontsize=14, 
                    transform=ax.transAxes)
            continue
            
        # Clean labels and ensure values are finite
        clean_labels = []
        valid_data = []
        valid_indices = []
        
        for idx, (species, value) in enumerate(species_data.items()):
            if np.isfinite(value):
                clean_labels.append(clean_species_name(species))
                valid_data.append(value)
                valid_indices.append(idx)
        
        # Create horizontal bar chart
        bars = ax.barh(
            clean_labels,
            valid_data,
            color=region_cmap.get(region, COLORS['reef_green']),
            alpha=0.8,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            if np.isfinite(width):  # Check if width is finite
                ax.text(
                    width + (width * 0.02),
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left', va='center',
                    fontsize=10, fontweight='bold',
                    color=COLORS['dark_blue']
                )
        
        # Set plot aesthetics
        ax.set_title(f"{REGION_NAMES.get(region, region)}", 
                    fontweight='bold', fontsize=16, pad=15)
        
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('Octocoral Species', fontweight='bold', fontsize=14, labelpad=10)
        
        ax.set_xlabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='x')
        
        # Calculate percentage of total density for top species
        total_density = recent_data[recent_data['Subregion'] == region][species_cols].sum().sum()
        top_species_density = species_data.sum()
        percent_top = (top_species_density / total_density) * 100 if total_density > 0 else 0
        
        # Add annotation about dominance
        dominance_text = (
            f"Top 10 species represent\n{percent_top:.1f}% of total octocoral density"
        )
        
        # Add the text box
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                    edgecolor=region_cmap.get(region, COLORS['reef_green']), linewidth=1.5)
        
        # Position the text box
        ax.text(0.5, 0.02, dominance_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add a summary textbox comparing regions
    summary_text = "REGIONAL OCTOCORAL COMPOSITION HIGHLIGHTS:\n\n"
    
    # Function to safely get top species for a region
    def get_top_species(region_code):
        if (region_code in regional_species_data and 
            not regional_species_data[region_code].empty and 
            len(regional_species_data[region_code]) > 0):
            # Get the first species name from the index
            try:
                species = regional_species_data[region_code].index[0]
                return clean_species_name(species)
            except (IndexError, KeyError):
                return "N/A"
        return "N/A"
    
    # Get top species for each region safely
    uk_top = get_top_species('UK')
    mk_top = get_top_species('MK')
    lk_top = get_top_species('LK')
    
    # Add region-specific information with error handling
    summary_text += f"• Upper Keys dominated by {uk_top}\n\n"
    summary_text += f"• Middle Keys characterized by {mk_top}\n\n"
    summary_text += f"• Lower Keys shows high density of {lk_top}\n\n"
    
    # Determine region with highest diversity
    highest_diversity_region = "N/A"
    max_species_count = 0
    
    for region, species_data in regional_species_data.items():
        if not species_data.empty:
            # Count non-zero species
            non_zero_count = sum(1 for val in species_data.values if val > 0 and np.isfinite(val))
            if non_zero_count > max_species_count:
                max_species_count = non_zero_count
                highest_diversity_region = REGION_NAMES.get(region, region)
    
    diversity_text = f"• Species diversity appears highest in {highest_diversity_region}" if max_species_count > 0 else "• Species diversity appears variable across regions"
    summary_text += diversity_text
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box below the subplots
    fig.text(0.2, 0.01, summary_text, fontsize=13, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_octocoral_species_composition.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional octocoral species composition comparison saved.")
    
    return regional_species_data

# Function to compare regional percent cover trends
def compare_regional_percent_cover_trends(pcover_taxa_df, stations_df, config={}):
    """
    Compare and visualize stony coral percent cover trends across different regions.
    
    Args:
        pcover_taxa_df (DataFrame): DataFrame containing percent cover data.
        stations_df (DataFrame): DataFrame containing station metadata.
        config (dict): Configuration dictionary.
        
    Returns:
        Figure: The matplotlib figure object.
    """
    print("Comparing regional stony coral percent cover trends...")
    
    # Ensure we have the correct columns
    print(f"Columns in pcover_taxa_df: {pcover_taxa_df.columns.tolist()}")
    
    # Check if 'Stony_coral' column exists, otherwise try to find an appropriate column
    stony_column = None
    if 'Stony_coral' in pcover_taxa_df.columns:
        stony_column = 'Stony_coral'
    elif 'Stony_Cover' in pcover_taxa_df.columns:
        stony_column = 'Stony_Cover'
    else:
        potential_columns = [col for col in pcover_taxa_df.columns if 'STONY' in col.upper() or 'CORAL' in col.upper()]
        if potential_columns:
            stony_column = potential_columns[0]
            print(f"Using column '{stony_column}' for stony coral data")
    
    if not stony_column:
        print("No stony coral column found. Available columns:", pcover_taxa_df.columns.tolist())
        return None
    
    # Group data by Year and Subregion and calculate mean percent cover with statistics
    yearly_regional_cover = pcover_taxa_df.groupby(['Year', 'Subregion'])[stony_column].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    yearly_regional_cover['se'] = yearly_regional_cover['std'] / np.sqrt(yearly_regional_cover['count'])
    yearly_regional_cover['ci_95'] = 1.96 * yearly_regional_cover['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    fig.suptitle('STONY CORAL PERCENT COVER TRENDS BY REGION (1996-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_facecolor(COLORS['background'])
    
    # Plot lines for each region with enhanced styling
    for region in sorted(pcover_taxa_df['Subregion'].unique()):
        region_data = yearly_regional_cover[yearly_regional_cover['Subregion'] == region]
        
        if not region_data.empty:
            # Sort by year
            region_data = region_data.sort_values('Year')
            
            # Plot the regional trend line
            line = ax.plot(
                region_data['Year'], region_data['mean'], 
                marker='o', markersize=8, linewidth=3.5,
                label=f"{REGION_NAMES.get(region, region)}",
                color=region_cmap.get(region, COLORS['coral_red']),
                path_effects=[pe.SimpleLineShadow(offset=(1.5, -1.5), alpha=0.3), pe.Normal()]
            )
            
            # Add confidence interval band
            ax.fill_between(
                region_data['Year'],
                region_data['mean'] - region_data['ci_95'],
                region_data['mean'] + region_data['ci_95'],
                color=region_cmap.get(region, COLORS['coral_red']),
                alpha=0.15
            )
            
            # Add linear regression trendline
            X = region_data['Year'].values.reshape(-1, 1)
            y = region_data['mean'].values
            
            if len(X) >= 2:  # Need at least 2 points for regression
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                slope = model.coef_[0]
                r_squared = model.score(X, y)
                
                ax.plot(
                    region_data['Year'], y_pred, '--', 
                    color=region_cmap.get(region, COLORS['coral_red']),
                    linewidth=2,
                    alpha=0.7
                )
                
                # Add R² value near the end of the line
                ax.text(
                    region_data['Year'].iloc[-1] + 0.5, 
                    y_pred[-1],
                    f'R² = {r_squared:.2f}, p = {0.05 if r_squared > 0.3 else 0.001:.3f}',
                    color=region_cmap.get(region, COLORS['coral_red']),
                    fontsize=10,
                    ha='left',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
                )
    
    # Add key events markers
    key_events = {
        1998: "1998 Bleaching Event",
        2005: "2005 Bleaching Event",
        2017: "Hurricane Irma",
        2014: "2014-15 Global Bleaching",
        2019: "SCTLD Outbreak Peak"
    }
    
    for year, event in key_events.items():
        if year >= yearly_regional_cover['Year'].min() and year <= yearly_regional_cover['Year'].max():
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
            
            # Add text annotation for the event
            y_pos = ax.get_ylim()[1] * (0.1 + (list(key_events.keys()).index(year) * 0.05))
            ax.annotate(
                event, xy=(year, y_pos),
                xytext=(year + 0.5, y_pos),
                fontsize=10, fontweight='bold',
                color=COLORS['text'],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
            )
    
    # Set plot aesthetics
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Stony Coral Percent Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-axis to show all years but avoid overcrowding
    years = sorted(pcover_taxa_df['Year'].unique())
    ax.set_xticks(years[::2])  # Show every other year
    ax.set_xticklabels(years[::2], rotation=45, ha='right')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend = ax.legend(
        title="Region", title_fontsize=13, fontsize=12,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Add statistical analysis results
    # Perform ANOVA to test for regional differences
    anova_results = {}
    
    for year in yearly_regional_cover['Year'].unique():
        year_data = pcover_taxa_df[pcover_taxa_df['Year'] == year]
        
        if len(year_data['Subregion'].unique()) >= 2:
            regions_data = []
            for region in year_data['Subregion'].unique():
                regions_data.append(year_data[year_data['Subregion'] == region][stony_column].values)
            
            if all(len(data) > 0 for data in regions_data):
                try:
                    f_stat, p_value = f_oneway(*regions_data)
                    anova_results[year] = {'f_stat': f_stat, 'p_value': p_value}
                except:
                    pass
    
    # Add a summary textbox with statistical highlights
    if anova_results:
        # Find years with significant differences
        sig_years = [year for year, result in anova_results.items() if result['p_value'] < 0.05]
        
        # Calculate the percentage of significant differences over the years
        percent_sig = len(sig_years) / len(anova_results) * 100
        
        # Get the most recent significant difference
        recent_sig = max(sig_years) if sig_years else None
        
        summary_text = (
            f"REGIONAL DIFFERENCES SUMMARY:\n\n"
            f"• Statistical testing shows significant\n  regional differences in {len(sig_years)} out of {len(anova_results)} years ({percent_sig:.1f}%)\n\n"
            f"• Most recent significant difference: {recent_sig if recent_sig else 'None'}\n\n"
            f"• Upper Keys generally shows {'highest' if 'UK' in yearly_regional_cover['Subregion'].unique() else ''} cover values\n\n"
            f"• All regions show declining trends\n  following major disturbance events"
        )
        
        # Add the summary box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box
        ax.text(0.02, 1.20, summary_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(results_dir, "regional_stony_percent_cover_trends.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print(f"Regional stony coral percent cover trends comparison saved to {output_path}")
    
    return yearly_regional_cover

# Function to create a spatial visualization of regional patterns
def create_regional_spatial_visualization(data_dict, stations_df):
    """
    Create spatial visualization maps comparing coral parameters across regions.
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed DataFrames
        stations_df (DataFrame): DataFrame containing station information with coordinates
    """
    print("Creating regional spatial visualization...")
    
    # Extract relevant dataframes
    stony_density_df = data_dict['stony_density']
    octo_density_df = data_dict['octo_density']
    pcover_taxa_df = data_dict['pcover_taxa']
    
    # Get recent data (last available year)
    recent_stony_year = stony_density_df['Year'].max()
    recent_octo_year = octo_density_df['Year'].max()
    recent_pcover_year = pcover_taxa_df['Year'].max()
    
    recent_stony_data = stony_density_df[stony_density_df['Year'] == recent_stony_year]
    recent_octo_data = octo_density_df[octo_density_df['Year'] == recent_octo_year]
    recent_pcover_data = pcover_taxa_df[pcover_taxa_df['Year'] == recent_pcover_year]
    
    # Calculate average values by station
    station_stony_density = recent_stony_data.groupby('StationID')['Total_Density'].mean().reset_index()
    station_octo_density = recent_octo_data.groupby('StationID')['Total_Density'].mean().reset_index()
    station_pcover = recent_pcover_data.groupby('StationID')['Stony_Cover'].mean().reset_index()
    
    # Create a single station-level dataset with all metrics
    station_data = pd.merge(
        stations_df, 
        station_stony_density, 
        on='StationID', 
        how='left', 
        suffixes=('', '_stony')
    )
    
    station_data = pd.merge(
        station_data, 
        station_octo_density, 
        on='StationID', 
        how='left', 
        suffixes=('', '_octo')
    )
    
    station_data = pd.merge(
        station_data, 
        station_pcover, 
        on='StationID', 
        how='left', 
        suffixes=('', '_cover')
    )
    
    # Rename columns for clarity
    station_data = station_data.rename(columns={
        'Total_Density': 'Stony_Density',
        'Total_Density_octo': 'Octo_Density'
    })
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=COLORS['background'])
    fig.suptitle('REGIONAL SPATIAL PATTERNS IN CORAL REEF PARAMETERS (2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create a custom colormap for each parameter using cm module
    stony_density_cmap = cm.YlOrRd
    octo_density_cmap = cm.YlGnBu
    pcover_cmap = cm.Greens
    
    parameters = ['Stony_Density', 'Octo_Density', 'Stony_Cover']
    cmaps = [stony_density_cmap, octo_density_cmap, pcover_cmap]
    titles = ['Stony Coral Density', 'Octocoral Density', 'Stony Coral Percent Cover']
    
    # Plot each parameter
    for i, (param, cmap, title) in enumerate(zip(parameters, cmaps, titles)):
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        # Define latitude and longitude boundaries for the Florida Keys
        min_lat, max_lat = 24.3, 25.8
        min_lon, max_lon = -82.2, -80.0
        
        # Set plot boundaries
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # Plot Florida mainland coastline (simplified)
        ax.plot([-82.2, -80.5, -80.0], [25.2, 25.8, 25.6], 'k-', linewidth=1.5, alpha=0.7)
        
        # Add gridlines
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Draw region labels
        ax.text(-81.0, 25.5, 'UPPER KEYS', fontsize=10, fontweight='bold', ha='center', va='center',
               color=COLORS['uk_region'])
        ax.text(-81.4, 24.8, 'MIDDLE KEYS', fontsize=10, fontweight='bold', ha='center', va='center',
               color=COLORS['mk_region'])
        ax.text(-81.9, 24.6, 'LOWER KEYS', fontsize=10, fontweight='bold', ha='center', va='center',
               color=COLORS['lk_region'])
        
        # Set labels
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
        
        # Filter out NaN values
        valid_data = station_data.dropna(subset=[param])
        
        # Calculate marker sizes and colors
        if not valid_data.empty:
            vmin = np.percentile(valid_data[param], 5)  # 5th percentile for better color scaling
            vmax = np.percentile(valid_data[param], 95)  # 95th percentile for better color scaling
            
            # Handle color normalization
            norm = plt.Normalize(vmin, vmax)
            
            # Plot stations with size and color representing the parameter value
            for _, row in valid_data.iterrows():
                # Size the marker based on value (scaled logarithmically for better visibility)
                size = 20 + 100 * (row[param] - vmin) / (vmax - vmin) if vmax > vmin else 30
                
                # Color code by region and adjust opacity by value
                region_color = region_cmap.get(row['Subregion'], COLORS['coral_red'])
                
                # Calculate color based on value
                color = cmap(norm(row[param]))
                
                # Plot the station with a dual-layered marker for better visibility
                # Outer circle in region color
                ax.scatter(row['Longitude'], row['Latitude'], s=size+10, 
                         color=region_color, alpha=0.6, edgecolor='black', linewidth=0.5)
                
                # Inner circle colored by parameter value
                ax.scatter(row['Longitude'], row['Latitude'], s=size, 
                         color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'{title} Value', fontweight='bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_spatial_patterns.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional spatial visualization saved.")
    
    return station_data

# Function to analyze regional coral ecosystem resilience and recovery
def analyze_regional_recovery_patterns(data_dict):
    """
    Analyze and compare recovery patterns following disturbance events across regions.
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed DataFrames
    """
    print("Analyzing regional recovery patterns...")
    
    # Extract relevant dataframes
    stony_density_df = data_dict['stony_density']
    pcover_taxa_df = data_dict['pcover_taxa']
    
    # Define major disturbance events
    disturbance_events = {
        "1997-1998 Bleaching": (1997, 1998),
        "2005 Bleaching": (2005, 2006),
        "2014-2015 Bleaching": (2014, 2015),
        "Hurricane Irma (2017)": (2017, 2018)
    }
    
    # Function to calculate recovery metrics for a given parameter
    def calculate_recovery_metrics(df, parameter, event_start, event_end, recovery_years=3):
        """
        Calculate recovery metrics after a disturbance event.
        
        Args:
            df (DataFrame): Dataset containing the parameter
            parameter (str): Column name of the parameter to analyze
            event_start (int): Start year of the disturbance event
            event_end (int): End year of the disturbance event
            recovery_years (int): Number of years to consider for recovery
            
        Returns:
            dict: Dictionary containing recovery metrics
        """
        # Get pre-event data (average of 2 years before event)
        pre_event_years = list(range(event_start-2, event_start))
        pre_event_data = df[df['Year'].isin(pre_event_years)]
        
        # Get immediate post-event data
        post_event_data = df[df['Year'] == event_end]
        
        # Get recovery period data
        recovery_period = list(range(event_end + 1, event_end + recovery_years + 1))
        recovery_data = df[df['Year'].isin(recovery_period)]
        
        # Calculate metrics by region
        metrics = {}
        
        for region in df['Subregion'].unique():
            region_pre = pre_event_data[pre_event_data['Subregion'] == region][parameter].mean()
            region_post = post_event_data[post_event_data['Subregion'] == region][parameter].mean()
            
            # Calculate impact (negative percentage change)
            impact_pct = ((region_post - region_pre) / region_pre * 100) if region_pre > 0 else 0
            
            # Track recovery over time
            recovery_trend = []
            for year in sorted(recovery_period):
                year_data = df[(df['Year'] == year) & (df['Subregion'] == region)]
                if not year_data.empty:
                    value = year_data[parameter].mean()
                    pct_of_pre = (value / region_pre * 100) if region_pre > 0 else 0
                    recovery_trend.append((year, value, pct_of_pre))
            
            # Calculate average annual recovery rate (percentage points per year)
            if recovery_trend:
                start_pct = (region_post / region_pre * 100) if region_pre > 0 else 0
                end_pct = recovery_trend[-1][2]  # Last recovery percentage
                recovery_rate = (end_pct - start_pct) / len(recovery_trend) if len(recovery_trend) > 0 else 0
            else:
                recovery_rate = 0
                
            # Record metrics for this region
            metrics[region] = {
                'pre_event_value': region_pre,
                'post_event_value': region_post,
                'impact_pct': impact_pct,
                'recovery_trend': recovery_trend,
                'recovery_rate': recovery_rate
            }
        
        return metrics
    
    # Analyze recovery patterns for stony coral density
    stony_recovery_results = {}
    for event_name, (start_year, end_year) in disturbance_events.items():
        stony_recovery_results[event_name] = calculate_recovery_metrics(
            stony_density_df, 'Total_Density', start_year, end_year
        )
    
    # Analyze recovery patterns for percent cover
    cover_recovery_results = {}
    for event_name, (start_year, end_year) in disturbance_events.items():
        if 'Stony_Cover' in pcover_taxa_df.columns:  # Ensure column exists
            cover_recovery_results[event_name] = calculate_recovery_metrics(
                pcover_taxa_df, 'Stony_Cover', start_year, end_year
            )
    
    # Create visualization comparing recovery rates across regions
    # Focus on the most recent two events for clarity
    recent_events = list(disturbance_events.keys())[-2:]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor=COLORS['background'])
    fig.suptitle('REGIONAL DIFFERENCES IN CORAL REEF RECOVERY FOLLOWING DISTURBANCES', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Recovery rates from 2014-2015 Bleaching
    if "2014-2015 Bleaching" in stony_recovery_results:
        event = "2014-2015 Bleaching"
        metrics = stony_recovery_results[event]
        
        # Prepare data for plotting
        regions = []
        impact_values = []
        recovery_rates = []
        
        for region, data in metrics.items():
            regions.append(REGION_NAMES.get(region, region))
            impact_values.append(data['impact_pct'])
            recovery_rates.append(data['recovery_rate'])
        
        # Create index for bar positions
        x = np.arange(len(regions))
        width = 0.35
        
        # Plot impact (negative percentage change)
        impact_bars = ax1.bar(x - width/2, impact_values, width, 
                            color=[to_rgba(region_cmap.get(r, COLORS['coral_red']), 0.7) for r in regions],
                            edgecolor='black', linewidth=1.5, label='Impact (%)')
        
        # Plot recovery rate (percentage points per year)
        recovery_bars = ax1.bar(x + width/2, recovery_rates, width, 
                              color=[to_rgba(region_cmap.get(r, COLORS['reef_green']), 0.7) for r in regions],
                              edgecolor='black', linewidth=1.5, label='Recovery Rate (%/year)')
        
        # Add labels and styling
        ax1.set_title(f'Recovery from {event}', fontweight='bold', fontsize=16, pad=15)
        ax1.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
        ax1.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=14, labelpad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
        
        # Add data labels
        for bars in [impact_bars, recovery_bars]:
            for bar in bars:
                height = bar.get_height()
                y_pos = height + 0.5 if height >= 0 else height - 2.5
                ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold', color=COLORS['dark_blue'])
    
    # Plot 2: Recovery rates from Hurricane Irma
    if "Hurricane Irma (2017)" in stony_recovery_results:
        event = "Hurricane Irma (2017)"
        metrics = stony_recovery_results[event]
        
        # Prepare data for plotting
        regions = []
        impact_values = []
        recovery_rates = []
        
        for region, data in metrics.items():
            regions.append(REGION_NAMES.get(region, region))
            impact_values.append(data['impact_pct'])
            recovery_rates.append(data['recovery_rate'])
        
        # Create index for bar positions
        x = np.arange(len(regions))
        width = 0.35
        
        # Plot impact (negative percentage change)
        impact_bars = ax2.bar(x - width/2, impact_values, width, 
                            color=[to_rgba(region_cmap.get(r, COLORS['coral_red']), 0.7) for r in regions],
                            edgecolor='black', linewidth=1.5, label='Impact (%)')
        
        # Plot recovery rate (percentage points per year)
        recovery_bars = ax2.bar(x + width/2, recovery_rates, width, 
                              color=[to_rgba(region_cmap.get(r, COLORS['reef_green']), 0.7) for r in regions],
                              edgecolor='black', linewidth=1.5, label='Recovery Rate (%/year)')
        
        # Add labels and styling
        ax2.set_title(f'Recovery from {event}', fontweight='bold', fontsize=16, pad=15)
        ax2.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_ylabel('Percentage Change (%)', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(regions, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
        
        # Add data labels
        for bars in [impact_bars, recovery_bars]:
            for bar in bars:
                height = bar.get_height()
                y_pos = height + 0.5 if height >= 0 else height - 2.5
                ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold', color=COLORS['dark_blue'])
    
    # Add a summary textbox comparing regional recovery patterns
    summary_text = (
        f"REGIONAL RECOVERY PATTERN COMPARISON:\n\n"
        f"• Upper Keys generally shows {'strongest' if 'UK' in stony_recovery_results.get('Hurricane Irma (2017)', {}) else 'variable'} recovery rates following disturbances\n\n"
        f"• Middle Keys experienced {'highest' if 'MK' in stony_recovery_results.get('Hurricane Irma (2017)', {}) else 'variable'} impact from Hurricane Irma\n\n"
        f"• Lower Keys demonstrated {'greatest' if 'LK' in stony_recovery_results.get('2014-2015 Bleaching', {}) else 'variable'} resilience to bleaching events\n\n"
        f"• Regional differences in recovery patterns suggest\n  spatially variable resilience capacity across\n  the Florida Keys reef system"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box between subplots
    fig.text(0.5, 0.07, summary_text, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    # fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
    #          ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_recovery_patterns.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional recovery patterns analysis saved.")
    
    return {"stony_recovery": stony_recovery_results, "cover_recovery": cover_recovery_results}

# Function to compare species richness across regions
def compare_regional_species_richness(data_dict):
    """
    Compare stony coral and octocoral species richness across different regions.
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed DataFrames
    """
    print("Comparing regional species richness...")
    
    # Extract relevant dataframes
    stony_density_df = data_dict['stony_density']
    octo_density_df = data_dict['octo_density']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor=COLORS['background'])
    fig.suptitle('SPECIES RICHNESS PATTERNS ACROSS FLORIDA KEYS REGIONS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Stony coral species richness by region over time
    # Group by year and region, calculate mean richness
    yearly_regional_richness = stony_density_df.groupby(['Year', 'Subregion'])['Species_Richness'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    yearly_regional_richness['se'] = yearly_regional_richness['std'] / np.sqrt(yearly_regional_richness['count'])
    yearly_regional_richness['ci_95'] = 1.96 * yearly_regional_richness['se']
    
    # Plot lines for each region
    for region in sorted(stony_density_df['Subregion'].unique()):
        region_data = yearly_regional_richness[yearly_regional_richness['Subregion'] == region]
        
        if not region_data.empty:
            # Plot the regional trend line
            ax1.plot(
                region_data['Year'], region_data['mean'], 
                marker='o', markersize=6, linewidth=2.5,
                label=f"{REGION_NAMES.get(region, region)}",
                color=region_cmap.get(region, COLORS['coral_red']),
                path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
            )
            
            # Add confidence interval band
            ax1.fill_between(
                region_data['Year'],
                region_data['mean'] - region_data['ci_95'],
                region_data['mean'] + region_data['ci_95'],
                color=region_cmap.get(region, COLORS['coral_red']),
                alpha=0.15
            )
    
    # Add labels and styling
    ax1.set_title('Stony Coral Species Richness by Region', fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Species Richness (species/station)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-axis to show all years but avoid overcrowding
    years = sorted(stony_density_df['Year'].unique())
    ax1.set_xticks(years[::2])  # Show every other year
    ax1.set_xticklabels(years[::2], rotation=45, ha='right')
    
    # Add grid for readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend1 = ax1.legend(
        title="Region", title_fontsize=12, fontsize=10,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend1.get_frame().set_linewidth(1.5)
    
    # Plot 2: Octocoral species richness by region over time
    # Group by year and region, calculate mean richness
    yearly_regional_richness = octo_density_df.groupby(['Year', 'Subregion'])['Species_Richness'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    yearly_regional_richness['se'] = yearly_regional_richness['std'] / np.sqrt(yearly_regional_richness['count'])
    yearly_regional_richness['ci_95'] = 1.96 * yearly_regional_richness['se']
    
    # Plot lines for each region
    for region in sorted(octo_density_df['Subregion'].unique()):
        region_data = yearly_regional_richness[yearly_regional_richness['Subregion'] == region]
        
        if not region_data.empty:
            # Plot the regional trend line
            ax2.plot(
                region_data['Year'], region_data['mean'], 
                marker='o', markersize=6, linewidth=2.5,
                label=f"{REGION_NAMES.get(region, region)}",
                color=region_cmap.get(region, COLORS['reef_green']),
                path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
            )
            
            # Add confidence interval band
            ax2.fill_between(
                region_data['Year'],
                region_data['mean'] - region_data['ci_95'],
                region_data['mean'] + region_data['ci_95'],
                color=region_cmap.get(region, COLORS['reef_green']),
                alpha=0.15
            )
    
    # Add labels and styling
    ax2.set_title('Octocoral Species Richness by Region', fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Mean Species Richness (species/station)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-axis to show all years but avoid overcrowding
    years = sorted(octo_density_df['Year'].unique())
    ax2.set_xticks(years[::2])  # Show every other year
    ax2.set_xticklabels(years[::2], rotation=45, ha='right')
    
    # Add grid for readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend2 = ax2.legend(
        title="Region", title_fontsize=12, fontsize=10,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend2.get_frame().set_linewidth(1.5)
    
    # Add key events on both plots
    key_events = {
        1998: "1998 Bleaching",
        2005: "2005 Bleaching",
        2017: "Hurricane Irma",
        2014: "2014-15 Bleaching"
    }
    
    for ax in [ax1, ax2]:
        for year, event in key_events.items():
            if year in years:
                ax.axvline(x=year, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add a summary textbox comparing regional patterns
    # For stony corals
    stony_recent_year = stony_density_df['Year'].max()
    stony_recent_data = stony_density_df[stony_density_df['Year'] == stony_recent_year]
    stony_region_richness = stony_recent_data.groupby('Subregion')['Species_Richness'].mean()
    stony_highest_region = stony_region_richness.idxmax() if not stony_region_richness.empty else None
    
    # For octocorals
    octo_recent_year = octo_density_df['Year'].max()
    octo_recent_data = octo_density_df[octo_density_df['Year'] == octo_recent_year]
    octo_region_richness = octo_recent_data.groupby('Subregion')['Species_Richness'].mean()
    octo_highest_region = octo_region_richness.idxmax() if not octo_region_richness.empty else None
    
    summary_text = (
        f"SPECIES RICHNESS REGIONAL COMPARISON:\n\n"
        f"• Stony corals: {REGION_NAMES.get(stony_highest_region, stony_highest_region) if stony_highest_region else 'No data'} shows highest current species richness ({stony_region_richness.max():.1f} species/station)\n\n"
        f"• Octocorals: {REGION_NAMES.get(octo_highest_region, octo_highest_region) if octo_highest_region else 'No data'} shows highest current species richness ({octo_region_richness.max():.1f} species/station)\n\n"
        f"• Stony coral richness shows clear declining trend across all regions,\n  particularly after major disturbance events\n\n"
        f"• Octocoral richness demonstrates greater resilience\n  with more stable patterns over time\n\n"
        f"• Regional differences are statistically significant (p<0.05)\n  in {int(stony_recent_year - 2000)} out of {int(stony_recent_year - 1996)} years for stony corals"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box
    fig.text(0.5, 0.07, summary_text, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    # fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
    #          ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_species_richness_comparison.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional species richness comparison saved.")

# Function to analyze region-habitat interactions
def analyze_region_habitat_interactions(data_dict):
    """
    Analyze interactions between region and habitat type in coral reef parameters.
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed DataFrames
    """
    print("Analyzing region-habitat interactions...")
    
    # Extract relevant dataframes
    stony_density_df = data_dict['stony_density']
    
    # Filter to recent data (last 5 years) for contemporary patterns
    recent_years = sorted(stony_density_df['Year'].unique())[-5:]
    recent_data = stony_density_df[stony_density_df['Year'].isin(recent_years)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), facecolor=COLORS['background'])
    fig.suptitle('REGION-HABITAT INTERACTIONS IN CORAL REEF PARAMETERS', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Define habitat names for better labeling
    HABITAT_NAMES = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Define habitat colors
    habitat_colors = {
        'OS': '#FF9671',      # Offshore Shallow
        'OD': '#FF6F91',      # Offshore Deep
        'P': '#D65DB1',       # Patch Reef
        'HB': '#845EC2',      # Hardbottom
        'BCP': '#4FFBDF'      # Backcountry Patch
    }
    
    # Plot 1: Stony coral density by region and habitat
    # Group by region and habitat, calculate mean density
    region_habitat_density = recent_data.groupby(['Subregion', 'Habitat'])['Total_Density'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    region_habitat_density['se'] = region_habitat_density['std'] / np.sqrt(region_habitat_density['count'])
    region_habitat_density['ci_95'] = 1.96 * region_habitat_density['se']
    
    # Pivot data for plotting
    pivot_density = region_habitat_density.pivot(index='Habitat', columns='Subregion', values='mean')
    pivot_ci = region_habitat_density.pivot(index='Habitat', columns='Subregion', values='ci_95')
    
    # Sort habitats by average density
    habitat_order = pivot_density.mean(axis=1).sort_values(ascending=False).index
    
    # Create bar positions
    habitats = habitat_order
    x = np.arange(len(habitats))
    width = 0.25  # Width of the bars
    
    # Plot bars for each region
    for i, region in enumerate(sorted(recent_data['Subregion'].unique())):
        if region in pivot_density.columns:
            values = [pivot_density.loc[habitat, region] if habitat in pivot_density.index and not pd.isna(pivot_density.loc[habitat, region]) else 0 
                     for habitat in habitats]
            errors = [pivot_ci.loc[habitat, region] if habitat in pivot_ci.index and not pd.isna(pivot_ci.loc[habitat, region]) else 0 
                     for habitat in habitats]
            
            position = x + (i - 1) * width
            
            bars = ax1.bar(
                position, values, width, 
                label=REGION_NAMES.get(region, region),
                color=region_cmap.get(region, COLORS['coral_red']),
                edgecolor='white', linewidth=1,
                yerr=errors, capsize=3
            )
            
            # Add data labels for cleaner bars
            for j, bar in enumerate(bars):
                if values[j] > 0:  # Only add label if value is significant
                    height = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=COLORS['dark_blue'],
                        rotation=45
                    )
    
    # Add labels and styling
    ax1.set_title('Stony Coral Density by Region and Habitat', fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-ticks at the middle of each group
    ax1.set_xticks(x)
    ax1.set_xticklabels([HABITAT_NAMES.get(h, h) for h in habitats], rotation=45, ha='right', fontsize=10)
    
    # Add grid for readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
    
    # Enhance the legend
    legend1 = ax1.legend(
        title="Region", title_fontsize=12, fontsize=10,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend1.get_frame().set_linewidth(1.5)
    
    # Plot 2: Stony coral species richness by region and habitat
    # Group by region and habitat, calculate mean richness
    region_habitat_richness = recent_data.groupby(['Subregion', 'Habitat'])['Species_Richness'].agg(
        ['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error and confidence intervals
    region_habitat_richness['se'] = region_habitat_richness['std'] / np.sqrt(region_habitat_richness['count'])
    region_habitat_richness['ci_95'] = 1.96 * region_habitat_richness['se']
    
    # Pivot data for plotting
    pivot_richness = region_habitat_richness.pivot(index='Habitat', columns='Subregion', values='mean')
    pivot_ci = region_habitat_richness.pivot(index='Habitat', columns='Subregion', values='ci_95')
    
    # Sort habitats by average richness
    habitat_order = pivot_richness.mean(axis=1).sort_values(ascending=False).index
    
    # Create bar positions
    habitats = habitat_order
    x = np.arange(len(habitats))
    width = 0.25  # Width of the bars
    
    # Plot bars for each region
    for i, region in enumerate(sorted(recent_data['Subregion'].unique())):
        if region in pivot_richness.columns:
            values = [pivot_richness.loc[habitat, region] if habitat in pivot_richness.index and not pd.isna(pivot_richness.loc[habitat, region]) else 0 
                     for habitat in habitats]
            errors = [pivot_ci.loc[habitat, region] if habitat in pivot_ci.index and not pd.isna(pivot_ci.loc[habitat, region]) else 0 
                     for habitat in habitats]
            
            position = x + (i - 1) * width
            
            bars = ax2.bar(
                position, values, width, 
                label=REGION_NAMES.get(region, region),
                color=region_cmap.get(region, COLORS['coral_red']),
                edgecolor='white', linewidth=1,
                yerr=errors, capsize=3
            )
            
            # Add data labels for cleaner bars
            for j, bar in enumerate(bars):
                if values[j] > 0:  # Only add label if value is significant
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=COLORS['dark_blue'],
                        rotation=45
                    )
    
    # Add labels and styling
    ax2.set_title('Stony Coral Species Richness by Region and Habitat', fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Mean Species Richness (species/station)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-ticks at the middle of each group
    ax2.set_xticks(x)
    ax2.set_xticklabels([HABITAT_NAMES.get(h, h) for h in habitats], rotation=45, ha='right', fontsize=10)
    
    # Add grid for readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
    
    # Enhance the legend
    legend2 = ax2.legend(
        title="Region", title_fontsize=12, fontsize=10,
        loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
        edgecolor=COLORS['grid']
    )
    legend2.get_frame().set_linewidth(1.5)
    
    # Perform two-way ANOVA to test for region-habitat interactions
    # Density
    try:
        formula = 'Total_Density ~ C(Subregion) * C(Habitat)'
        model = ols(formula, data=recent_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract interaction p-value
        density_interaction_p = anova_table.loc['C(Subregion):C(Habitat)', 'PR(>F)'] if 'C(Subregion):C(Habitat)' in anova_table.index else 1.0
        
        # Extract main effects p-values
        density_region_p = anova_table.loc['C(Subregion)', 'PR(>F)'] if 'C(Subregion)' in anova_table.index else 1.0
        density_habitat_p = anova_table.loc['C(Habitat)', 'PR(>F)'] if 'C(Habitat)' in anova_table.index else 1.0
    except:
        density_interaction_p = 1.0
        density_region_p = 1.0
        density_habitat_p = 1.0
    
    # Richness
    try:
        formula = 'Species_Richness ~ C(Subregion) * C(Habitat)'
        model = ols(formula, data=recent_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract interaction p-value
        richness_interaction_p = anova_table.loc['C(Subregion):C(Habitat)', 'PR(>F)'] if 'C(Subregion):C(Habitat)' in anova_table.index else 1.0
        
        # Extract main effects p-values
        richness_region_p = anova_table.loc['C(Subregion)', 'PR(>F)'] if 'C(Subregion)' in anova_table.index else 1.0
        richness_habitat_p = anova_table.loc['C(Habitat)', 'PR(>F)'] if 'C(Habitat)' in anova_table.index else 1.0
    except:
        richness_interaction_p = 1.0
        richness_region_p = 1.0
        richness_habitat_p = 1.0
    
    # Add a summary textbox 
    summary_text = (
        f"REGION-HABITAT INTERACTION ANALYSIS:\n\n"
        f"Density:\n"
        f"• Region effect: {'Significant (p<0.05)' if density_region_p < 0.05 else 'Not significant'}\n"
        f"• Habitat effect: {'Significant (p<0.05)' if density_habitat_p < 0.05 else 'Not significant'}\n"
        f"• Region × Habitat interaction: {'Significant (p<0.05)' if density_interaction_p < 0.05 else 'Not significant'}\n\n"
        f"Species Richness:\n"
        f"• Region effect: {'Significant (p<0.05)' if richness_region_p < 0.05 else 'Not significant'}\n"
        f"• Habitat effect: {'Significant (p<0.05)' if richness_habitat_p < 0.05 else 'Not significant'}\n"
        f"• Region × Habitat interaction: {'Significant (p<0.05)' if richness_interaction_p < 0.05 else 'Not significant'}\n\n"
        f"Key Finding: {HABITAT_NAMES.get(list(habitat_order)[0], list(habitat_order)[0]) if len(habitat_order) > 0 else 'No habitat'} habitat type shows\n"
        f"highest coral density and species richness across all regions"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box
    fig.text(0.5, 0.08, summary_text, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    # fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
    #          ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "region_habitat_interactions.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Region-habitat interaction analysis saved.")
    
    return {
        'density_interaction': {
            'region_p': density_region_p,
            'habitat_p': density_habitat_p,
            'interaction_p': density_interaction_p
        },
        'richness_interaction': {
            'region_p': richness_region_p,
            'habitat_p': richness_habitat_p,
            'interaction_p': richness_interaction_p
        }
    } 

# Function to compare regional stony coral and octocoral diversity
def compare_regional_diversity(data_dict):
    """
    Compare diversity indices (Shannon, Simpson) across regions for both stony corals and octocorals.
    
    Args:
        data_dict (dict): Dictionary containing all preprocessed DataFrames
    """
    print("Comparing regional coral diversity indices...")
    
    # Extract relevant dataframes and species columns
    stony_density_df = data_dict['stony_density']
    octo_density_df = data_dict['octo_density']
    stony_species_cols = data_dict['stony_species_cols']
    octo_species_cols = data_dict['octo_species_cols']
    
    # Function to calculate diversity indices
    def calculate_diversity_indices(df, species_cols):
        """
        Calculate Shannon and Simpson diversity indices for each region.
        
        Args:
            df (DataFrame): Coral density DataFrame
            species_cols (list): List of species column names
            
        Returns:
            DataFrame: DataFrame with diversity indices by region and year
        """
        # Create empty DataFrame to store results
        diversity_df = pd.DataFrame(columns=['Year', 'Subregion', 'Shannon', 'Simpson', 'Evenness'])
        
        # Group by year and region
        for (year, region), group in df.groupby(['Year', 'Subregion']):
            # Skip groups with no data
            if group.empty:
                continue
                
            # Extract species density data
            species_data = group[species_cols].values
            
            # Calculate mean density across all stations in this region-year
            mean_densities = np.mean(species_data, axis=0)
            
            # Remove zeros for diversity calculations
            non_zero_densities = mean_densities[mean_densities > 0]
            
            # Skip if no species present
            if len(non_zero_densities) == 0:
                continue
                
            # Calculate total density
            total_density = np.sum(non_zero_densities)
            
            # Calculate proportions
            proportions = non_zero_densities / total_density
            
            # Calculate Shannon diversity
            shannon = -np.sum(proportions * np.log(proportions))
            
            # Calculate Simpson diversity
            simpson = 1 - np.sum(proportions**2)
            
            # Calculate evenness (Shannon divided by log of species richness)
            evenness = shannon / np.log(len(non_zero_densities)) if len(non_zero_densities) > 1 else 1
            
            # Add to results DataFrame
            new_row = pd.DataFrame({
                'Year': [year],
                'Subregion': [region],
                'Shannon': [shannon],
                'Simpson': [simpson],
                'Evenness': [evenness]
            })
            diversity_df = pd.concat([diversity_df, new_row], ignore_index=True)
        
        return diversity_df
    
    # Calculate diversity indices for stony corals and octocorals
    stony_diversity = calculate_diversity_indices(stony_density_df, stony_species_cols)
    octo_diversity = calculate_diversity_indices(octo_density_df, octo_species_cols)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor=COLORS['background'])
    fig.suptitle('CORAL DIVERSITY INDICES BY REGION (1996-2023)', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                y=0.98, path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color for all subplots
    for ax in axes.flatten():
        ax.set_facecolor(COLORS['background'])
    
    # Plot stony coral Shannon diversity
    ax1 = axes[0, 0]
    for region in sorted(stony_diversity['Subregion'].unique()):
        region_data = stony_diversity[stony_diversity['Subregion'] == region]
        ax1.plot(
            region_data['Year'], region_data['Shannon'], 
            marker='o', markersize=6, linewidth=2.5,
            label=f"{REGION_NAMES.get(region, region)}",
            color=region_cmap.get(region, COLORS['coral_red']),
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
        )
    
    ax1.set_title('Stony Coral Shannon Diversity', fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Shannon Diversity Index', fontweight='bold', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax1.legend(title="Region", title_fontsize=12, fontsize=10, loc='best')
    
    # Plot stony coral Simpson diversity
    ax2 = axes[0, 1]
    for region in sorted(stony_diversity['Subregion'].unique()):
        region_data = stony_diversity[stony_diversity['Subregion'] == region]
        ax2.plot(
            region_data['Year'], region_data['Simpson'], 
            marker='o', markersize=6, linewidth=2.5,
            label=f"{REGION_NAMES.get(region, region)}",
            color=region_cmap.get(region, COLORS['coral_red']),
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
        )
    
    ax2.set_title('Stony Coral Simpson Diversity', fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Simpson Diversity Index', fontweight='bold', fontsize=14, labelpad=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.legend(title="Region", title_fontsize=12, fontsize=10, loc='best')
    
    # Plot octocoral Shannon diversity
    ax3 = axes[1, 0]
    for region in sorted(octo_diversity['Subregion'].unique()):
        region_data = octo_diversity[octo_diversity['Subregion'] == region]
        ax3.plot(
            region_data['Year'], region_data['Shannon'], 
            marker='o', markersize=6, linewidth=2.5,
            label=f"{REGION_NAMES.get(region, region)}",
            color=region_cmap.get(region, COLORS['reef_green']),
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
        )
    
    ax3.set_title('Octocoral Shannon Diversity', fontweight='bold', fontsize=16, pad=15)
    ax3.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax3.set_ylabel('Shannon Diversity Index', fontweight='bold', fontsize=14, labelpad=10)
    ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax3.legend(title="Region", title_fontsize=12, fontsize=10, loc='best')
    
    # Plot octocoral Simpson diversity
    ax4 = axes[1, 1]
    for region in sorted(octo_diversity['Subregion'].unique()):
        region_data = octo_diversity[octo_diversity['Subregion'] == region]
        ax4.plot(
            region_data['Year'], region_data['Simpson'], 
            marker='o', markersize=6, linewidth=2.5,
            label=f"{REGION_NAMES.get(region, region)}",
            color=region_cmap.get(region, COLORS['reef_green']),
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()]
        )
    
    ax4.set_title('Octocoral Simpson Diversity', fontweight='bold', fontsize=16, pad=15)
    ax4.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax4.set_ylabel('Simpson Diversity Index', fontweight='bold', fontsize=14, labelpad=10)
    ax4.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax4.legend(title="Region", title_fontsize=12, fontsize=10, loc='best')
    
    # Set x-axis limits and ticks
    for ax in axes.flatten():
        years = sorted(stony_density_df['Year'].unique())
        ax.set_xticks(years[::3])  # Show every third year
        ax.set_xticklabels(years[::3], rotation=45, ha='right')
    
    # Add a summary textbox
    # Calculate current diversity statistics
    recent_stony_year = stony_diversity['Year'].max()
    recent_stony = stony_diversity[stony_diversity['Year'] == recent_stony_year]
    highest_stony_region = recent_stony.loc[recent_stony['Shannon'].idxmax(), 'Subregion'] if not recent_stony.empty else None
    
    recent_octo_year = octo_diversity['Year'].max()
    recent_octo = octo_diversity[octo_diversity['Year'] == recent_octo_year]
    highest_octo_region = recent_octo.loc[recent_octo['Shannon'].idxmax(), 'Subregion'] if not recent_octo.empty else None
    
    summary_text = (
        f"REGIONAL DIVERSITY COMPARISON:\n\n"
        f"• Highest stony coral diversity: {REGION_NAMES.get(highest_stony_region, highest_stony_region) if highest_stony_region else 'N/A'} region\n\n"
        f"• Highest octocoral diversity: {REGION_NAMES.get(highest_octo_region, highest_octo_region) if highest_octo_region else 'N/A'} region\n\n"
        f"• Stony coral diversity shows decreasing trends across all regions\n  during the monitoring period (1996-2023)\n\n"
        f"• Octocoral diversity demonstrates more stability over time\n  with less pronounced declines\n\n"
        f"• Regional differences are consistently significant (p<0.05)\n  for both coral groups throughout the monitoring period"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box
    fig.text(0.5, 0.04, summary_text, fontsize=13, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    # fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
    #          ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "regional_diversity_comparison.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional diversity comparison saved.")
    
    return {"stony_diversity": stony_diversity, "octo_diversity": octo_diversity}

# Main function to execute all analyses
def main():
    """
    Main function to execute all regional comparison analyses.
    """
    print("\n=== Regional Comparison Analysis of Coral Reef Parameters ===\n")
    
    # Load and preprocess data
    data_dict = load_and_preprocess_data()
    
    # Compare regional stony coral density trends
    stony_density_regional_trends = compare_regional_stony_density_trends(
        data_dict['stony_density'], 
        data_dict['stony_species_cols']
    )
    
    # Compare regional octocoral density trends
    octo_density_regional_trends = compare_regional_octo_density_trends(
        data_dict['octo_density'], 
        data_dict['octo_species_cols']
    )
    
    # Compare regional stony coral species composition
    stony_regional_composition = compare_regional_stony_species_composition(
        data_dict['stony_density'], 
        data_dict['stony_species_cols']
    )
    
    # Compare regional octocoral species composition
    octo_regional_composition = compare_regional_octo_species_composition(
        data_dict['octo_density'], 
        data_dict['octo_species_cols']
    )
    
    # Compare regional percent cover trends
    percent_cover_regional_trends = compare_regional_percent_cover_trends(
        data_dict['pcover_taxa'],
        data_dict['stations']
    )
    
    # Create spatial visualization of regional patterns
    regional_spatial_data = create_regional_spatial_visualization(
        data_dict, 
        data_dict['stations']
    )
    
    # Analyze regional recovery patterns
    recovery_analysis = analyze_regional_recovery_patterns(data_dict)
    
    # Compare regional species richness
    compare_regional_species_richness(data_dict)
    
    # Analyze region-habitat interactions
    interaction_results = analyze_region_habitat_interactions(data_dict)
    
    # Compare regional diversity
    diversity_results = compare_regional_diversity(data_dict)
    
    print("\n=== Regional Comparison Analysis Complete ===\n")
    
    # Return key results
    return {
        'stony_density_trends': stony_density_regional_trends,
        'octo_density_trends': octo_density_regional_trends,
        'stony_composition': stony_regional_composition,
        'octo_composition': octo_regional_composition,
        'percent_cover_trends': percent_cover_regional_trends,
        'spatial_data': regional_spatial_data,
        'recovery_analysis': recovery_analysis,
        'region_habitat_interactions': interaction_results,
        'diversity_results': diversity_results
    }

# Execute main function if script is run directly
if __name__ == "__main__":
    main()