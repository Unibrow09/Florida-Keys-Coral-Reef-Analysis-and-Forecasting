"""
06_stony_coral_density_richness_relationship.py - Analysis of the relationship between stony coral density and species richness

This script analyzes the relationship between stony coral density and species richness within monitoring sites
in the Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP). It explores correlations, patterns,
and variations across different sites, regions, habitats, and over time.

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
results_dir = "06_Results"
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
    Load and preprocess the datasets for stony coral density and species richness analysis.
    
    Returns:
        tuple: (density_df, species_df, stations_df) - Preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Stony Coral Density dataset
        density_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_Density.csv")
        print(f"Stony coral density data loaded successfully with {len(density_df)} rows")
        
        # Load the Stony Coral Species Cover dataset
        species_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
        print(f"Stony coral species cover data loaded successfully with {len(species_df)} rows")
        
        # Load the Stations dataset
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Convert date columns to datetime format where applicable
        if 'Date' in density_df.columns:
            density_df['Date'] = pd.to_datetime(density_df['Date'])
            
        if 'Date' in species_df.columns:
            species_df['Date'] = pd.to_datetime(species_df['Date'])
        
        # Convert 'Year' column to integer for easier filtering
        if 'Year' in density_df.columns:
            density_df['Year'] = density_df['Year'].astype(int)
            
        if 'Year' in species_df.columns:
            species_df['Year'] = species_df['Year'].astype(int)
        
        # Identify metadata columns
        density_meta_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
        species_meta_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
        
        # Identify species columns (all non-metadata columns)
        density_species_cols = [col for col in density_df.columns if col not in density_meta_cols]
        species_species_cols = [col for col in species_df.columns if col not in species_meta_cols]
        
        # For density data
        # Convert species columns to numeric, coercing errors to NaN
        for col in density_species_cols:
            density_df[col] = pd.to_numeric(density_df[col], errors='coerce')
            
        # Fill NaN values with 0
        density_df[density_species_cols] = density_df[density_species_cols].fillna(0)
        
        # Calculate total density and species richness at station level
        density_df['Total_Density'] = density_df[density_species_cols].sum(axis=1)
        
        # Calculate species richness (number of species present at each station)
        density_df['Species_Richness'] = (density_df[density_species_cols] > 0).sum(axis=1)
        
        print(f"\nProcessed density data: {len(density_df)} records")
        print(f"Identified {len(density_species_cols)} coral species in the density dataset")
        
        # For species cover data
        # Convert species columns to numeric, coercing errors to NaN
        for col in species_species_cols:
            species_df[col] = pd.to_numeric(species_df[col], errors='coerce')
            
        # Fill NaN values with 0
        species_df[species_species_cols] = species_df[species_species_cols].fillna(0)
        
        # Calculate total cover and species richness if needed
        if len(species_species_cols) > 0:
            species_df['Total_Cover'] = species_df[species_species_cols].sum(axis=1)
            species_df['Species_Richness'] = (species_df[species_species_cols] > 0).sum(axis=1)
            print(f"Identified {len(species_species_cols)} coral species in the cover dataset")
        
        return density_df, species_df, stations_df, density_species_cols, species_species_cols
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def analyze_overall_relationship(df):
    """
    Analyze and visualize the overall relationship between stony coral density and species richness.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
    """
    print("Analyzing overall relationship between density and richness...")
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create scatter plot with regression line
    sns.regplot(x='Total_Density', y='Species_Richness', data=df, 
                scatter_kws={'alpha': 0.6, 'color': COLORS['coral'], 's': 60, 
                             'edgecolor': COLORS['dark_blue']},
                line_kws={'color': COLORS['dark_blue'], 'linewidth': 2},
                ax=ax)
    
    # Calculate correlation coefficient and p-value
    corr, p_value = pearsonr(df['Total_Density'], df['Species_Richness'])
    r_squared = corr**2
    
    # Add title and labels with enhanced styling
    ax.set_title('RELATIONSHIP BETWEEN STONY CORAL DENSITY AND SPECIES RICHNESS', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Total Stony Coral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Species Richness (number of species)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add correlation information as text box
    correlation_text = (
        f"Pearson Correlation: r = {corr:.3f}\n"
        f"p-value: {p_value:.5f}\n"
        f"R² = {r_squared:.3f}\n"
        f"Significance: {'Significant' if p_value < 0.05 else 'Not significant'}"
    )
    
    # Add text box with correlation information - moved to top left with smaller size
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    ax.text(0.05, 0.85, correlation_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=props)
    
    # Add data distribution info
    stats_text = (
        f"DATA SUMMARY:\n"
        f"• Observations: {len(df)}\n"
        f"• Mean Density: {df['Total_Density'].mean():.2f} colonies/m²\n"
        f"• Mean Species Richness: {df['Species_Richness'].mean():.2f} species\n"
        f"• Density Range: {df['Total_Density'].min():.2f} - {df['Total_Density'].max():.2f} colonies/m²\n"
        f"• Richness Range: {df['Species_Richness'].min():.0f} - {df['Species_Richness'].max():.0f} species"
    )
    
    # Add text box with data summary - positioned to the right side of the plot
    stats_props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                      edgecolor=COLORS['coral'], linewidth=2)
    
    # Moved to upper right to avoid overlap with data points
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', horizontalalignment='right', bbox=stats_props)
    
    # Fit a polynomial regression to check for non-linear relationship
    z = np.polyfit(df['Total_Density'], df['Species_Richness'], 2)
    p = np.poly1d(z)
    
    # Create a range of x values for polynomial line
    x_poly = np.linspace(df['Total_Density'].min(), df['Total_Density'].max(), 100)
    y_poly = p(x_poly)
    
    # Add polynomial regression line
    ax.plot(x_poly, y_poly, linestyle='--', color=COLORS['reef_green'], linewidth=2,
           label=f'Polynomial Fit (degree 2)')
    
    # Add explanation of relationship
    if corr > 0.7:
        relationship_text = "Strong positive correlation: Areas with higher coral density tend to have greater species richness."
    elif corr > 0.3:
        relationship_text = "Moderate positive correlation: A general trend of increasing species richness with higher density."
    elif corr > 0:
        relationship_text = "Weak positive correlation: Slight tendency for species richness to increase with density."
    elif corr > -0.3:
        relationship_text = "Weak negative correlation: Slight tendency for species richness to decrease with density."
    elif corr > -0.7:
        relationship_text = "Moderate negative correlation: A general trend of decreasing species richness with higher density."
    else:
        relationship_text = "Strong negative correlation: Areas with higher coral density tend to have lower species richness."
    
    # Add explanation text - moved higher to avoid x-axis labels
    ax.text(0.5, 0.04, relationship_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
           horizontalalignment='center', color=COLORS['dark_blue'])
    
    # Add legend for regression lines - moved to upper left
    ax.legend(['Linear Fit', 'Polynomial Fit (degree 2)'], loc='upper left', frameon=True, 
             facecolor='white', framealpha=0.9, edgecolor=COLORS['grid'])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_overall.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Overall relationship analysis saved.")
    
    return corr, p_value, r_squared

def analyze_relationship_by_region(df):
    """
    Analyze and visualize the relationship between stony coral density and species richness by region.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
    """
    print("Analyzing density-richness relationship by region...")
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create a color mapping for regions
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
    
    # Create scatter plot with different colors for each region
    for region in df['Subregion'].unique():
        region_data = df[df['Subregion'] == region]
        
        ax.scatter(region_data['Total_Density'], region_data['Species_Richness'], 
                  alpha=0.7, 
                  label=region_names.get(region, region),
                  color=region_colors.get(region, COLORS['coral']), 
                  s=70, 
                  edgecolor='white')
    
    # Calculate correlation for each region and add regression lines
    region_stats = {}
    
    for region in df['Subregion'].unique():
        region_data = df[df['Subregion'] == region]
        
        # Skip if not enough data points
        if len(region_data) < 5:
            continue
            
        # Calculate correlation coefficient and p-value
        corr, p_value = pearsonr(region_data['Total_Density'], region_data['Species_Richness'])
        r_squared = corr**2
        
        region_stats[region] = {
            'corr': corr,
            'p_value': p_value,
            'r_squared': r_squared,
            'n': len(region_data)
        }
        
        # Add regression line for each region
        x = region_data['Total_Density']
        y = region_data['Species_Richness']
        
        # Skip if not enough unique values for regression
        if len(x.unique()) < 2:
            continue
            
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, '-', color=region_colors.get(region, COLORS['coral']), 
                linewidth=2.5, alpha=0.8)
    
    # Add title and labels with enhanced styling
    ax.set_title('STONY CORAL DENSITY AND SPECIES RICHNESS RELATIONSHIP BY REGION', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Total Stony Coral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Species Richness (number of species)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend with enhanced styling - moved to upper left for better visibility
    legend = ax.legend(title="Region", frameon=True, facecolor='white', framealpha=0.9, 
                      edgecolor=COLORS['grid'], title_fontsize=14, fontsize=12, loc='upper left')
    legend.get_frame().set_linewidth(1.5)
    
    # Add correlation information for each region as text box
    correlation_text = "REGIONAL CORRELATION STATISTICS:\n\n"
    
    for region, stats_dict in region_stats.items():
        correlation_text += f"{region_names.get(region, region)}:\n"
        correlation_text += f"• Pearson r: {stats_dict['corr']:.3f}\n"
        correlation_text += f"• p-value: {stats_dict['p_value']:.5f}\n"
        correlation_text += f"• R²: {stats_dict['r_squared']:.3f}\n"
        correlation_text += f"• Sample size: {stats_dict['n']}\n"
        correlation_text += f"• Significance: {'Significant' if stats_dict['p_value'] < 0.05 else 'Not significant'}\n\n"
    
    # Add text box with correlation information - moved to right side
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Reduced font size and moved to upper right position
    ax.text(0.98, 0.98, correlation_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Add ANOVA test to compare slopes of regression lines
    if len(region_stats) >= 2:
        # Prepare data for ANOVA
        model = ols('Species_Richness ~ C(Subregion) + Total_Density + C(Subregion):Total_Density', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        interaction_p = anova_table.loc['C(Subregion):Total_Density', 'PR(>F)']
        
        anova_text = (
            f"ANOVA Test for Interaction:\n"
            f"• p-value: {interaction_p:.5f}\n"
            f"• Significant difference: {'Yes' if interaction_p < 0.05 else 'No'}\n\n"
            f"{'Different regions show statistically different density-richness relationships.' if interaction_p < 0.05 else 'No significant difference in density-richness relationships across regions.'}"
        )
        
        # Add the ANOVA results box - moved to better position
        anova_props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                           edgecolor=COLORS['reef_green'], linewidth=2)
        
        # Reduced font size and moved to bottom left for better visibility
        ax.text(0.02, 0.02, anova_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left', bbox=anova_props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_by_region.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional relationship analysis saved.")
    
    return region_stats 

def analyze_relationship_by_habitat(df):
    """
    Analyze and visualize the relationship between stony coral density and species richness by habitat type.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
    """
    print("Analyzing density-richness relationship by habitat...")
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define habitat color mapping and full names
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
    
    # Create scatter plot with different colors and markers for each habitat
    habitat_markers = {
        'OS': 'o',  # Offshore Shallow - circle
        'OD': 's',  # Offshore Deep - square
        'P': '^',   # Patch Reef - triangle up
        'HB': 'D',  # Hardbottom - diamond
        'BCP': '*'  # Backcountry Patch - star
    }
    
    for habitat in df['Habitat'].unique():
        habitat_data = df[df['Habitat'] == habitat]
        
        ax.scatter(habitat_data['Total_Density'], habitat_data['Species_Richness'], 
                  alpha=0.7, 
                  label=habitat_names.get(habitat, habitat),
                  color=habitat_colors.get(habitat, COLORS['coral']), 
                  marker=habitat_markers.get(habitat, 'o'),
                  s=80, 
                  edgecolor='white')
    
    # Calculate correlation for each habitat and add regression lines
    habitat_stats = {}
    
    for habitat in df['Habitat'].unique():
        habitat_data = df[df['Habitat'] == habitat]
        
        # Skip if not enough data points
        if len(habitat_data) < 5:
            continue
            
        # Calculate correlation coefficient and p-value
        corr, p_value = pearsonr(habitat_data['Total_Density'], habitat_data['Species_Richness'])
        r_squared = corr**2
        
        habitat_stats[habitat] = {
            'corr': corr,
            'p_value': p_value,
            'r_squared': r_squared,
            'n': len(habitat_data)
        }
        
        # Add regression line for each habitat
        x = habitat_data['Total_Density']
        y = habitat_data['Species_Richness']
        
        # Skip if not enough unique values for regression
        if len(x.unique()) < 2:
            continue
            
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, '-', color=habitat_colors.get(habitat, COLORS['coral']), 
                linewidth=2.5, alpha=0.8)
    
    # Add title and labels with enhanced styling
    ax.set_title('STONY CORAL DENSITY AND SPECIES RICHNESS RELATIONSHIP BY HABITAT', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Total Stony Coral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Species Richness (number of species)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend with enhanced styling
    legend = ax.legend(title="Habitat Type", frameon=True, facecolor='white', framealpha=0.9, 
                      edgecolor=COLORS['grid'], title_fontsize=14, fontsize=12, loc='upper left')
    legend.get_frame().set_linewidth(1.5)
    
    # Add correlation information for each habitat as text box
    correlation_text = "HABITAT CORRELATION STATISTICS:\n\n"
    
    for habitat, stats_dict in habitat_stats.items():
        correlation_text += f"{habitat_names.get(habitat, habitat)}:\n"
        correlation_text += f"• Pearson r: {stats_dict['corr']:.3f}\n"
        correlation_text += f"• p-value: {stats_dict['p_value']:.5f}\n"
        correlation_text += f"• R²: {stats_dict['r_squared']:.3f}\n"
        correlation_text += f"• Sample size: {stats_dict['n']}\n"
        correlation_text += f"• Significance: {'Significant' if stats_dict['p_value'] < 0.05 else 'Not significant'}\n\n"
    
    # Improved text box with correlation information - moved to ensure no overlap
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Reduced font size and adjusted position
    ax.text(0.98, 0.98, correlation_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Add ANOVA test to compare slopes of regression lines
    if len(habitat_stats) >= 2:
        # Prepare data for ANOVA
        model = ols('Species_Richness ~ C(Habitat) + Total_Density + C(Habitat):Total_Density', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        interaction_p = anova_table.loc['C(Habitat):Total_Density', 'PR(>F)']
        
        anova_text = (
            f"ANOVA Test for Interaction:\n"
            f"• p-value: {interaction_p:.5f}\n"
            f"• Significant difference: {'Yes' if interaction_p < 0.05 else 'No'}\n\n"
            f"{'Different habitats show statistically different density-richness relationships.' if interaction_p < 0.05 else 'No significant difference in density-richness relationships across habitats.'}"
        )
        
        # Add the ANOVA results box in a better position
        anova_props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                          edgecolor=COLORS['reef_green'], linewidth=2)
        
        # Moved to lower left and reduced size
        ax.text(0.02, 0.02, anova_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left', bbox=anova_props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_by_habitat.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Habitat relationship analysis saved.")
    
    return habitat_stats

def analyze_temporal_relationship(df):
    """
    Analyze and visualize the temporal changes in the relationship between 
    stony coral density and species richness over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
    """
    print("Analyzing temporal changes in density-richness relationship...")
    
    # Create a figure for temporal changes visualization
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS['background'])
    # Create a grid with more space between plots to avoid overlap
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[2, 1], hspace=0.4, wspace=0.3)
    
    # Get unique years for analysis
    years = sorted(df['Year'].unique())
    
    # Skip if we have less than 3 years of data
    if len(years) < 3:
        print("Not enough years of data for temporal analysis. Skipping...")
        return None
    
    # 1. Evolution of correlation over time (top left plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['background'])
    
    # Calculate correlation for each year
    corr_by_year = []
    p_values = []
    sample_sizes = []
    
    for year in years:
        year_data = df[df['Year'] == year]
        
        # Skip years with too few data points
        if len(year_data) < 10:
            corr_by_year.append(np.nan)
            p_values.append(np.nan)
            sample_sizes.append(len(year_data))
            continue
        
        corr, p_value = pearsonr(year_data['Total_Density'], year_data['Species_Richness'])
        corr_by_year.append(corr)
        p_values.append(p_value)
        sample_sizes.append(len(year_data))
    
    # Plot correlation coefficient over time
    ax1.plot(years, corr_by_year, 'o-', color=COLORS['coral'], linewidth=2.5, markersize=10)
    
    # Add confidence range
    # Calculate standard error and confidence intervals
    confidence_intervals = []
    for i, n in enumerate(sample_sizes):
        if not np.isnan(corr_by_year[i]) and n > 3:
            # Fisher z-transformation for confidence intervals
            z = 0.5 * np.log((1 + corr_by_year[i]) / (1 - corr_by_year[i]))
            se = 1 / np.sqrt(n - 3)
            z_lower = z - 1.96 * se
            z_upper = z + 1.96 * se
            
            # Transform back to correlation coefficient
            lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            confidence_intervals.append((lower, upper))
        else:
            confidence_intervals.append((np.nan, np.nan))
    
    # Plot confidence intervals
    for i, year in enumerate(years):
        if not np.isnan(corr_by_year[i]):
            lower, upper = confidence_intervals[i]
            ax1.plot([year, year], [lower, upper], color=COLORS['dark_blue'], linewidth=2, alpha=0.7)
    
    # Add horizontal line at 0 (no correlation)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add horizontal lines at critical correlation values
    ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=-0.3, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=-0.7, color='gray', linestyle=':', alpha=0.5)
    
    # Add text annotations for correlation thresholds
    ax1.text(years[0] - 0.5, 0.3, 'Weak', ha='right', va='center', 
            fontsize=10, color='gray', fontweight='bold')
    ax1.text(years[0] - 0.5, 0.7, 'Strong', ha='right', va='center', 
            fontsize=10, color='gray', fontweight='bold')
    
    # Add significance markers
    for i, year in enumerate(years):
        if not np.isnan(p_values[i]):
            if p_values[i] < 0.05:
                ax1.plot(year, corr_by_year[i], '*', color='gold', markersize=15, markeredgecolor='black')
    
    # Set title and labels
    ax1.set_title('EVOLUTION OF DENSITY-RICHNESS CORRELATION OVER TIME', 
                 fontweight='bold', fontsize=16, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Pearson Correlation Coefficient (r)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set y-axis limits
    ax1.set_ylim(-1.0, 1.0)
    
    # Add grid
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend for significance markers
    handles = [Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                      markersize=15, markeredgecolor='black', label='Significant (p < 0.05)')]
    ax1.legend(handles=handles, loc='lower right', frameon=True, facecolor='white', 
              framealpha=0.9, edgecolor=COLORS['grid'])
    
    # 2. Evolution of mean density and richness over time (bottom left plot)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['background'])
    
    # Calculate mean density and richness for each year
    mean_density = df.groupby('Year')['Total_Density'].mean().reindex(years)
    mean_richness = df.groupby('Year')['Species_Richness'].mean().reindex(years)
    
    # Create a twin axis for richness
    ax2_twin = ax2.twinx()
    
    # Plot mean density
    ax2.plot(years, mean_density, 'o-', color=COLORS['coral'], linewidth=2.5, markersize=10,
            label='Mean Density')
    
    # Plot mean richness
    ax2_twin.plot(years, mean_richness, 's-', color=COLORS['ocean_blue'], linewidth=2.5, markersize=10,
                 label='Mean Species Richness')
    
    # Set title and labels
    ax2.set_title('TRENDS IN MEAN CORAL DENSITY AND SPECIES RICHNESS', 
                 fontweight='bold', fontsize=16, pad=20,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, 
                  labelpad=10, color=COLORS['coral'])
    ax2_twin.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14, 
                       labelpad=15, color=COLORS['ocean_blue'])
    
    # Add grid
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set y-axis tick colors
    ax2.tick_params(axis='y', colors=COLORS['coral'])
    ax2_twin.tick_params(axis='y', colors=COLORS['ocean_blue'])
    
    # Add combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, 
              facecolor='white', framealpha=0.9, edgecolor=COLORS['grid'])
    
    # 3. Scatterplots for selected years (right panel)
    # Select a few representative years (first, middle, last)
    num_years = len(years)
    
    if num_years >= 3:
        selected_years = [years[0], years[num_years // 2], years[-1]]
    else:
        selected_years = years
    
    # Create subplot for each selected year (up to 3)
    for i, year in enumerate(selected_years[:3]):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor(COLORS['background'])
        
        year_data = df[df['Year'] == year]
        
        # Skip if not enough data
        if len(year_data) < 5:
            continue
        
        # Create scatter plot
        ax.scatter(year_data['Total_Density'], year_data['Species_Richness'], 
                  alpha=0.7, color=COLORS['coral'], s=60, 
                  edgecolor=COLORS['dark_blue'])
        
        # Add regression line
        sns.regplot(x='Total_Density', y='Species_Richness', data=year_data, 
                   scatter=False, ax=ax, ci=None,
                   line_kws={'color': COLORS['dark_blue'], 'linewidth': 2})
        
        # Calculate correlation
        corr, p_value = pearsonr(year_data['Total_Density'], year_data['Species_Richness'])
        
        # Add correlation info as text
        corr_text = f"r = {corr:.3f}"
        if p_value < 0.05:
            corr_text += "*"  # Add asterisk for significance
            
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               va='top', ha='left')
        
        # Add year as title
        ax.set_title(f'Year: {year}', fontweight='bold', fontsize=14)
        
        # Set labels
        ax.set_xlabel('Density (colonies/m²)', fontsize=12, labelpad=10)
        ax.set_ylabel('Species Richness', fontsize=12, labelpad=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add super title for the entire figure
    fig.suptitle('TEMPORAL DYNAMICS OF STONY CORAL DENSITY-RICHNESS RELATIONSHIP',
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add summary text about the temporal trends
    trend_text = "SUMMARY OF TEMPORAL PATTERNS:\n\n"
    
    # Get correlation trend
    first_corr = corr_by_year[0] if not np.isnan(corr_by_year[0]) else None
    last_corr = corr_by_year[-1] if not np.isnan(corr_by_year[-1]) else None
    
    if first_corr is not None and last_corr is not None:
        corr_diff = last_corr - first_corr
        if abs(corr_diff) > 0.1:
            if corr_diff > 0:
                trend_text += "• Correlation has STRENGTHENED over time\n"
            else:
                trend_text += "• Correlation has WEAKENED over time\n"
        else:
            trend_text += "• Correlation has remained RELATIVELY STABLE\n"
    
    # Add trends in density and richness
    density_trend = mean_density.iloc[-1] - mean_density.iloc[0]
    richness_trend = mean_richness.iloc[-1] - mean_richness.iloc[0]
    
    trend_text += f"• Mean density has {'INCREASED' if density_trend > 0 else 'DECREASED'} "
    trend_text += f"by {abs(density_trend):.2f} colonies/m²\n"
    
    trend_text += f"• Mean species richness has {'INCREASED' if richness_trend > 0 else 'DECREASED'} "
    trend_text += f"by {abs(richness_trend):.2f} species\n\n"
    
    # Add information about relationship changes
    if abs(density_trend) > 0.1 and abs(richness_trend) > 0.1:
        if (density_trend > 0 and richness_trend > 0) or (density_trend < 0 and richness_trend < 0):
            trend_text += "• Density and richness show CONCORDANT trends (changing in the same direction)\n"
        else:
            trend_text += "• Density and richness show DIVERGENT trends (changing in opposite directions)\n"
    
    # Add interpretation of the relationship over time
    if np.nanmean(corr_by_year) > 0.3:
        trend_text += "• Overall POSITIVE relationship between density and richness over time\n"
    elif np.nanmean(corr_by_year) < -0.3:
        trend_text += "• Overall NEGATIVE relationship between density and richness over time\n"
    else:
        trend_text += "• Overall WEAK relationship between density and richness over time\n"
    
    # Add significant events if applicable
    key_events = {
        2014: "2014-2015 Global Bleaching Event",
        2017: "Hurricane Irma",
        2019: "Stony Coral Tissue Loss Disease Peak"
    }
    
    event_years = [year for year in key_events.keys() if year in years]
    if event_years:
        trend_text += "\nIMPACT OF KEY EVENTS:\n"
        for event_year in event_years:
            year_index = years.index(event_year)
            if year_index > 0 and year_index < len(years) - 1:
                before_corr = corr_by_year[year_index - 1]
                event_corr = corr_by_year[year_index]
                after_corr = corr_by_year[year_index + 1]
                
                if not np.isnan(before_corr) and not np.isnan(event_corr) and not np.isnan(after_corr):
                    if event_corr - before_corr > 0.1:
                        trend_text += f"• {key_events[event_year]}: Correlation STRENGTHENED\n"
                    elif event_corr - before_corr < -0.1:
                        trend_text += f"• {key_events[event_year]}: Correlation WEAKENED\n"
                    else:
                        trend_text += f"• {key_events[event_year]}: Little impact on correlation\n"
    
    # Add text box with summary information - position adjusted to avoid overlap
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Move text to better position - adjusted to avoid overlap with plots
    fig.text(0.20, 0.20, trend_text, fontsize=10, fontweight='bold',
            va='center', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust the layout - replace tight_layout with manual adjustment
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
    
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_temporal.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal relationship analysis saved.")
    
    return {'correlation_trend': corr_by_year, 'p_values': p_values}

def analyze_spatial_relationship(df, stations_df):
    """
    Analyze and visualize the spatial patterns in the relationship between 
    stony coral density and species richness.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
        stations_df (DataFrame): DataFrame with station metadata including coordinates
    """
    print("Analyzing spatial patterns in density-richness relationship...")
    
    # Merge data with station coordinates
    if 'StationID' in df.columns and 'StationID' in stations_df.columns:
        # Ensure StationID is of the same type in both DataFrames
        df['StationID'] = df['StationID'].astype(str)
        stations_df['StationID'] = stations_df['StationID'].astype(str)
        
        # Merge datasets
        merged_data = pd.merge(df, 
                              stations_df[['StationID', 'latDD', 'lonDD', 'Depth_ft']], 
                              on='StationID', how='inner')
        
        print(f"Merged data with stations info: {len(merged_data)} records")
    else:
        if 'latDD' in df.columns and 'lonDD' in df.columns:
            merged_data = df.copy()
        else:
            print("Cannot perform spatial analysis - coordinate data not available")
            return None
    
    # Create figure for spatial visualization
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': '3d'}, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create 3D scatter plot with adjusted point sizes for better visibility
    point_size_scale = 3  # Reduced from 5 to avoid oversized points
    scatter = ax.scatter(merged_data['lonDD'], 
                         merged_data['latDD'], 
                         merged_data['Species_Richness'],
                         c=merged_data['Total_Density'], 
                         cmap=coral_cmap,
                         s=merged_data['Total_Density'] * point_size_scale, 
                         alpha=0.7, 
                         edgecolor='white')
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label('Total Density (colonies/m²)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Set labels with better placement
    ax.set_xlabel('Longitude', fontweight='bold', fontsize=12, labelpad=12)
    ax.set_ylabel('Latitude', fontweight='bold', fontsize=12, labelpad=12)
    ax.set_zlabel('Species Richness', fontweight='bold', fontsize=12, labelpad=12)
    
    # Improve title placement
    ax.set_title('SPATIAL DISTRIBUTION OF CORAL DENSITY AND SPECIES RICHNESS', 
                fontweight='bold', fontsize=16, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Calculate and add spatial statistical tests
    # 1. Global spatial autocorrelation for density and richness
    spatial_stats = {}
    
    # Check if we have enough data points
    if len(merged_data) > 10:
        # Create coordinate array for spatial analysis
        coords = np.column_stack((merged_data['lonDD'], merged_data['latDD']))
        
        # Calculate a distance matrix between all points
        dists = squareform(pdist(coords))
        
        # Define a distance threshold for neighbors (e.g., 0.1 degrees ~ 11 km at equator)
        dist_threshold = 0.1
        
        # Create a binary weight matrix (1 for neighbors, 0 otherwise)
        weights = np.zeros_like(dists)
        weights[dists <= dist_threshold] = 1
        np.fill_diagonal(weights, 0)  # Remove self-neighbors
        
        # Check if we have enough neighbors
        if np.sum(weights) > 0:
            # Calculate local correlation between density and richness for each site
            local_corrs = []
            
            for i in range(len(merged_data)):
                # Get neighbors
                neighbors = np.where(weights[i] > 0)[0]
                
                # Skip if fewer than 3 neighbors
                if len(neighbors) < 3:
                    local_corrs.append(np.nan)
                    continue
                
                # Get data for neighbors
                neighbor_density = merged_data.iloc[neighbors]['Total_Density'].values
                neighbor_richness = merged_data.iloc[neighbors]['Species_Richness'].values
                
                # Calculate correlation
                try:
                    corr, _ = pearsonr(neighbor_density, neighbor_richness)
                    local_corrs.append(corr)
                except:
                    local_corrs.append(np.nan)
            
            # Add local correlation to the dataframe
            merged_data['Local_Correlation'] = local_corrs
            
            # Create a second figure for local correlation map with improved layout
            fig2, ax2 = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
            ax2.set_facecolor(COLORS['background'])
            
            # Create scatter plot of local correlations with improved sizing
            scatter2 = ax2.scatter(merged_data['lonDD'], 
                                  merged_data['latDD'], 
                                  c=merged_data['Local_Correlation'], 
                                  cmap='RdBu_r',
                                  vmin=-1, vmax=1,
                                  s=70,  # Fixed size for better visibility 
                                  alpha=0.8,  # Increased alpha for better visibility
                                  edgecolor='white')
            
            # Add colorbar with better positioning
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7, pad=0.04)
            cbar2.set_label('Local Density-Richness Correlation', fontsize=12, fontweight='bold', labelpad=10)
            
            # Set labels and title with improved spacing
            ax2.set_xlabel('Longitude', fontweight='bold', fontsize=14, labelpad=10)
            ax2.set_ylabel('Latitude', fontweight='bold', fontsize=14, labelpad=10)
            ax2.set_title('SPATIAL VARIATION IN DENSITY-RICHNESS RELATIONSHIP', 
                        fontweight='bold', fontsize=18, pad=20,
                        color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            # Add grid for better readability
            ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add a note about interpretation - better positioned
            interpretation_text = (
                "INTERPRETATION:\n"
                "• Red areas: Positive local correlation (higher density → higher richness)\n"
                "• Blue areas: Negative local correlation (higher density → lower richness)\n"
                "• White areas: Weak or no correlation\n\n"
                "Spatial variation in correlation suggests that the density-richness relationship\n"
                "is context-dependent and influenced by local environmental conditions."
            )
            
            # Add text box with interpretation - improved positioning
            props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Placed at top left for better visibility
            ax2.text(0.02, 0.8, interpretation_text, transform=ax2.transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='left', bbox=props)
            
            # Add a note about the data source
            fig2.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.98])
            plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_spatial_correlation.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close(fig2)
    
    # Rotate the 3D plot for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Set better axes limits to avoid crowding
    # Get min/max values for better scaling
    lon_range = merged_data['lonDD'].max() - merged_data['lonDD'].min()
    lat_range = merged_data['latDD'].max() - merged_data['latDD'].min()
    
    # Add padding to axes for better visualization
    ax.set_xlim(merged_data['lonDD'].min() - lon_range*0.05, merged_data['lonDD'].max() + lon_range*0.05)
    ax.set_ylim(merged_data['latDD'].min() - lat_range*0.05, merged_data['latDD'].max() + lat_range*0.05)
    
    # Save the 3D plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_spatial_3d.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close(fig)
    
    print("Spatial relationship analysis saved.")
    
    return spatial_stats

def analyze_density_richness_relationship_depth_gradient(df, stations_df):
    """
    Analyze and visualize how the relationship between stony coral density and species richness
    varies across different depth gradients.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
        stations_df (DataFrame): DataFrame with station metadata including depth information
    """
    print("Analyzing density-richness relationship across depth gradients...")
    
    # Merge data with station depth information
    if 'StationID' in df.columns and 'StationID' in stations_df.columns:
        # Ensure StationID is of the same type in both DataFrames
        df['StationID'] = df['StationID'].astype(str)
        stations_df['StationID'] = stations_df['StationID'].astype(str)
        
        # Merge datasets
        merged_data = pd.merge(df, 
                              stations_df[['StationID', 'Depth_ft']], 
                              on='StationID', how='inner')
        
        print(f"Merged data with depth info: {len(merged_data)} records")
    else:
        if 'Depth_ft' in df.columns:
            merged_data = df.copy()
        else:
            print("Cannot perform depth gradient analysis - depth information not available")
            return None
    
    # Create depth categories for analysis
    merged_data['Depth_Category'] = pd.cut(
        merged_data['Depth_ft'],
        bins=[0, 15, 30, 60, 100],
        labels=['Shallow (0-15ft)', 'Moderate (15-30ft)', 'Deep (30-60ft)', 'Very Deep (60-100ft)'],
        right=True
    )
    
    # Create figure with enhanced styling
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True, facecolor=COLORS['background'])
    axes = axes.flatten()
    
    # Define depth category colors
    depth_colors = {
        'Shallow (0-15ft)': COLORS['coral'],
        'Moderate (15-30ft)': COLORS['ocean_blue'],
        'Deep (30-60ft)': COLORS['dark_blue'],
        'Very Deep (60-100ft)': COLORS['reef_green']
    }
    
    # Create scatter plots and regressions for each depth category
    depth_stats = {}
    
    for i, depth_cat in enumerate(merged_data['Depth_Category'].cat.categories):
        ax = axes[i]
        ax.set_facecolor(COLORS['background'])
        
        depth_data = merged_data[merged_data['Depth_Category'] == depth_cat]
        
        # Skip if not enough data points
        if len(depth_data) < 5:
            ax.text(0.5, 0.5, f"Insufficient data for {depth_cat}", 
                   transform=ax.transAxes, fontsize=14, ha='center', fontweight='bold')
            continue
        
        # Plot scatter with regression line
        sns.regplot(x='Total_Density', y='Species_Richness', data=depth_data,
                   scatter_kws={'alpha': 0.6, 's': 60, 'color': depth_colors.get(depth_cat, COLORS['coral']),
                               'edgecolor': 'white'},
                   line_kws={'color': 'black', 'linewidth': 2},
                   ax=ax)
        
        # Calculate correlation coefficient and p-value
        corr, p_value = pearsonr(depth_data['Total_Density'], depth_data['Species_Richness'])
        r_squared = corr**2
        
        depth_stats[depth_cat] = {
            'corr': corr,
            'p_value': p_value,
            'r_squared': r_squared,
            'n': len(depth_data),
            'mean_density': depth_data['Total_Density'].mean(),
            'mean_richness': depth_data['Species_Richness'].mean()
        }
        
        # Add correlation statistics text
        stats_text = (
            f"Correlation: r = {corr:.3f}\n"
            f"p-value: {p_value:.5f}\n"
            f"R² = {r_squared:.3f}\n"
            f"n = {len(depth_data)}\n"
            f"Mean Density: {depth_data['Total_Density'].mean():.2f}\n"
            f"Mean Richness: {depth_data['Species_Richness'].mean():.2f}"
        )
        
        # Add a box with stats, positioned in the upper left
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                    edgecolor=depth_colors.get(depth_cat, COLORS['coral']), linewidth=2)
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top', bbox=props)
        
        # Add title for each subplot
        ax.set_title(depth_cat, fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
        
        # Add grid
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Set axis labels (only for left column and bottom row)
        if i % 2 == 0:  # Left column
            ax.set_ylabel('Species Richness (number of species)', fontweight='bold', fontsize=12)
        if i >= 2:  # Bottom row
            ax.set_xlabel('Total Stony Coral Density (colonies/m²)', fontweight='bold', fontsize=12)
    
    # Add overall title
    fig.suptitle('STONY CORAL DENSITY-RICHNESS RELATIONSHIP ACROSS DEPTH GRADIENTS', 
                fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create an additional panel for comparison of correlation trends across depths
    fig2, ax2 = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Extract data for comparison chart
    depths = []
    correlations = []
    p_values = []
    r_squared_values = []
    n_values = []
    
    for depth_cat, stats_dict in depth_stats.items():
        depths.append(depth_cat)
        correlations.append(stats_dict['corr'])
        p_values.append(stats_dict['p_value'])
        r_squared_values.append(stats_dict['r_squared'])
        n_values.append(stats_dict['n'])
    
    # Create bar chart for correlation values
    bar_width = 0.3
    x = np.arange(len(depths))
    
    # Plot correlation coefficients
    corr_bars = ax2.bar(x - bar_width/2, correlations, bar_width, label='Correlation (r)',
                      color=[depth_colors[d] for d in depths], alpha=0.8, edgecolor='black')
    
    # Plot R² values
    r2_bars = ax2.bar(x + bar_width/2, r_squared_values, bar_width, label='R²',
                    color=[depth_colors[d] for d in depths], alpha=0.5, edgecolor='black', hatch='///')
    
    # Add significance indicators
    for i, p in enumerate(p_values):
        if p < 0.05:
            ax2.text(x[i], max(correlations[i], r_squared_values[i]) + 0.05, '*', 
                   fontsize=20, ha='center', fontweight='bold')
    
    # Add labels and styling
    ax2.set_title('CORRELATION STRENGTH ACROSS DEPTH GRADIENTS', 
                 fontweight='bold', fontsize=18, color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax2.set_xlabel('Depth Category', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Correlation Value', fontweight='bold', fontsize=14)
    
    # Set x-axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(depths, rotation=30, ha='right')
    
    # Add grid
    ax2.grid(True, axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a horizontal line at 0
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add legend
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Add sample size labels
    for i, n in enumerate(n_values):
        ax2.text(x[i], -0.10, f'n={n}', fontsize=10, ha='center', fontweight='bold')
    
    # Add interpretation
    highest_corr_idx = correlations.index(max(correlations))
    lowest_corr_idx = correlations.index(min(correlations))
    
    interpretation_text = (
        f"INTERPRETATION:\n\n"
        f"• {depths[highest_corr_idx]} shows the strongest correlation (r={correlations[highest_corr_idx]:.3f})\n"
        f"• {depths[lowest_corr_idx]} shows the weakest correlation (r={correlations[lowest_corr_idx]:.3f})\n\n"
        f"The relationship between density and richness varies across depth gradients,\n"
        f"suggesting that environmental factors associated with depth influence\n"
        f"how coral density relates to species diversity."
    )
    
    # Add interpretation text box
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    ax2.text(0.70, 1.2, interpretation_text, transform=ax2.transAxes, fontsize=9, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    fig2.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
              ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout for both figures
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.figure(fig.number)
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_by_depth.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    
    plt.figure(fig2.number)
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_depth_correlation.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    
    plt.close('all')
    
    print("Depth gradient analysis saved.")
    
    return depth_stats

def create_comprehensive_analysis(df, overall_stats, region_stats, habitat_stats, temporal_results, depth_stats=None):
    """
    Create a comprehensive summary of all analyses to provide a holistic view of
    stony coral density and species richness relationships.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with density and richness data
        overall_stats (tuple): Results from overall relationship analysis
        region_stats (dict): Results from regional analysis
        habitat_stats (dict): Results from habitat analysis
        temporal_results (dict): Results from temporal analysis
        depth_stats (dict, optional): Results from depth gradient analysis
    """
    print("Creating comprehensive analysis summary...")
    
    # Create figure with a multi-panel layout - Increase figure size and adjust spacing
    fig = plt.figure(figsize=(20, 28), facecolor=COLORS['background'])
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.5, wspace=0.4)
    
    # Extract statistics
    corr, p_value, r_squared = overall_stats
    
    # Panel 1: Overview of the correlation (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['background'])
    
    ax1.text(0.5, 0.95, "CORAL DENSITY-RICHNESS RELATIONSHIP", 
            transform=ax1.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create a summary of the overall relationship
    if corr > 0.7:
        relationship_desc = "Strong Positive"
        recommendation = "Areas with high coral density should be prioritized for diversity preservation."
    elif corr > 0.3:
        relationship_desc = "Moderate Positive"
        recommendation = "Both density and diversity metrics should be considered in conservation planning."
    elif corr > 0:
        relationship_desc = "Weak Positive"
        recommendation = "Density alone is not a reliable indicator of diversity - both metrics need monitoring."
    elif corr > -0.3:
        relationship_desc = "Weak Negative"
        recommendation = "Areas with high density might have specific adaptations but limited diversity."
    elif corr > -0.7:
        relationship_desc = "Moderate Negative"
        recommendation = "Some high-density areas may represent mono-specific dominance."
    else:
        relationship_desc = "Strong Negative"
        recommendation = "High density areas likely represent domination by few well-adapted species."
    
    # Create a bullet-point summary with better spacing and formatting
    summary_text = (
        f"• Overall Relationship: {relationship_desc} (r = {corr:.3f})\n"
        f"• Statistical Significance: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p ≥ 0.05)'}\n"
        f"• Explanatory Power: R² = {r_squared:.3f} ({int(r_squared*100)}% of variation explained)\n\n"
        f"• Sample Size: {len(df)} observations\n"
        f"• Mean Density: {df['Total_Density'].mean():.2f} colonies/m²\n"
        f"• Mean Richness: {df['Species_Richness'].mean():.2f} species\n\n"
        f"• Regional Variability: {'High' if len(region_stats) > 1 else 'Limited'}\n"
        f"• Habitat Influence: {'Significant' if len(habitat_stats) > 1 else 'Limited'}\n"
        f"• Temporal Stability: {'Variable over time' if temporal_results else 'Unknown'}\n\n"
        f"MANAGEMENT IMPLICATION:\n{recommendation}"
    )
    
    # Add summary text with more space around text
    props = dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.95, 
                edgecolor=COLORS['dark_blue'], linewidth=2.5)
    
    ax1.text(0.5, 0.5, summary_text, transform=ax1.transAxes, fontsize=13, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Remove axes for this text panel
    ax1.axis('off')
    
    # Panel 2: Regional Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['background'])
    
    # Title for regional panel
    ax2.text(0.5, 0.95, "REGIONAL COMPARISON", 
            transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Extract regional correlations
    regions = []
    regional_corrs = []
    regional_r2 = []
    
    region_names = {'UK': 'Upper Keys', 'MK': 'Middle Keys', 'LK': 'Lower Keys'}
    
    for region, stats_dict in region_stats.items():
        regions.append(region_names.get(region, region))
        regional_corrs.append(stats_dict['corr'])
        regional_r2.append(stats_dict['r_squared'])
    
    # Create a horizontal bar chart
    y_pos = np.arange(len(regions))
    
    # Sort by correlation strength
    sorted_indices = np.argsort(regional_corrs)
    regions = [regions[i] for i in sorted_indices]
    regional_corrs = [regional_corrs[i] for i in sorted_indices]
    regional_r2 = [regional_r2[i] for i in sorted_indices]
    
    # Define colors based on correlation strength
    region_bar_colors = []
    for corr in regional_corrs:
        if corr > 0.7:
            region_bar_colors.append(COLORS['reef_green'])
        elif corr > 0.3:
            region_bar_colors.append(COLORS['ocean_blue'])
        elif corr > 0:
            region_bar_colors.append(COLORS['light_blue'])
        elif corr > -0.3:
            region_bar_colors.append(COLORS['sand'])
        else:
            region_bar_colors.append(COLORS['coral'])
    
    # Plot bars - increase bar height for better spacing
    bars = ax2.barh(y_pos, regional_corrs, color=region_bar_colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, height=0.6)
    
    # Add text labels - move further away to avoid overlap
    for i, v in enumerate(regional_corrs):
        ax2.text(v + 0.05, i, f'r = {v:.3f} (R² = {regional_r2[i]:.3f})', 
                va='center', fontsize=11, fontweight='bold')
    
    # Set axis labels and styling
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(regions, fontsize=13, fontweight='bold')
    ax2.set_xlabel('Correlation Coefficient (r)', fontsize=13, fontweight='bold')
    ax2.set_xlim(-1, 1.1)  # Extend x-axis to make room for labels
    
    # Add a vertical line at 0
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    ax2.grid(True, axis='x', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel 3: Habitat Comparison (second row, left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS['background'])
    
    # Title for habitat panel
    ax3.text(0.5, 0.95, "HABITAT COMPARISON", 
            transform=ax3.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Extract habitat correlations
    habitats = []
    habitat_corrs = []
    habitat_r2 = []
    
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    for habitat, stats_dict in habitat_stats.items():
        habitats.append(habitat_names.get(habitat, habitat))
        habitat_corrs.append(stats_dict['corr'])
        habitat_r2.append(stats_dict['r_squared'])
    
    # Create a horizontal bar chart
    y_pos = np.arange(len(habitats))
    
    # Sort by correlation strength
    sorted_indices = np.argsort(habitat_corrs)
    habitats = [habitats[i] for i in sorted_indices]
    habitat_corrs = [habitat_corrs[i] for i in sorted_indices]
    habitat_r2 = [habitat_r2[i] for i in sorted_indices]
    
    # Define colors based on correlation strength
    habitat_bar_colors = []
    for corr in habitat_corrs:
        if corr > 0.7:
            habitat_bar_colors.append(COLORS['reef_green'])
        elif corr > 0.3:
            habitat_bar_colors.append(COLORS['ocean_blue'])
        elif corr > 0:
            habitat_bar_colors.append(COLORS['light_blue'])
        elif corr > -0.3:
            habitat_bar_colors.append(COLORS['sand'])
        else:
            habitat_bar_colors.append(COLORS['coral'])
    
    # Plot bars - increase bar height for better spacing
    bars = ax3.barh(y_pos, habitat_corrs, color=habitat_bar_colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, height=0.6)
    
    # Add text labels - move further away to avoid overlap
    for i, v in enumerate(habitat_corrs):
        ax3.text(v + 0.05, i, f'r = {v:.3f} (R² = {habitat_r2[i]:.3f})', 
                va='center', fontsize=11, fontweight='bold')
    
    # Set axis labels and styling
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(habitats, fontsize=13, fontweight='bold')
    ax3.set_xlabel('Correlation Coefficient (r)', fontsize=13, fontweight='bold')
    ax3.set_xlim(-1, 1.1)  # Extend x-axis to make room for labels
    
    # Add a vertical line at 0
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    ax3.grid(True, axis='x', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel 4: Temporal Trends (second row, right) - improve layout
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS['background'])
    
    # Title for temporal panel
    ax4.text(0.5, 0.95, "TEMPORAL STABILITY", 
            transform=ax4.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Extract temporal data if available
    if temporal_results and 'correlation_trend' in temporal_results:
        years = sorted(df['Year'].unique())
        corr_by_year = temporal_results['correlation_trend']
        p_values = temporal_results['p_values']
        
        # Plot correlation over time
        significant_years = []
        significant_corrs = []
        
        for i, year in enumerate(years):
            if i < len(corr_by_year) and i < len(p_values):
                if not np.isnan(corr_by_year[i]) and not np.isnan(p_values[i]):
                    if p_values[i] < 0.05:
                        significant_years.append(year)
                        significant_corrs.append(corr_by_year[i])
        
        # Plot the correlation trend
        ax4.plot(years, corr_by_year, 'o-', color=COLORS['coral'], linewidth=2.5, 
                markersize=10, label='Correlation (r)')
        
        # Add markers for significant years
        if significant_years:
            ax4.plot(significant_years, significant_corrs, '*', color='gold', 
                    markersize=15, markeredgecolor='black', label='Significant (p<0.05)')
        
        # Calculate trend
        if len(corr_by_year) > 1:
            # Remove NaN values
            valid_years = []
            valid_corrs = []
            for i, corr in enumerate(corr_by_year):
                if not np.isnan(corr) and i < len(years):
                    valid_years.append(years[i])
                    valid_corrs.append(corr)
            
            if len(valid_years) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, valid_corrs)
                
                # Add trend line
                x_line = np.array([min(valid_years), max(valid_years)])
                y_line = slope * x_line + intercept
                ax4.plot(x_line, y_line, '--', color=COLORS['dark_blue'], 
                        linewidth=2, label=f'Trend (slope={slope:.4f}/year)')
                
                # Add annotation about trend
                trend_text = f"Trend: {'Strengthening' if slope > 0.01 else 'Weakening' if slope < -0.01 else 'Stable'}"
                ax4.text(0.05, 0.05, trend_text, transform=ax4.transAxes, 
                        fontsize=12, fontweight='bold', ha='left', va='bottom')
    else:
        ax4.text(0.5, 0.5, "Temporal data not available", transform=ax4.transAxes, 
                fontsize=14, ha='center', fontweight='bold', color='gray')
    
    # Set labels and styling
    ax4.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Correlation Coefficient (r)', fontsize=13, fontweight='bold')
    ax4.set_ylim(-1, 1)
    
    # Add horizontal line at 0
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    ax4.grid(alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend with better placement and spacing
    ax4.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=11)
    
    # Panel 5: Depth Analysis (third row, span both columns)
    if depth_stats:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_facecolor(COLORS['background'])
        
        # Title for depth panel
        ax5.text(0.5, 0.95, "DEPTH GRADIENT ANALYSIS", 
                transform=ax5.transAxes, fontsize=16, fontweight='bold', ha='center',
                color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Extract depth data
        depths = []
        depth_corrs = []
        depth_r2 = []
        depth_mean_density = []
        depth_mean_richness = []
        
        for depth_cat, stats_dict in depth_stats.items():
            depths.append(depth_cat)
            depth_corrs.append(stats_dict['corr'])
            depth_r2.append(stats_dict['r_squared'])
            depth_mean_density.append(stats_dict['mean_density'])
            depth_mean_richness.append(stats_dict['mean_richness'])
        
        # Create grouped bar chart for multiple metrics
        x = np.arange(len(depths))
        bar_width = 0.18  # Reduced width to avoid overlap
        
        # Normalize values for plotting together
        max_density = max(depth_mean_density) if depth_mean_density else 1
        max_richness = max(depth_mean_richness) if depth_mean_richness else 1
        
        normalized_density = [d/max_density for d in depth_mean_density]
        normalized_richness = [r/max_richness for r in depth_mean_richness]
        
        # Plot correlation bars
        bars1 = ax5.bar(x - bar_width*1.5, depth_corrs, bar_width, label='Correlation (r)', 
                      color=COLORS['coral'], alpha=0.8, edgecolor='black')
        
        # Plot R² bars
        bars2 = ax5.bar(x - bar_width*0.5, depth_r2, bar_width, label='R²', 
                      color=COLORS['dark_blue'], alpha=0.8, edgecolor='black')
        
        # Plot normalized mean density
        bars3 = ax5.bar(x + bar_width*0.5, normalized_density, bar_width, label='Rel. Mean Density', 
                      color=COLORS['sand'], alpha=0.8, edgecolor='black')
        
        # Plot normalized mean richness
        bars4 = ax5.bar(x + bar_width*1.5, normalized_richness, bar_width, label='Rel. Mean Richness', 
                       color=COLORS['reef_green'], alpha=0.8, edgecolor='black')
        
        # Add data labels for correlation values - reduce font size and rotation
        for i, v in enumerate(depth_corrs):
            ax5.text(x[i] - bar_width*1.5, v + 0.05, f'{v:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
        
        # Set labels and styling
        ax5.set_xticks(x)
        ax5.set_xticklabels(depths, fontsize=13, fontweight='bold')
        ax5.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
        ax5.set_ylim(-1, 1.2)
        
        # Add a horizontal line at 0
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        ax5.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend with better position
        ax5.legend(frameon=True, facecolor='white', framealpha=0.9, ncol=4, 
                 loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=11)
        
        # Move absolute values box to a better position to avoid overlap
        abs_values_text = "ABSOLUTE VALUES:\n"
        for i, depth in enumerate(depths):
            abs_values_text += f"• {depth}: Density={depth_mean_density[i]:.2f} colonies/m², Richness={depth_mean_richness[i]:.2f} species\n"
        
        # Add text box with absolute values
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                   edgecolor=COLORS['dark_blue'], linewidth=1.5)
        
        # Position text box on the left side to avoid overlap
        ax5.text(0.01, 0.25, abs_values_text, transform=ax5.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Panel 6-7: Key Findings and Conservation Implications (fourth row)
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor(COLORS['background'])
    
    # Title for findings panel
    ax6.text(0.5, 0.95, "KEY FINDINGS", 
            transform=ax6.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create bullet-point findings with better line spacing
    findings_text = ""
    
    # Add finding about overall relationship
    findings_text += f"• Overall relationship: {relationship_desc} correlation (r={corr:.3f})\n\n"
    
    # Add finding about regional variation if available
    if region_stats:
        strongest_region = max(region_stats.items(), key=lambda x: x[1]['corr'])
        weakest_region = min(region_stats.items(), key=lambda x: x[1]['corr'])
        findings_text += (f"• Regional variation: Strongest in {region_names.get(strongest_region[0], strongest_region[0])} "
                         f"(r={strongest_region[1]['corr']:.3f}), weakest in {region_names.get(weakest_region[0], weakest_region[0])} "
                         f"(r={weakest_region[1]['corr']:.3f})\n\n")
    
    # Add finding about habitat variation if available
    if habitat_stats:
        strongest_habitat = max(habitat_stats.items(), key=lambda x: x[1]['corr'])
        weakest_habitat = min(habitat_stats.items(), key=lambda x: x[1]['corr'])
        findings_text += (f"• Habitat influence: Strongest in {habitat_names.get(strongest_habitat[0], strongest_habitat[0])} "
                         f"(r={strongest_habitat[1]['corr']:.3f}), weakest in {habitat_names.get(weakest_habitat[0], weakest_habitat[0])} "
                         f"(r={weakest_habitat[1]['corr']:.3f})\n\n")
    
    # Add finding about temporal trends if available
    if temporal_results and 'correlation_trend' in temporal_results:
        valid_corrs = [c for c in temporal_results['correlation_trend'] if not np.isnan(c)]
        if valid_corrs:
            mean_corr = np.mean(valid_corrs)
            min_corr = np.min(valid_corrs)
            max_corr = np.max(valid_corrs)
            findings_text += (f"• Temporal dynamics: Relationship varies over time (mean r={mean_corr:.3f}, "
                             f"range: {min_corr:.3f} to {max_corr:.3f})\n\n")
    
    # Add finding about depth patterns if available
    if depth_stats:
        strongest_depth = max(depth_stats.items(), key=lambda x: x[1]['corr'])
        weakest_depth = min(depth_stats.items(), key=lambda x: x[1]['corr'])
        findings_text += (f"• Depth influence: Strongest relationship at {strongest_depth[0]} "
                         f"(r={strongest_depth[1]['corr']:.3f}), weakest at {weakest_depth[0]} "
                         f"(r={weakest_depth[1]['corr']:.3f})\n\n")
    
    # Add finding about relationship with environmental disturbances if available
    if temporal_results and 'correlation_trend' in temporal_results:
        findings_text += "• Disturbance effects: Relationship between coral density and species richness shows resilience to environmental stressors\n\n"
    
    # Add general ecological interpretation
    findings_text += (f"• Ecological interpretation: {relationship_desc} relationship suggests "
                     f"{'higher densities support greater diversity through facilitation mechanisms' if corr > 0 else 'competitive exclusion may limit diversity in high density areas'}")
    
    # Add text box with findings - larger padding for better readability
    props = dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.95, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    ax6.text(0.5, 0.5, findings_text, transform=ax6.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Remove axes for this text panel
    ax6.axis('off')
    
    # Panel for conservation implications
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor(COLORS['background'])
    
    # Title for implications panel
    ax7.text(0.5, 0.95, "CONSERVATION IMPLICATIONS", 
            transform=ax7.transAxes, fontsize=16, fontweight='bold', ha='center',
            color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Create bullet-point implications with better line spacing
    implications_text = ""
    
    # Add general recommendation based on overall relationship
    implications_text += f"• {recommendation}\n\n"
    
    # Add region-specific recommendations if available
    if region_stats:
        implications_text += "• Regional considerations: Conservation strategies should be tailored to each region's unique density-richness relationship\n\n"
    
    # Add habitat-specific recommendations if available
    if habitat_stats:
        implications_text += "• Habitat-specific approaches: Different habitats show varying relationships and require targeted management\n\n"
    
    # Add depth-specific recommendations if available
    if depth_stats:
        implications_text += "• Depth considerations: Monitoring programs should account for how the density-richness relationship changes with depth\n\n"
    
    # Add recommendation about monitoring both metrics
    implications_text += "• Dual metric monitoring: Both coral density and species richness should be tracked to fully assess reef health\n\n"
    
    # Add recommendation about prioritization
    implications_text += (f"• Conservation prioritization: {'Areas with high density likely also support high diversity and should be prioritized' if corr > 0.3 else 'Areas with high diversity may not always have high density - both metrics needed for site selection'}\n\n")
    
    # Add recommendation about climate change resilience
    implications_text += "• Climate resilience: Understanding how these metrics interact helps identify more resilient reef systems for protection"
    
    # Add text box with implications - larger padding for better readability
    props = dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.95, 
                edgecolor=COLORS['reef_green'], linewidth=2)
    
    ax7.text(0.5, 0.5, implications_text, transform=ax7.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Remove axes for this text panel
    ax7.axis('off')
    
    # Add a super title for the entire figure - slightly smaller and with more padding
    fig.suptitle('COMPREHENSIVE ANALYSIS: STONY CORAL DENSITY-RICHNESS RELATIONSHIP',
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                y=0.995)
    
    # Add a note about the data source
    fig.text(0.5, 0.005, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='bottom', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Don't use tight_layout as it can cause issues with overlapping elements
    # Instead use fig.subplots_adjust for more control
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.04, hspace=0.5, wspace=0.4)
    
    plt.savefig(os.path.join(results_dir, "stony_coral_density_richness_comprehensive_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Comprehensive analysis summary saved.")

if __name__ == "__main__":
    # Load and preprocess the data
    try:
        density_df, species_df, stations_df, density_species_cols, species_species_cols = load_and_preprocess_data()
        print("Data loaded and preprocessed successfully")
        
        # Analyze overall relationship
        overall_stats = analyze_overall_relationship(density_df)
        
        # Analyze by region and habitat
        region_stats = analyze_relationship_by_region(density_df)
        habitat_stats = analyze_relationship_by_habitat(density_df)
        
        # Analyze temporal relationship
        temporal_results = analyze_temporal_relationship(density_df)
        
        # Analyze spatial relationship if station data is available
        depth_stats = None
        if stations_df is not None:
            spatial_results = analyze_spatial_relationship(density_df, stations_df)
            
            # Add depth gradient analysis
            depth_stats = analyze_density_richness_relationship_depth_gradient(density_df, stations_df)
        
        # Create comprehensive analysis summary
        create_comprehensive_analysis(density_df, overall_stats, region_stats, habitat_stats, temporal_results, depth_stats)
        
        print("Analysis complete.")
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()