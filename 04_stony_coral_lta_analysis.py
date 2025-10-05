"""
04_stony_coral_lta_analysis.py - Analysis of Stony Coral Living Tissue Area Variations

This script analyzes the variations in stony coral living tissue area (LTA) across different
monitoring sites during the CREMP study period. It explores differences between sites,
regions, and habitat types, and performs statistical tests to determine significant variations.

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
from sklearn.linear_model import LinearRegression
import matplotlib.patheffects as pe  # For enhanced visual effects
from matplotlib.patches import Patch
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "04_Results"
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

# Create a custom colormap for coral reef visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP dataset for stony coral living tissue area analysis.
    
    Returns:
        tuple: (lta_df, stations_df, species_cols) - Preprocessed DataFrames and species columns
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Stony Coral LTA dataset
        lta_df = pd.read_csv("CREMP_CSV_files/CREMP_SCOR_Summaries_2023_LTA.csv")
        print(f"Stony coral LTA data loaded successfully with {len(lta_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Print column names to verify data structure
        print("\nStony coral LTA data columns (first 10):", lta_df.columns.tolist()[:10], "...")
        
        # Convert date column to datetime format
        lta_df['Date'] = pd.to_datetime(lta_df['Date'])
        
        # Extract just the year for easier grouping
        lta_df['Year'] = lta_df['Year'].astype(int)
        
        # Get list of all coral species columns (excluding metadata columns)
        metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID']
        species_cols = [col for col in lta_df.columns if col not in metadata_cols]
        
        print(f"\nIdentified {len(species_cols)} coral species in the dataset")
        
        # Calculate total LTA for each station
        lta_df['Total_LTA'] = lta_df[species_cols].sum(axis=1, skipna=True)
        
        print(f"\nData loaded: {len(lta_df)} records from {lta_df['Year'].min()} to {lta_df['Year'].max()}")
        
        return lta_df, stations_df, species_cols
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to analyze overall LTA distribution
def analyze_overall_lta_distribution(df):
    """
    Analyze and visualize the overall distribution of stony coral living tissue area.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Analyzing overall LTA distribution...")
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.suptitle('STONY CORAL LIVING TISSUE AREA DISTRIBUTION', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Plot 1: Histogram of Total LTA
    sns.histplot(df['Total_LTA'], bins=30, kde=True, color=COLORS['coral'], 
                 alpha=0.7, ax=ax1, edgecolor=COLORS['dark_blue'], linewidth=0.5)
    
    # Add vertical line for mean and median
    mean_lta = df['Total_LTA'].mean()
    median_lta = df['Total_LTA'].median()
    
    ax1.axvline(mean_lta, color=COLORS['dark_blue'], linestyle='--', linewidth=2, 
               label=f'Mean: {mean_lta:.2f}')
    ax1.axvline(median_lta, color=COLORS['ocean_blue'], linestyle='-.', linewidth=2, 
               label=f'Median: {median_lta:.2f}')
    
    # Set plot aesthetics
    ax1.set_title('Distribution of Total Living Tissue Area', 
                 fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Total Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Plot 2: Boxplot of Total LTA by Year
    yearly_data = df.groupby(['Year'])['Total_LTA'].apply(list).reset_index()
    boxplot_data = []
    years = []
    
    for _, row in yearly_data.iterrows():
        boxplot_data.append(row['Total_LTA'])
        years.append(str(row['Year']))
    
    # Create boxplot with enhanced styling
    bp = ax2.boxplot(boxplot_data, patch_artist=True, notch=True, showfliers=False)
    
    # Customize boxplot colors
    for box in bp['boxes']:
        box.set(facecolor=COLORS['ocean_blue'], alpha=0.7, edgecolor=COLORS['dark_blue'], linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color=COLORS['dark_blue'], linewidth=1.5, linestyle='-')
    for cap in bp['caps']:
        cap.set(color=COLORS['dark_blue'], linewidth=1.5)
    for median in bp['medians']:
        median.set(color=COLORS['coral'], linewidth=2)
    
    # Set plot aesthetics
    ax2.set_title('Yearly Distribution of Living Tissue Area', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Total Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-tick labels to years
    ax2.set_xticklabels(years, rotation=45, ha='right')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a summary textbox
    summary_text = (
        f"SUMMARY STATISTICS:\n"
        f"• Mean LTA: {mean_lta:.2f} mm²\n"
        f"• Median LTA: {median_lta:.2f} mm²\n"
        f"• Min LTA: {df['Total_LTA'].min():.2f} mm²\n"
        f"• Max LTA: {df['Total_LTA'].max():.2f} mm²\n"
        f"• Standard Deviation: {df['Total_LTA'].std():.2f} mm²"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box in a free space - adjusted position to avoid overlap
    ax1.text(0.95, 0.85, summary_text, transform=ax1.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_distribution.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Overall LTA distribution analysis saved.")

# Function to analyze LTA by site
def analyze_lta_by_site(df):
    """
    Analyze and visualize the living tissue area variations by site.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Analyzing LTA by site...")
    
    # Group by site and calculate mean LTA
    site_lta = df.groupby('Site_name')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
    site_lta['se'] = site_lta['std'] / np.sqrt(site_lta['count'])
    site_lta['ci_95'] = 1.96 * site_lta['se']
    
    # Sort by mean LTA for better visualization
    site_lta = site_lta.sort_values('mean', ascending=False)
    
    # Take top 20 sites for better visualization
    top_sites = site_lta.head(20)
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create bar plot with error bars
    bars = ax.bar(top_sites['Site_name'], top_sites['mean'], 
                 yerr=top_sites['ci_95'], 
                 color=COLORS['ocean_blue'], 
                 alpha=0.8,
                 edgecolor=COLORS['dark_blue'],
                 linewidth=1.5,
                 error_kw={'ecolor': COLORS['dark_blue'], 'capsize': 5, 'capthick': 2})
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                rotation=0, color=COLORS['dark_blue'])
    
    # Set plot aesthetics
    ax.set_title('TOP 20 SITES BY MEAN LIVING TISSUE AREA', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Site Name', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Perform ANOVA to test for significant differences between sites
    site_groups = df.groupby('Site_name')['Total_LTA'].apply(list).reset_index()
    anova_data = [group for group in site_groups['Total_LTA'] if len(group) >= 5]  # Only include sites with sufficient data
    
    if len(anova_data) >= 2:  # Need at least 2 groups for ANOVA
        f_stat, p_value = f_oneway(*anova_data)
        
        # Add ANOVA results to the plot
        anova_text = (
            f"ANOVA Results:\n"
            f"F-statistic: {f_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add the ANOVA results box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax.text(0.02, 0.95, anova_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_by_site.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("LTA by site analysis saved.")
    
    return site_lta

# Function to analyze LTA by region
def analyze_lta_by_region(df):
    """
    Analyze and visualize the living tissue area variations by region.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Analyzing LTA by region...")
    
    # Group by region and calculate statistics
    region_lta = df.groupby('Subregion')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
    region_lta['se'] = region_lta['std'] / np.sqrt(region_lta['count'])
    region_lta['ci_95'] = 1.96 * region_lta['se']
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.suptitle('STONY CORAL LIVING TISSUE AREA BY REGION', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Define region color mapping
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
    
    # Plot 1: Bar chart of mean LTA by region
    bars = ax1.bar(region_lta['Subregion'].map(region_names), region_lta['mean'], 
                  yerr=region_lta['ci_95'],
                  color=[region_colors.get(region, COLORS['coral']) for region in region_lta['Subregion']],
                  alpha=0.8,
                  edgecolor=COLORS['dark_blue'],
                  linewidth=1.5,
                  error_kw={'ecolor': COLORS['dark_blue'], 'capsize': 5, 'capthick': 2})
    
    # Add data labels on top of bars - adjusted vertical position to prevent overlap
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 3000,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                rotation=0, color=COLORS['dark_blue'])
    
    # Set plot aesthetics
    ax1.set_title('Mean Living Tissue Area by Region', 
                 fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 2: Boxplot of LTA by region
    region_data = df.groupby(['Subregion'])['Total_LTA'].apply(list).reset_index()
    boxplot_data = []
    regions = []
    
    for _, row in region_data.iterrows():
        boxplot_data.append(row['Total_LTA'])
        regions.append(region_names.get(row['Subregion'], row['Subregion']))
    
    # Create boxplot with enhanced styling
    bp = ax2.boxplot(boxplot_data, patch_artist=True, notch=True, showfliers=False)
    
    # Customize boxplot colors based on region
    for i, box in enumerate(bp['boxes']):
        region_code = region_data.iloc[i]['Subregion']
        box.set(facecolor=region_colors.get(region_code, COLORS['coral']), 
                alpha=0.7, 
                edgecolor=COLORS['dark_blue'], 
                linewidth=1.5)
    
    for whisker in bp['whiskers']:
        whisker.set(color=COLORS['dark_blue'], linewidth=1.5, linestyle='-')
    for cap in bp['caps']:
        cap.set(color=COLORS['dark_blue'], linewidth=1.5)
    for median in bp['medians']:
        median.set(color=COLORS['coral'], linewidth=2)
    
    # Set plot aesthetics
    ax2.set_title('Distribution of Living Tissue Area by Region', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-tick labels to regions
    ax2.set_xticklabels(regions, fontweight='bold')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Perform statistical test to compare regions
    if len(boxplot_data) >= 2:  # Need at least 2 groups for comparison
        # Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
        h_stat, p_value = kruskal(*boxplot_data)
        
        # Add test results to the plot
        test_text = (
            f"Kruskal-Wallis Test:\n"
            f"H-statistic: {h_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add the test results box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax2.text(0.05, 0.95, test_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_by_region.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("LTA by region analysis saved.")
    
    return region_lta

# Function to analyze LTA by habitat
def analyze_lta_by_habitat(df):
    """
    Analyze and visualize the living tissue area variations by habitat type.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Analyzing LTA by habitat...")
    
    # Group by habitat and calculate statistics
    habitat_lta = df.groupby('Habitat')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
    habitat_lta['se'] = habitat_lta['std'] / np.sqrt(habitat_lta['count'])
    habitat_lta['ci_95'] = 1.96 * habitat_lta['se']
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.suptitle('STONY CORAL LIVING TISSUE AREA BY HABITAT TYPE', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # Define habitat color mapping
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
    
    # Plot 1: Bar chart of mean LTA by habitat
    bars = ax1.bar(habitat_lta['Habitat'].map(habitat_names), habitat_lta['mean'], 
                  yerr=habitat_lta['ci_95'],
                  color=[habitat_colors.get(habitat, COLORS['coral']) for habitat in habitat_lta['Habitat']],
                  alpha=0.8,
                  edgecolor=COLORS['dark_blue'],
                  linewidth=1.5,
                  error_kw={'ecolor': COLORS['dark_blue'], 'capsize': 5, 'capthick': 2})
    
    # Add data labels on top of bars - adjusted vertical position to prevent overlap
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 3000,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                rotation=0, color=COLORS['dark_blue'])
    
    # Set plot aesthetics
    ax1.set_title('Mean Living Tissue Area by Habitat Type', 
                 fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Mean Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 2: Boxplot of LTA by habitat
    habitat_data = df.groupby(['Habitat'])['Total_LTA'].apply(list).reset_index()
    boxplot_data = []
    habitats = []
    
    for _, row in habitat_data.iterrows():
        boxplot_data.append(row['Total_LTA'])
        habitats.append(habitat_names.get(row['Habitat'], row['Habitat']))
    
    # Create boxplot with enhanced styling
    bp = ax2.boxplot(boxplot_data, patch_artist=True, notch=True, showfliers=False)
    
    # Customize boxplot colors based on habitat
    for i, box in enumerate(bp['boxes']):
        habitat_code = habitat_data.iloc[i]['Habitat']
        box.set(facecolor=habitat_colors.get(habitat_code, COLORS['coral']), 
                alpha=0.7, 
                edgecolor=COLORS['dark_blue'], 
                linewidth=1.5)
    
    for whisker in bp['whiskers']:
        whisker.set(color=COLORS['dark_blue'], linewidth=1.5, linestyle='-')
    for cap in bp['caps']:
        cap.set(color=COLORS['dark_blue'], linewidth=1.5)
    for median in bp['medians']:
        median.set(color=COLORS['coral'], linewidth=2)
    
    # Set plot aesthetics
    ax2.set_title('Distribution of Living Tissue Area by Habitat Type', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set x-tick labels to habitats
    ax2.set_xticklabels(habitats, fontweight='bold')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Perform statistical test to compare habitats
    if len(boxplot_data) >= 2:  # Need at least 2 groups for comparison
        # Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
        h_stat, p_value = kruskal(*boxplot_data)
        
        # Add test results to the plot
        test_text = (
            f"Kruskal-Wallis Test:\n"
            f"H-statistic: {h_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add the test results box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax2.text(0.05, 0.95, test_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_by_habitat.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("LTA by habitat analysis saved.")
    
    return habitat_lta

# Function to create a heatmap of LTA by site and species
def create_species_site_heatmap(df, species_cols):
    """
    Create a heatmap visualization of living tissue area by site and species.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
        species_cols (list): List of species column names
    """
    print("Creating species-site heatmap...")
    
    # Calculate mean LTA for each species at each site
    # First group by site
    site_species_lta = df.groupby('Site_name')[species_cols].mean().reset_index()
    
    # Select top species by overall mean LTA for better visualization
    species_means = df[species_cols].mean().sort_values(ascending=False)
    top_species = species_means.head(15).index.tolist()
    
    # Select top sites by total LTA for better visualization
    site_total_lta = df.groupby('Site_name')['Total_LTA'].mean().sort_values(ascending=False)
    top_sites = site_total_lta.head(20).index.tolist()
    
    # Filter data for top species and sites
    heatmap_data = site_species_lta[site_species_lta['Site_name'].isin(top_sites)]
    heatmap_data = heatmap_data[['Site_name'] + top_species]
    
    # Pivot data for heatmap
    heatmap_pivot = heatmap_data.set_index('Site_name')
    
    # Create figure with enhanced styling
    plt.figure(figsize=(16, 12), facecolor=COLORS['background'])
    
    # Create the heatmap with improved styling
    ax = sns.heatmap(
        heatmap_pivot, 
        cmap=coral_cmap, 
        annot=True, 
        fmt='.0f',
        linewidths=0.5, 
        cbar_kws={'label': 'Mean Living Tissue Area (mm²)'}
    )
    
    # Set plot aesthetics
    plt.title('MEAN LIVING TISSUE AREA BY SITE AND SPECIES', 
             fontweight='bold', fontsize=20, pad=20,
             color=COLORS['dark_blue'],
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    plt.xlabel('Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    plt.ylabel('Site Name', fontweight='bold', fontsize=14, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    
    # Add a note about the data source
    plt.figtext(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
               ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_species_site_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species-site heatmap saved.")

# Function to analyze temporal trends in LTA
def analyze_temporal_trends(df):
    """
    Analyze and visualize temporal trends in living tissue area.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Analyzing temporal trends in LTA...")
    
    # Group by year and calculate statistics
    yearly_lta = df.groupby('Year')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
    yearly_lta['se'] = yearly_lta['std'] / np.sqrt(yearly_lta['count'])
    yearly_lta['ci_95'] = 1.96 * yearly_lta['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the mean line with enhanced styling
    ax.plot(yearly_lta['Year'], yearly_lta['mean'], marker='o', color=COLORS['coral'], 
            linewidth=3.5, markersize=10, label='Mean Living Tissue Area',
            path_effects=[pe.SimpleLineShadow(offset=(2, -2), alpha=0.3), pe.Normal()])
    
    # Add 95% confidence interval band with enhanced styling
    ax.fill_between(yearly_lta['Year'], 
                    yearly_lta['mean'] - yearly_lta['ci_95'],
                    yearly_lta['mean'] + yearly_lta['ci_95'],
                    color=COLORS['coral'], alpha=0.2, label='95% Confidence Interval')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot aesthetics
    ax.set_title('EVOLUTION OF STONY CORAL LIVING TISSUE AREA (2011-2023)', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set axis limits with more space for annotations
    ax.set_xlim(yearly_lta['Year'].min() - 1, yearly_lta['Year'].max() + 1)
    y_min = max(0, yearly_lta['mean'].min() - yearly_lta['ci_95'].max() - 5000)
    y_max = yearly_lta['mean'].max() + yearly_lta['ci_95'].max() + 5000
    ax.set_ylim(y_min, y_max)
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    legend.get_frame().set_linewidth(1.5)
    
    # Add text annotations for key events in coral reef history with improved positioning
    key_events = {
        2014: {"name": "2014-2015 Global Bleaching Event", "pos": "bottom", "offset": -0.8},
        2017: {"name": "Hurricane Irma", "pos": "top", "offset": 0.8},
        2019: {"name": "Stony Coral Tissue Loss Disease Peak", "pos": "bottom", "offset": -0.8}
    }
    
    # Calculate available vertical space
    y_range = y_max - y_min
    top_space = y_range * 0.2  # Use 20% of the range for top annotations
    bottom_space = y_range * 0.2  # Use 20% of the range for bottom annotations
    
    # Position counter to track used positions
    top_positions_used = 0
    bottom_positions_used = 0
    
    for year, event_info in key_events.items():
        if year >= yearly_lta['Year'].min() and year <= yearly_lta['Year'].max():
            # Find the data point for this year
            year_data = yearly_lta[yearly_lta['Year'] == year]
            
            if not year_data.empty:
                y_val = year_data['mean'].values[0]
                
                # Add vertical line for the event
                ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5)
                
                # Calculate position based on specified position and available space
                if event_info["pos"] == "top":
                    # For top positions, start from the highest value and work down
                    offset = top_positions_used * (top_space / len(key_events)) + 0.5
                    y_pos = y_val + offset
                    top_positions_used += 1
                else:
                    # For bottom positions, start from the lowest value and work up
                    offset = bottom_positions_used * (bottom_space / len(key_events)) + 0.5
                    y_pos = y_val - offset
                    bottom_positions_used += 1
                
                # Add text annotation with enhanced styling
                ax.annotate(
                    event_info["name"],
                    xy=(year, y_val),
                    xytext=(year, y_pos),
                    arrowprops=dict(
                        arrowstyle='->', 
                        connectionstyle='arc3,rad=0.15', 
                        color='gray', 
                        lw=1.5
                    ),
                    ha='center',
                    va='center' if event_info["pos"] == "top" else 'center',
                    fontsize=11,
                    fontweight='bold',
                    color=COLORS['text'],
                    bbox=dict(
                        boxstyle='round,pad=0.4', 
                        facecolor='white', 
                        alpha=0.9, 
                        edgecolor=COLORS['coral'],
                        linewidth=1.5
                    )
                )
    
    # Add a trend line (linear regression)
    X = yearly_lta['Year'].values.reshape(-1, 1)
    y = yearly_lta['mean'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate slope and plot trend line
    slope = model.coef_[0]
    ax.plot(yearly_lta['Year'], y_pred, '--', color=COLORS['dark_blue'], 
            linewidth=2, label=f'Linear Trend (Slope: {slope:.2f} mm²/year)',
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
    
    # Update legend with the new trend line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, facecolor='white', framealpha=0.9, 
             fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Add a summary textbox
    summary_text = (
        f"SUMMARY:\n"
        f"• Overall trend: {slope:.2f} mm² per year\n"
        f"• Maximum LTA: {yearly_lta['mean'].max():.2f} mm² ({yearly_lta.loc[yearly_lta['mean'].idxmax(), 'Year']})\n"
        f"• Minimum LTA: {yearly_lta['mean'].min():.2f} mm² ({yearly_lta.loc[yearly_lta['mean'].idxmin(), 'Year']})\n"
        f"• Current level (2023): {yearly_lta.loc[yearly_lta['Year'] == yearly_lta['Year'].max(), 'mean'].values[0]:.2f} mm²"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box in a free space (lower left)
    ax.text(0.02, 0.05, summary_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_temporal_trend.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal trends analysis saved.")

# Function to perform statistical analysis of LTA differences
def perform_statistical_analysis(df):
    """
    Perform detailed statistical analysis to determine significant differences in LTA.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral LTA data
    """
    print("Performing statistical analysis of LTA differences...")
    
    # Create a figure to visualize statistical results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.suptitle('STATISTICAL ANALYSIS OF LIVING TISSUE AREA DIFFERENCES', 
                fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set background color
    ax1.set_facecolor(COLORS['background'])
    ax2.set_facecolor(COLORS['background'])
    
    # 1. Region Analysis - ANOVA
    region_model = ols('Total_LTA ~ C(Subregion)', data=df).fit()
    region_anova_table = sm.stats.anova_lm(region_model, typ=2)
    
    # 2. Habitat Analysis - ANOVA
    habitat_model = ols('Total_LTA ~ C(Habitat)', data=df).fit()
    habitat_anova_table = sm.stats.anova_lm(habitat_model, typ=2)
    
    # 3. Year Analysis - ANOVA
    year_model = ols('Total_LTA ~ C(Year)', data=df).fit()
    year_anova_table = sm.stats.anova_lm(year_model, typ=2)
    
    # Create a bar chart of F-statistics for different factors
    factors = ['Region', 'Habitat', 'Year']
    f_values = [region_anova_table.loc['C(Subregion)', 'F'], 
                habitat_anova_table.loc['C(Habitat)', 'F'],
                year_anova_table.loc['C(Year)', 'F']]
    p_values = [region_anova_table.loc['C(Subregion)', 'PR(>F)'], 
                habitat_anova_table.loc['C(Habitat)', 'PR(>F)'],
                year_anova_table.loc['C(Year)', 'PR(>F)']]
    
    # Plot F-statistics
    bars = ax1.bar(factors, f_values, color=[COLORS['dark_blue'], COLORS['ocean_blue'], COLORS['coral']], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                rotation=0, color=COLORS['dark_blue'])
    
    # Set plot aesthetics
    ax1.set_title('F-Statistics from ANOVA Tests', 
                 fontweight='bold', fontsize=16, pad=15)
    ax1.set_xlabel('Factor', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('F-Statistic', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot p-values
    bars = ax2.bar(factors, p_values, color=[COLORS['dark_blue'], COLORS['ocean_blue'], COLORS['coral']], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add significance threshold line
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (p=0.05)')
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                rotation=0, color=COLORS['dark_blue'])
    
    # Set plot aesthetics
    ax2.set_title('P-Values from ANOVA Tests', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Factor', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('P-Value', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add legend
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
              fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Add a summary textbox
    summary_text = (
        f"ANOVA RESULTS SUMMARY:\n\n"
        f"Region Analysis:\n"
        f"• F-statistic: {region_anova_table.loc['C(Subregion)', 'F']:.2f}\n"
        f"• p-value: {region_anova_table.loc['C(Subregion)', 'PR(>F)']:.4f}\n"
        f"• Significant: {'Yes' if region_anova_table.loc['C(Subregion)', 'PR(>F)'] < 0.05 else 'No'}\n\n"
        f"Habitat Analysis:\n"
        f"• F-statistic: {habitat_anova_table.loc['C(Habitat)', 'F']:.2f}\n"
        f"• p-value: {habitat_anova_table.loc['C(Habitat)', 'PR(>F)']:.4f}\n"
        f"• Significant: {'Yes' if habitat_anova_table.loc['C(Habitat)', 'PR(>F)'] < 0.05 else 'No'}\n\n"
        f"Year Analysis:\n"
        f"• F-statistic: {year_anova_table.loc['C(Year)', 'F']:.2f}\n"
        f"• p-value: {year_anova_table.loc['C(Year)', 'PR(>F)']:.4f}\n"
        f"• Significant: {'Yes' if year_anova_table.loc['C(Year)', 'PR(>F)'] < 0.05 else 'No'}"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box to the right side of the figure to avoid overlap with plots
    fig.text(0.78, 0.5, summary_text, fontsize=11, fontweight='bold',
           verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_lta_statistical_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Statistical analysis saved.")
    
    # Return the ANOVA tables for further analysis if needed
    return {
        'region_anova': region_anova_table,
        'habitat_anova': habitat_anova_table,
        'year_anova': year_anova_table
    }

# Main execution
def main():
    """Main execution function."""
    print("\n=== Stony Coral Living Tissue Area Analysis ===\n")
    
    # Load and preprocess data
    lta_df, stations_df, species_cols = load_and_preprocess_data()
    
    # Analyze overall LTA distribution
    analyze_overall_lta_distribution(lta_df)
    
    # Analyze LTA by site
    site_lta = analyze_lta_by_site(lta_df)
    
    # Analyze LTA by region
    region_lta = analyze_lta_by_region(lta_df)
    
    # Analyze LTA by habitat
    habitat_lta = analyze_lta_by_habitat(lta_df)
    
    # Create species-site heatmap
    create_species_site_heatmap(lta_df, species_cols)
    
    # Analyze temporal trends
    analyze_temporal_trends(lta_df)
    
    # Perform statistical analysis
    anova_results = perform_statistical_analysis(lta_df)
    
    print("\n=== Analysis Complete ===\n")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()