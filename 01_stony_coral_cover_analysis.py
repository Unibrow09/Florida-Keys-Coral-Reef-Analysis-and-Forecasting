"""
01_stony_coral_cover_analysis.py - Analysis of Stony Coral Percentage Cover Evolution

This script analyzes the evolution of stony coral percentage cover across different
stations during the CREMP study period (1996-2023). It explores trends over time
by region, habitat type, and at specific key sites.

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

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "01_Results"
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
    Load and preprocess the CREMP dataset for analysis.
    
    Returns:
        tuple: (taxa_groups_df, stations_df) - Preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Taxa Groups dataset (contains stony coral percentage)
        taxa_groups_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_TaxaGroups.csv")
        print(f"Taxa groups data loaded successfully with {len(taxa_groups_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Print column names to verify data structure
        print("\nTaxa groups columns:", taxa_groups_df.columns.tolist())
        print("\nFirst few rows of stony coral data:")
        print(taxa_groups_df[['Year', 'Subregion', 'Habitat', 'Site_name', 'Stony_coral']].head())
        
        # Convert proportion to actual percentage
        taxa_groups_df['Stony_coral'] = taxa_groups_df['Stony_coral'] * 100
        
        # Convert date column to datetime format
        taxa_groups_df['Date'] = pd.to_datetime(taxa_groups_df['Date'])
        
        # Extract just the year for easier grouping
        taxa_groups_df['Year'] = taxa_groups_df['Year'].astype(int)
        
        print(f"\nData loaded: {len(taxa_groups_df)} records from {taxa_groups_df['Year'].min()} to {taxa_groups_df['Year'].max()}")
        
        return taxa_groups_df, stations_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to plot overall trend of stony coral cover over time
def plot_overall_trend(df):
    """
    Plot the overall trend of stony coral percentage cover over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating overall trend plot...")
    
    # Group by year and calculate mean stony coral cover
    yearly_avg = df.groupby('Year')['Stony_coral'].agg(['mean', 'std', 'count']).reset_index()
    yearly_avg['se'] = yearly_avg['std'] / np.sqrt(yearly_avg['count'])
    yearly_avg['ci_95'] = 1.96 * yearly_avg['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the mean line with enhanced styling
    ax.plot(yearly_avg['Year'], yearly_avg['mean'], marker='o', color=COLORS['coral'], 
            linewidth=3.5, markersize=10, label='Mean Stony Coral Cover',
            path_effects=[pe.SimpleLineShadow(offset=(2, -2), alpha=0.3), pe.Normal()])
    
    # Add 95% confidence interval band with enhanced styling
    ax.fill_between(yearly_avg['Year'], 
                    yearly_avg['mean'] - yearly_avg['ci_95'],
                    yearly_avg['mean'] + yearly_avg['ci_95'],
                    color=COLORS['coral'], alpha=0.2, label='95% Confidence Interval')
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot aesthetics
    ax.set_title('EVOLUTION OF STONY CORAL PERCENTAGE COVER (1996-2023)', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Stony Coral Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Set axis limits with more space for annotations
    ax.set_xlim(yearly_avg['Year'].min() - 1, yearly_avg['Year'].max() + 1)
    y_min = max(0, yearly_avg['mean'].min() - yearly_avg['ci_95'].max() - 1)
    y_max = yearly_avg['mean'].max() + yearly_avg['ci_95'].max() + 2
    ax.set_ylim(y_min, y_max)
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    legend.get_frame().set_linewidth(1.5)
    
    # Add text annotations for key events in coral reef history with improved positioning
    key_events = {
        1998: {"name": "1998 Global Coral Bleaching Event", "pos": "top", "offset": 0.8},
        2005: {"name": "2005 Caribbean Bleaching Event", "pos": "bottom", "offset": -0.8},
        2010: {"name": "2010 Cold Water Mortality Event", "pos": "top", "offset": 0.8},
        2014: {"name": "2014-2015 Global Bleaching Event", "pos": "bottom", "offset": -0.8},
        2017: {"name": "Hurricane Irma", "pos": "top", "offset": 0.8}
    }
    
    # Calculate available vertical space
    y_range = y_max - y_min
    top_space = y_range * 0.2  # Use 20% of the range for top annotations
    bottom_space = y_range * 0.2  # Use 20% of the range for bottom annotations
    
    # Position counter to track used positions
    top_positions_used = 0
    bottom_positions_used = 0
    
    for year, event_info in key_events.items():
        if year >= yearly_avg['Year'].min() and year <= yearly_avg['Year'].max():
            # Find the data point for this year
            year_data = yearly_avg[yearly_avg['Year'] == year]
            
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
    X = yearly_avg['Year'].values.reshape(-1, 1)
    y = yearly_avg['mean'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate slope and plot trend line
    slope = model.coef_[0]
    ax.plot(yearly_avg['Year'], y_pred, '--', color=COLORS['dark_blue'], 
            linewidth=2, label=f'Linear Trend (Slope: {slope:.4f}%/year)',
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
    
    # Update legend with the new trend line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, facecolor='white', framealpha=0.9, 
             fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Add a summary textbox
    summary_text = (
        f"SUMMARY:\n"
        f"• Overall trend: {slope:.4f}% per year\n"
        f"• Maximum cover: {yearly_avg['mean'].max():.2f}% ({yearly_avg.loc[yearly_avg['mean'].idxmax(), 'Year']})\n"
        f"• Minimum cover: {yearly_avg['mean'].min():.2f}% ({yearly_avg.loc[yearly_avg['mean'].idxmin(), 'Year']})\n"
        f"• Current level (2023): {yearly_avg.loc[yearly_avg['Year'] == 2023, 'mean'].values[0]:.2f}%"
    )
    
    # Add the summary box with enhanced styling
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=2)
    
    # Position the text box in a free space (lower left)
    ax.text(0.02, 0.05, summary_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add a subtle background element for visual interest
    fig.text(0.5, 0.5, 'CREMP Data', fontsize=100, color='gray', alpha=0.05,
            ha='center', va='center', rotation=30, transform=fig.transFigure)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_overall_trend.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Overall trend plot saved.")

# Function to plot trends by region
def plot_trends_by_region(df):
    """
    Plot stony coral cover trends separated by region (Upper Keys, Middle Keys, Lower Keys).
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating regional trends plot...")
    
    # Group by year and region, calculate mean stony coral cover
    region_yearly_avg = df.groupby(['Year', 'Subregion'])['Stony_coral'].mean().reset_index()
    
    # Plot trends by region
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    
    # Plot each region
    for region, color in region_colors.items():
        region_data = region_yearly_avg[region_yearly_avg['Subregion'] == region]
        if not region_data.empty:
            ax.plot(region_data['Year'], region_data['Stony_coral'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=region_names.get(region, region))
    
    # Set plot aesthetics
    ax.set_title('Stony Coral Cover by Region (1996-2023)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Stony Coral Cover (%)')
    
    # Set axis limits
    ax.set_xlim(region_yearly_avg['Year'].min() - 0.5, region_yearly_avg['Year'].max() + 0.5)
    ax.set_ylim(0, region_yearly_avg['Stony_coral'].max() * 1.1)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, title='Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_regional_trends.png"), bbox_inches='tight')
    plt.close()
    
    print("Regional trends plot saved.")

# Function to plot trends by habitat type
def plot_trends_by_habitat(df):
    """
    Plot stony coral cover trends separated by habitat type.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating habitat-based trends plot...")
    
    # Group by year and habitat, calculate mean stony coral cover
    habitat_yearly_avg = df.groupby(['Year', 'Habitat'])['Stony_coral'].mean().reset_index()
    
    # Plot trends by habitat
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    
    # Plot each habitat
    for habitat, color in habitat_colors.items():
        habitat_data = habitat_yearly_avg[habitat_yearly_avg['Habitat'] == habitat]
        if not habitat_data.empty:
            ax.plot(habitat_data['Year'], habitat_data['Stony_coral'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=habitat_names.get(habitat, habitat))
    
    # Set plot aesthetics
    ax.set_title('Stony Coral Cover by Habitat Type (1996-2023)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Stony Coral Cover (%)')
    
    # Set axis limits
    ax.set_xlim(habitat_yearly_avg['Year'].min() - 0.5, habitat_yearly_avg['Year'].max() + 0.5)
    ax.set_ylim(0, habitat_yearly_avg['Stony_coral'].max() * 1.1)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, title='Habitat Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_habitat_trends.png"), bbox_inches='tight')
    plt.close()
    
    print("Habitat-based trends plot saved.")

# Function to create a heatmap of stony coral cover trends
def create_temporal_heatmap(df):
    """
    Create a heatmap visualization of stony coral cover changes over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating temporal heatmap...")
    
    # Get a sample of sites to visualize (to avoid overcrowding)
    # Focus on sites that have been surveyed for the longest periods
    site_survey_counts = df.groupby('Site_name')['Year'].nunique().sort_values(ascending=False)
    sites_to_include = site_survey_counts[site_survey_counts > 15].index.tolist()[:15]  # Reduced to top 15 most surveyed sites
    
    # Filter data for these sites and pivot to create a matrix suitable for a heatmap
    heatmap_data = df[df['Site_name'].isin(sites_to_include)]
    
    # Calculate average for each site and year
    heatmap_pivot = heatmap_data.groupby(['Site_name', 'Year'])['Stony_coral'].mean().reset_index()
    heatmap_pivot = heatmap_pivot.pivot(index='Site_name', columns='Year', values='Stony_coral')
    
    # Sort sites by their habitat type and region for better visualization
    site_info = df[df['Site_name'].isin(sites_to_include)][['Site_name', 'Habitat', 'Subregion']].drop_duplicates()
    # Sort first by habitat, then by region, then by site name for better organization
    site_order = site_info.sort_values(['Habitat', 'Subregion', 'Site_name']).Site_name.tolist()
    heatmap_pivot = heatmap_pivot.reindex(site_order)
    
    # Create the heatmap with enhanced styling
    fig, ax = plt.subplots(figsize=(18, 12), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Generate the heatmap with improved styling
    heatmap = sns.heatmap(
        heatmap_pivot, 
        cmap=coral_cmap, 
        ax=ax, 
        vmin=0, 
        vmax=max(15, heatmap_pivot.max().max()),  # Set reasonable limits
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={
            'label': 'Stony Coral Cover (%)', 
            'shrink': 0.8,
            'aspect': 20,
            'pad': 0.01,
            'orientation': 'horizontal',
            'location': 'top'
        },
        annot=True,  # Add value annotations
        annot_kws={'size': 9, 'weight': 'bold', 'color': 'black'},
        fmt='.1f'  # Format to 1 decimal place
    )
    
    # Improve the colorbar appearance
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Stony Coral Cover (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set plot aesthetics
    ax.set_title('Temporal Evolution of Stony Coral Cover by Site (1996-2023)', 
                fontweight='bold', pad=20, fontsize=20,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Improve X and Y labels
    ax.set_xlabel('Year', labelpad=15, fontweight='bold', fontsize=14)
    ax.set_ylabel('Site Name', labelpad=15, fontweight='bold', fontsize=14)
    
    # Improve tick labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    # Create a separate axis for the site metadata (habitat and region)
    # This prevents overlapping with the main heatmap
    metadata_width = 0.15  # Width of the metadata column
    ax_meta = fig.add_axes([ax.get_position().x0 - metadata_width, 
                           ax.get_position().y0, 
                           metadata_width, 
                           ax.get_position().height])
    ax_meta.set_facecolor(COLORS['background'])
    
    # Remove axes elements for the metadata axis
    ax_meta.set_xticks([])
    ax_meta.set_yticks([])
    for spine in ax_meta.spines.values():
        spine.set_visible(False)
    
    # Add habitat and region information on the left side
    for i, site in enumerate(heatmap_pivot.index):
        site_row = site_info[site_info['Site_name'] == site].iloc[0]
        habitat = site_row['Habitat']
        region = site_row['Subregion']
        
        # Map habitat codes to full names
        habitat_names = {
            'OS': 'Offshore Shallow',
            'OD': 'Offshore Deep',
            'P': 'Patch Reef',
            'HB': 'Hardbottom',
            'BCP': 'Backcountry Patch'
        }
        
        # Map region codes to full names
        region_names = {
            'UK': 'Upper Keys',
            'MK': 'Middle Keys',
            'LK': 'Lower Keys'
        }
        
        # Get full names
        habitat_full = habitat_names.get(habitat, habitat)
        region_full = region_names.get(region, region)
        
        # Set colors based on habitat and region
        habitat_colors = {
            'OS': COLORS['coral'],
            'OD': COLORS['sand'],
            'P': COLORS['reef_green'],
            'HB': COLORS['ocean_blue'],
            'BCP': COLORS['dark_blue']
        }
        
        # Use colored rectangular patches for habitat
        rect = plt.Rectangle(
            (0.1, i + 0.25), 0.3, 0.5, 
            facecolor=habitat_colors.get(habitat, 'gray'),
            alpha=0.8,
            transform=ax_meta.transData,
            edgecolor='white',
            linewidth=1
        )
        ax_meta.add_patch(rect)
        
        # Add text labels for habitat and region
        ax_meta.text(0.5, i + 0.5, f"{habitat_full}", 
                   va='center', ha='left', 
                   fontsize=10, fontweight='bold',
                   color='white', transform=ax_meta.transData,
                   path_effects=[pe.withStroke(linewidth=1, foreground='black')])
        
        ax_meta.text(0.5, i + 0.8, f"{region_full}", 
                   va='bottom', ha='left', 
                   fontsize=9, fontweight='normal',
                   color=COLORS['text'], transform=ax_meta.transData)
    
    # Add a title for the metadata column
    ax_meta.text(0.5, len(heatmap_pivot.index) + 0.5, "HABITAT & REGION", 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color=COLORS['dark_blue'], transform=ax_meta.transData)
    
    # Add a subtle grid to the metadata section
    for i in range(len(heatmap_pivot.index) + 1):
        ax_meta.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "stony_coral_temporal_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal heatmap saved.")

# Function to analyze rate of change in stony coral cover
def analyze_rate_of_change(df):
    """
    Analyze and visualize the rate of change in stony coral cover over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Analyzing rate of change...")
    
    # Group by year and calculate mean stony coral cover
    yearly_avg = df.groupby('Year')['Stony_coral'].mean().reset_index()
    
    # Calculate year-over-year change
    yearly_avg['Change'] = yearly_avg['Stony_coral'].diff()
    yearly_avg['Percent_Change'] = yearly_avg['Stony_coral'].pct_change() * 100
    
    # Filter out the first year which will have NaN for change
    yearly_avg = yearly_avg.dropna()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot absolute change
    bars1 = ax1.bar(yearly_avg['Year'], yearly_avg['Change'], color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['Change']], 
             alpha=0.7, width=0.7)
    
    ax1.set_title('Absolute Change in Stony Coral Cover (Year-over-Year)', fontweight='bold')
    ax1.set_ylabel('Change in Coral Cover (%)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot percent change
    bars2 = ax2.bar(yearly_avg['Year'], yearly_avg['Percent_Change'], color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['Percent_Change']], 
             alpha=0.7, width=0.7)
    
    ax2.set_title('Percentage Change in Stony Coral Cover (Year-over-Year)', fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percent Change (%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on top of bars
    def add_labels(bars, ax, format_str='{:.1f}'):
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:  # Only label bars with significant values
                ax.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.3 if height > 0 else -0.7),
                        format_str.format(height), 
                        ha='center', va='bottom' if height > 0 else 'top', 
                        color=COLORS['text'], fontsize=9, fontweight='bold')
    
    add_labels(bars1, ax1)
    add_labels(bars2, ax2)
    
    # Label the significant events
    key_events = {
        1998: "1998 Bleaching",
        2005: "2005 Bleaching",
        2010: "2010 Cold Snap",
        2014: "2014-15 Bleaching",
        2017: "Hurricane Irma"
    }
    
    for year, event in key_events.items():
        if year in yearly_avg['Year'].values:
            year_idx = yearly_avg[yearly_avg['Year'] == year].index[0]
            change = yearly_avg.loc[year_idx, 'Change']
            ax1.annotate(event, xy=(year, change), xytext=(0, 20 if change > 0 else -20),
                         textcoords='offset points', ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_rate_of_change.png"), bbox_inches='tight')
    plt.close()
    
    print("Rate of change analysis saved.")

# Function to plot regional comparison
def plot_regional_comparison(df):
    """
    Create a comparative box plot of stony coral cover by region and time period.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating regional comparison plot...")
    
    # Create time periods for comparison
    df['Period'] = pd.cut(df['Year'], 
                          bins=[1995, 2000, 2005, 2010, 2015, 2023],
                          labels=['1996-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2023'],
                          right=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define color palette for regions
    region_colors = {'UK': COLORS['dark_blue'], 'MK': COLORS['ocean_blue'], 'LK': COLORS['light_blue']}
    
    # Create the boxplot
    sns.boxplot(x='Period', y='Stony_coral', hue='Subregion', data=df, palette=region_colors,
                ax=ax, width=0.7, fliersize=3, linewidth=1.2)
    
    # Overlay a swarm plot for actual data points
    sns.swarmplot(x='Period', y='Stony_coral', hue='Subregion', data=df.sample(frac=0.1, random_state=42),  # Sample to avoid overcrowding
                  dodge=True, size=3, edgecolor='black', linewidth=0.5, ax=ax, alpha=0.6)
    
    # Set plot aesthetics
    ax.set_title('Regional Comparison of Stony Coral Cover Over Time', fontweight='bold')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Stony Coral Cover (%)')
    
    # Fix the legend
    handles, labels = ax.get_legend_handles_labels()
    region_labels = {'UK': 'Upper Keys', 'MK': 'Middle Keys', 'LK': 'Lower Keys'}
    unique_labels = dict(zip(labels[:3], [region_labels.get(l, l) for l in labels[:3]]))
    ax.legend(handles[:3], unique_labels.values(), title='Region', frameon=True, 
              facecolor='white', framealpha=0.9, loc='upper right')
    
    # Add a grid
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_regional_comparison.png"), bbox_inches='tight')
    plt.close()
    
    print("Regional comparison plot saved.")

# Function to plot stony coral cover vs. other key taxa groups
def plot_coral_vs_other_taxa(df):
    """
    Plot the relationship between stony coral cover and other major benthic taxa.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating coral vs. other taxa plot...")
    
    # Select key taxa groups to compare
    key_taxa = ['Stony_coral', 'Macroalgae', 'Octocoral', 'Porifera', 'Substrate']
    
    # Create a long-format dataframe for these taxa
    df_long = df.melt(id_vars=['Year', 'Subregion', 'Habitat', 'Site_name'], 
                      value_vars=key_taxa,
                      var_name='Taxa', value_name='Cover_percent')
    
    # For Macroalgae, Octocoral, Porifera, Substrate - convert to percentage
    df_long.loc[df_long['Taxa'] != 'Stony_coral', 'Cover_percent'] *= 100
    
    # Group by year and taxa, calculate mean cover
    yearly_taxa_avg = df_long.groupby(['Year', 'Taxa'])['Cover_percent'].mean().reset_index()
    
    # Create a multi-line plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for different taxa
    taxa_colors = {
        'Stony_coral': COLORS['coral'],
        'Macroalgae': COLORS['reef_green'],
        'Octocoral': COLORS['dark_blue'],
        'Porifera': COLORS['sand'],  # Sponges
        'Substrate': COLORS['light_blue']  # Bare substrate
    }
    
    taxa_labels = {
        'Stony_coral': 'Stony Coral',
        'Macroalgae': 'Macroalgae',
        'Octocoral': 'Octocoral',
        'Porifera': 'Sponges',
        'Substrate': 'Bare Substrate'
    }
    
    # Plot each taxa group
    for taxa, color in taxa_colors.items():
        taxa_data = yearly_taxa_avg[yearly_taxa_avg['Taxa'] == taxa]
        ax.plot(taxa_data['Year'], taxa_data['Cover_percent'], marker='o', color=color, 
                linewidth=2.5, markersize=7, label=taxa_labels.get(taxa, taxa))
    
    # Set plot aesthetics
    ax.set_title('Evolution of Major Benthic Taxa Cover (1996-2023)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cover (%)')
    
    # Set axis limits
    ax.set_xlim(yearly_taxa_avg['Year'].min() - 0.5, yearly_taxa_avg['Year'].max() + 0.5)
    ax.set_ylim(0, yearly_taxa_avg['Cover_percent'].max() * 1.1)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, title='Benthic Taxa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_vs_other_taxa.png"), bbox_inches='tight')
    plt.close()
    
    print("Coral vs. other taxa plot saved.")

# Function to plot trend with environmental events
def plot_trend_with_events(df):
    """
    Plot the main trend of stony coral cover with major environmental events highlighted.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating trend with environmental events plot...")
    
    # Calculate yearly averages
    yearly_avg = df.groupby('Year')['Stony_coral'].mean().reset_index()
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot overall trend line with shadow effect for depth
    ax.plot(yearly_avg['Year'], yearly_avg['Stony_coral'], marker='o', color=COLORS['coral'], 
            linewidth=3.5, markersize=9, label='Mean Stony Coral Cover', 
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()])
    
    # Add subtle grid only on y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Define major environmental events
    events = {
        1998: {"name": "1998 Bleaching", "impact": "Severe", "description": "Global mass bleaching event"},
        2005: {"name": "2005 Bleaching", "impact": "Severe", "description": "Caribbean mass bleaching event"},
        2010: {"name": "2010 Cold Snap", "impact": "Moderate", "description": "Florida Keys cold temperature anomaly"},
        2014: {"name": "2014-2015 Bleaching", "impact": "Severe", "description": "Global mass bleaching event"},
        2017: {"name": "Hurricane Irma", "impact": "Severe", "description": "Category 5 hurricane"},
        2019: {"name": "SCTLD Outbreak", "impact": "Severe", "description": "Stony Coral Tissue Loss Disease spread"}
    }
    
    # Add event markers and annotations with improved spacing
    event_positions = []  # To track y-positions for better spacing
    
    for year, event_info in events.items():
        if year in yearly_avg['Year'].values:
            idx = yearly_avg[yearly_avg['Year'] == year].index[0]
            y_val = yearly_avg.loc[idx, 'Stony_coral']
            
            # Color coding based on impact severity
            color = COLORS['coral'] if event_info['impact'] == 'Severe' else COLORS['sand']
            
            # Add vertical line for the event
            ax.axvline(x=year, color='gray', linestyle=':', alpha=0.5, zorder=1)
            
            # Add marker
            ax.scatter(year, y_val, s=150, color=color, edgecolor='black', linewidth=1.5, zorder=10,
                     path_effects=[pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])
            
            # Calculate appropriate y position for annotation to avoid overlap
            # Use alternating positions and adjust based on previous annotations
            base_offset = yearly_avg['Stony_coral'].max() * 0.15
            
            if len(event_positions) > 0:
                # Adjust based on previous positions
                last_pos = event_positions[-1]['y']
                last_year = event_positions[-1]['year']
                year_diff = abs(year - last_year)
                
                if year_diff <= 3:  # If events are close in time
                    # Alternate up and down with more spacing
                    if last_pos > y_val:
                        ypos = y_val - base_offset
                    else:
                        ypos = y_val + base_offset
                else:
                    # If events are far apart, use alternate pattern
                    ypos = y_val + (base_offset if idx % 2 == 0 else -base_offset)
            else:
                # First event
                ypos = y_val + base_offset
            
            # Ensure annotation is within plot bounds
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            if ypos > ax.get_ylim()[1] - 0.1 * y_range:
                ypos = ax.get_ylim()[1] - 0.1 * y_range
            if ypos < ax.get_ylim()[0] + 0.1 * y_range:
                ypos = ax.get_ylim()[0] + 0.1 * y_range
                
            # Track this position
            event_positions.append({'year': year, 'y': ypos})
            
            # Add annotation with curved arrow for better visibility
            ax.annotate(
                event_info['name'],
                xy=(year, y_val), xytext=(year, ypos),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', color='gray', lw=1.5),
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color),
                fontsize=11, fontweight='bold'
            )
    
    # Set plot attributes
    ax.set_title('Stony Coral Cover Trend with Major Environmental Events (1996-2023)', 
                fontweight='bold', pad=15, fontsize=18)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Stony Coral Cover (%)', fontweight='bold')
    
    # Add a legend for event impact
    from matplotlib.patches import Patch
    impact_legend_elements = [
        Patch(facecolor=COLORS['coral'], edgecolor='black', label='Severe Impact Event'),
        Patch(facecolor=COLORS['sand'], edgecolor='black', label='Moderate Impact Event')
    ]
    ax.legend(handles=impact_legend_elements, loc='upper right', frameon=True, 
              facecolor='white', framealpha=0.9, edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_trend_with_events.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Trend with environmental events plot saved.")

# Function to plot habitat comparison
def plot_habitat_comparison(df):
    """
    Create a bar chart showing average stony coral cover by different habitat types.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating habitat comparison plot...")
    
    # Group by habitat
    habitat_avg = df.groupby('Habitat')['Stony_coral'].mean().reset_index()
    
    # Define habitat color mapping
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    habitat_names = {
        'OS': 'Offshore\nShallow',
        'OD': 'Offshore\nDeep',
        'P': 'Patch\nReef',
        'HB': 'Hard-\nbottom',
        'BCP': 'Backcountry\nPatch'
    }
    
    # Sort habitats by cover for better visualization
    habitat_avg = habitat_avg.sort_values('Stony_coral', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create the bar chart with enhanced styling
    bars = ax.bar(
        habitat_avg['Habitat'].map(lambda x: habitat_names.get(x, x)), 
        habitat_avg['Stony_coral'],
        color=[habitat_colors.get(h, 'gray') for h in habitat_avg['Habitat']],
        width=0.7,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Add value labels on top of bars with enhanced styling
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=COLORS['text'])
    
    # Add styling elements
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Set plot attributes
    ax.set_title('Average Stony Coral Cover by Habitat Type', fontweight='bold', pad=15, fontsize=18)
    ax.set_ylabel('Stony Coral Cover (%)', fontweight='bold')
    ax.set_xlabel('Habitat Type', fontweight='bold')
    
    # Add explanation text about habitat types
    habitat_descriptions = {
        'OS': 'Offshore shallow reefs (depth <6m)',
        'OD': 'Offshore deep reefs (depth >6m)',
        'P': 'Patch reefs (isolated reef structures)',
        'HB': 'Hardbottom (low relief limestone)',
        'BCP': 'Backcountry patch reefs (near shore)'
    }
    
    # Create text for habitat descriptions
    description_text = "Habitat Types:\n"
    for hab_code in habitat_avg['Habitat']:
        description_text += f"• {habitat_names.get(hab_code, hab_code).replace('\n', ' ')}: {habitat_descriptions.get(hab_code, '')}\n"
    
    # Add the description text
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    ax.text(0.60, 0.8, description_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_habitat_comparison.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Habitat comparison plot saved.")

# Function to plot before-after comparison of major events
def plot_event_impact_comparison(df):
    """
    Create a grouped bar chart showing coral cover before, during, and after major environmental events.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating before-after event impact comparison plot...")
    
    # Define key events for before-after comparison
    key_event_years = [2005, 2010, 2017]  # Bleaching, Cold snap, Hurricane Irma
    event_names = ["2005\nBleaching", "2010\nCold Snap", "2017\nHurricane Irma"]
    
    # Calculate before and after means (2 years before and after)
    before_after_data = []
    
    for event_year in key_event_years:
        before_years = [event_year - 2, event_year - 1]
        event_year_data = [event_year]
        after_years = [event_year + 1, event_year + 2]
        
        before_mean = df[df['Year'].isin(before_years)]['Stony_coral'].mean()
        event_mean = df[df['Year'].isin(event_year_data)]['Stony_coral'].mean()
        after_mean = df[df['Year'].isin(after_years)]['Stony_coral'].mean()
        
        before_after_data.append({
            'Event': event_names[key_event_years.index(event_year)],
            'Before': before_mean,
            'During': event_mean,
            'After': after_mean
        })
    
    # Convert to DataFrame for easier plotting
    ba_df = pd.DataFrame(before_after_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Set up positions and width for grouped bars
    x = np.arange(len(ba_df))
    width = 0.25
    
    # Create the grouped bar chart with enhanced styling
    ax.bar(x - width, ba_df['Before'], width, label='Before (2 yrs)', 
           color=COLORS['reef_green'], alpha=0.8, edgecolor='white', linewidth=1)
    ax.bar(x, ba_df['During'], width, label='During Event', 
           color=COLORS['coral'], alpha=0.8, edgecolor='white', linewidth=1)
    ax.bar(x + width, ba_df['After'], width, label='After (2 yrs)', 
           color=COLORS['ocean_blue'], alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add styling elements
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Add labels and formatting
    ax.set_title('Impact of Major Events on Coral Cover', fontweight='bold', pad=15, fontsize=18)
    ax.set_ylabel('Stony Coral Cover (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ba_df['Event'])
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right',
              edgecolor='lightgray')
    
    # Add percent change annotations
    for i, row in ba_df.iterrows():
        pct_change = ((row['After'] - row['Before']) / row['Before']) * 100
        color = 'red' if pct_change < 0 else 'green'
        
        max_val = max(row['Before'], row['During'], row['After'])
        # Position the label higher to avoid overlap
        y_pos = max_val + 0.5
        
        ax.text(i, y_pos, f"{pct_change:.1f}% change", 
                color=color, ha='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.2'))
    
    # Ensure enough space at the top of the plot for the annotations
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.3)
    
    # Add event descriptions
    event_descriptions = {
        "2005\nBleaching": "Caribbean-wide mass coral bleaching due to elevated sea temperatures",
        "2010\nCold Snap": "Unusual cold temperatures in Florida caused significant coral mortality",
        "2017\nHurricane Irma": "Category 5 hurricane with direct impacts on reef structure"
    }
    
    description_text = "Event Descriptions:\n"
    for event in ba_df['Event']:
        description_text += f"• {event.replace('\n', ' ')}: {event_descriptions.get(event, '')}\n"
    
    # Add the description text
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    ax.text(0.02, 0.85, description_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
           verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_event_impact.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Event impact comparison plot saved.")

# Function to plot yearly change in coral cover
def plot_yearly_change(df):
    """
    Create a bar chart showing the annual change in stony coral cover.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Creating yearly change plot...")
    
    # Group by year and calculate mean stony coral cover
    yearly_avg = df.groupby('Year')['Stony_coral'].mean().reset_index()
    
    # Calculate year-over-year change
    yearly_avg['Change'] = yearly_avg['Stony_coral'].diff()
    yearly_avg.dropna(subset=['Change'], inplace=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot bars with different colors for positive/negative and enhanced styling
    bars = ax.bar(yearly_avg['Year'], yearly_avg['Change'], 
                 color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['Change']], 
                 alpha=0.8, width=0.7, edgecolor='white', linewidth=1)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=5)
    
    # Add styling elements
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Add value labels to significant bars
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.3:  # Only label bars with significant values
            y_pos = height + 0.2 if height > 0 else height - 0.4
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.1f}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color=COLORS['text'],
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
    
    # Set plot attributes
    ax.set_title('Annual Change in Stony Coral Cover (1996-2023)', fontweight='bold', pad=15, fontsize=18)
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Change in Coral Cover (%)', fontweight='bold')
    
    # Add a legend for gain/loss
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['reef_green'], edgecolor='white', label='Coral Cover Gain'),
        Patch(facecolor=COLORS['coral'], edgecolor='white', label='Coral Cover Loss')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              facecolor='white', framealpha=0.9, edgecolor='lightgray')
    
    # Calculate and add statistics
    avg_change = yearly_avg['Change'].mean()
    max_gain = yearly_avg['Change'].max()
    max_loss = yearly_avg['Change'].min()
    max_gain_year = int(yearly_avg.loc[yearly_avg['Change'].idxmax(), 'Year'])
    max_loss_year = int(yearly_avg.loc[yearly_avg['Change'].idxmin(), 'Year'])
    
    stats_text = (
        f"Average annual change: {avg_change:.2f}%\n"
        f"Maximum gain: {max_gain:.2f}% ({max_gain_year})\n"
        f"Maximum loss: {max_loss:.2f}% ({max_loss_year})"
    )
    
    # Add the stats text with enhanced styling
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_yearly_change.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Yearly change plot saved.")

# Function to analyze top performing and declining sites
def analyze_site_performance(df, stations_df):
    """
    Analyze and visualize the performance of specific sites, highlighting the
    best performing and most declining sites over the monitoring period.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
        stations_df (DataFrame): DataFrame with station metadata
    """
    print("Analyzing site-specific performance...")
    
    # Calculate early and late period averages for each site
    # Early period: 1996-2000, Late period: 2018-2023
    df_early = df[df['Year'].between(1996, 2000)]
    df_late = df[df['Year'].between(2018, 2023)]
    
    # Calculate site averages for each period
    site_early_avg = df_early.groupby('Site_name')['Stony_coral'].mean().reset_index()
    site_early_avg.rename(columns={'Stony_coral': 'Early_avg'}, inplace=True)
    
    site_late_avg = df_late.groupby('Site_name')['Stony_coral'].mean().reset_index()
    site_late_avg.rename(columns={'Stony_coral': 'Late_avg'}, inplace=True)
    
    # Merge the two periods
    site_change = pd.merge(site_early_avg, site_late_avg, on='Site_name', how='inner')
    
    # Calculate absolute and relative change
    site_change['Abs_change'] = site_change['Late_avg'] - site_change['Early_avg']
    site_change['Rel_change'] = (site_change['Abs_change'] / site_change['Early_avg']) * 100
    
    # Add site metadata
    site_metadata = stations_df[['Site_name', 'Habitat', 'Subregion']].drop_duplicates()
    site_change = pd.merge(site_change, site_metadata, on='Site_name', how='left')
    
    # Some sites might not have data for both periods - filter these out
    site_change = site_change.dropna(subset=['Abs_change', 'Rel_change'])
    
    # Identify top performing and declining sites
    top_sites = site_change.nlargest(10, 'Abs_change')
    bottom_sites = site_change.nsmallest(10, 'Abs_change')
    
    # Create a combined dataframe for visualization
    top_bottom_sites = pd.concat([top_sites, bottom_sites])
    
    # Add a category column for better visualization
    top_bottom_sites['Category'] = np.where(top_bottom_sites['Abs_change'] >= 0, 'Least Decline/Recovery', 'Most Decline')
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), facecolor=COLORS['background'])
    
    # Center the title
    fig.suptitle('SITE-SPECIFIC ANALYSIS OF STONY CORAL COVER CHANGE', 
                 fontsize=22, fontweight='bold', y=0.98,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Color mapping for habitats
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    # Full habitat names
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Region names
    region_names = {
        'UK': 'Upper Keys',
        'MK': 'Middle Keys',
        'LK': 'Lower Keys'
    }
    
    # Plot 1: Top and Bottom Sites by Absolute Change
    ax1.set_facecolor(COLORS['background'])
    
    # Sort for better visualization
    plot_data = top_bottom_sites.sort_values('Abs_change')
    
    # Create horizontal bar chart
    bars = ax1.barh(plot_data['Site_name'], plot_data['Abs_change'], 
                   color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in plot_data['Abs_change']], 
                   alpha=0.8, height=0.6, edgecolor='white', linewidth=1)
    
    # Add a vertical line at x=0
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5, zorder=5)
    
    # Add styling elements
    ax1.grid(axis='x', linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        x_pos = width + 0.3 if width > 0 else width - 0.3
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', ha='left' if width > 0 else 'right', 
                fontsize=9, fontweight='bold', color=COLORS['text'])
    
    # Enhance site labels with habitat and region information
    new_labels = []
    for site in plot_data['Site_name']:
        site_row = plot_data[plot_data['Site_name'] == site].iloc[0]
        habitat_code = site_row['Habitat']
        region_code = site_row['Subregion']
        
        # Get full names
        habitat = habitat_names.get(habitat_code, habitat_code)
        region = region_names.get(region_code, region_code)
        
        new_labels.append(f"{site} ({habitat}, {region})")
    
    # Set new labels
    ax1.set_yticks(range(len(new_labels)))
    ax1.set_yticklabels(new_labels, fontsize=10)
    
    # Set plot attributes
    ax1.set_title('Top Performing and Most Declining Sites (Absolute Change)', 
                 fontweight='bold', pad=15, fontsize=18)
    ax1.set_xlabel('Absolute Change in Coral Cover (%)', fontweight='bold', fontsize=14)
    
    # Add legend for interpretation
    legend_elements = [
        Patch(facecolor=COLORS['reef_green'], edgecolor='white', label='Least Decline/Recovery'),
        Patch(facecolor=COLORS['coral'], edgecolor='white', label='Most Decline')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, 
              facecolor='white', framealpha=0.9, edgecolor='lightgray')
    
    # Plot 2: Site Performance Scatter Plot
    ax2.set_facecolor(COLORS['background'])
    
    # Create scatter plot of early vs late averages
    for habitat, color in habitat_colors.items():
        habitat_data = site_change[site_change['Habitat'] == habitat]
        if not habitat_data.empty:
            ax2.scatter(habitat_data['Early_avg'], habitat_data['Late_avg'], 
                       s=80, color=color, alpha=0.8, edgecolor='white', linewidth=1,
                       label=habitat_names.get(habitat, habitat))
    
    # Add site labels for extreme sites
    for _, site in pd.concat([top_sites.head(5), bottom_sites.head(5)]).iterrows():
        ax2.annotate(site['Site_name'], 
                    xy=(site['Early_avg'], site['Late_avg']),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add diagonal line (no change line)
    max_val = max(site_change['Early_avg'].max(), site_change['Late_avg'].max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Change Line')
    
    # Add grid
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot attributes
    ax2.set_title('Site Performance: Early Period (1996-2000) vs Late Period (2018-2023)', 
                 fontweight='bold', pad=15, fontsize=18)
    ax2.set_xlabel('Early Period Coral Cover (%)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Late Period Coral Cover (%)', fontweight='bold', fontsize=14)
    
    # Set equal aspect to properly show the diagonal line
    ax2.set_aspect('equal')
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    
    # Add legend for habitat types
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
              title='Habitat Type', loc='upper left')
    
    # Add quadrant labels
    ax2.text(max_val*0.75, max_val*0.75, "ABOVE AVERAGE\nMAINTAINERS", 
            ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['reef_green'], alpha=0.2))
    
    ax2.text(max_val*0.75, max_val*0.25, "POSITIVE\nSURPRISERS", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['reef_green'], alpha=0.2))
    
    ax2.text(max_val*0.25, max_val*0.75, "NEGATIVE\nSURPRISERS", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['coral'], alpha=0.2))
    
    ax2.text(max_val*0.25, max_val*0.25, "BELOW AVERAGE\nMAINTAINERS", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['coral'], alpha=0.2))
    
    # Add informative text about the analysis
    info_text = (
        "This analysis compares coral cover at each monitoring site\n"
        "between early monitoring (1996-2000) and recent years (2018-2023).\n"
        "Points above the diagonal line indicate sites that have maintained\n"
        "or improved coral cover, while points below show decline."
    )
    
    # Add the info text with enhanced styling and square shape
    props = dict(boxstyle='square,pad=1.0', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    ax2.text(-1.05, 0.60, info_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='left', bbox=props)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust layout to center the title
    plt.savefig(os.path.join(results_dir, "stony_coral_site_performance.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Site performance analysis saved.")

# Function to create a map visualization of sites and their coral cover
def create_site_map_visualization(df, stations_df):
    """
    Create a map visualization showing the geographic distribution of monitoring sites
    with their current coral cover status.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
        stations_df (DataFrame): DataFrame with station metadata including coordinates
    """
    print("Creating site map visualization...")
    
    # Get recent data (last 3 years) for current status
    recent_years = df['Year'].max() - 2
    recent_data = df[df['Year'] >= recent_years]
    
    # Calculate average coral cover for each site in recent years
    site_recent_avg = recent_data.groupby('Site_name')['Stony_coral'].mean().reset_index()
    site_recent_avg.rename(columns={'Stony_coral': 'Recent_cover'}, inplace=True)
    
    # Merge with station metadata to get coordinates
    site_coords = stations_df[['Site_name', 'Habitat', 'Subregion', 'latDD', 'lonDD']].drop_duplicates()
    site_map_data = pd.merge(site_recent_avg, site_coords, on='Site_name', how='left')
    
    # Create the map figure
    fig, ax = plt.subplots(figsize=(18, 12), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Basemap is a simpler approach since we're just doing Florida Keys
    # Define boundaries for the Florida Keys area
    min_lon, max_lon = site_map_data['lonDD'].min() - 0.2, site_map_data['lonDD'].max() + 0.2
    min_lat, max_lat = site_map_data['latDD'].min() - 0.1, site_map_data['latDD'].max() + 0.1
    
    # Plot Florida Keys outline (simplified approach)
    # This is a very simplified approach - in a real application you would use actual GIS data
    # For demonstration purposes, we'll just draw the basic shape of the Florida Keys
    keys_outline_x = np.linspace(min_lon, max_lon, 100)
    keys_outline_y = np.sin((keys_outline_x - min_lon) / (max_lon - min_lon) * np.pi) * 0.05 + min_lat + 0.1
    ax.plot(keys_outline_x, keys_outline_y, 'k-', linewidth=2, alpha=0.5)
    
    # Fill land area with a light color
    ax.fill_between(keys_outline_x, keys_outline_y, min_lat, color='lightgray', alpha=0.3)
    
    # Color mapping for habitats
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    # Create a colormap for coral cover
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=site_map_data['Recent_cover'].max())
    
    # Plot each site as a marker on the map
    for habitat, color in habitat_colors.items():
        habitat_sites = site_map_data[site_map_data['Habitat'] == habitat]
        
        if not habitat_sites.empty:
            # Create a scatter plot with marker size based on coral cover
            scatter = ax.scatter(
                habitat_sites['lonDD'], 
                habitat_sites['latDD'],
                s=habitat_sites['Recent_cover'] * 20 + 50,  # Scale marker size
                c=habitat_sites['Recent_cover'],
                cmap='YlOrRd',
                norm=norm,
                alpha=0.8,
                edgecolor='black',
                linewidth=1,
                marker='^' if habitat in ['OS', 'OD'] else 'o',  # Different markers for different habitat types
                label=habitat
            )
    
    # Add site labels for selected sites (top coral cover)
    top_sites = site_map_data.nlargest(5, 'Recent_cover')
    for _, site in top_sites.iterrows():
        ax.annotate(
            site['Site_name'],
            xy=(site['lonDD'], site['latDD']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )
    
    # Add colorbar for coral cover
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Recent Coral Cover (%)', fontweight='bold', fontsize=12)
    
    # Add legend for habitat types
    # Convert habitat codes to full names for legend
    from matplotlib.lines import Line2D
    
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Create custom legend elements
    legend_elements = []
    for habitat, name in habitat_names.items():
        if habitat in ['OS', 'OD']:
            marker = '^'
        else:
            marker = 'o'
        
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor=habitat_colors.get(habitat, 'gray'), 
                  markersize=10, label=name)
        )
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             facecolor='white', framealpha=0.9, title='Habitat Type')
    
    # Add region labels
    region_positions = {
        'Upper Keys': (max_lon - 0.3, max_lat - 0.05),
        'Middle Keys': ((max_lon + min_lon) / 2, min_lat + 0.12),
        'Lower Keys': (min_lon + 0.3, min_lat + 0.08)
    }
    
    for region, pos in region_positions.items():
        ax.text(pos[0], pos[1], region, fontsize=14, fontweight='bold', ha='center', va='center',
               color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Set plot attributes
    ax.set_title('Geographic Distribution of Monitoring Sites and Coral Cover Status', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Longitude', fontweight='bold', fontsize=14)
    ax.set_ylabel('Latitude', fontweight='bold', fontsize=14)
    
    # Set axis limits to focus on the Florida Keys
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_site_map.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Site map visualization saved.")

# Function to perform statistical trend analysis
def perform_trend_analysis(df):
    """
    Perform statistical trend analysis on stony coral cover data.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with stony coral data
    """
    print("Performing statistical trend analysis...")
    
    # Group by year for overall trend analysis
    yearly_avg = df.groupby('Year')['Stony_coral'].agg(['mean', 'std', 'count']).reset_index()
    yearly_avg.rename(columns={'mean': 'Stony_coral'}, inplace=True)
    
    # Create Figure with enhanced styling
    fig = plt.figure(figsize=(18, 14), facecolor=COLORS['background'])
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
    
    # Plot 1: Original Data with Linear Trend
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS['background'])
    
    # Calculate linear regression
    X = yearly_avg['Year'].values.reshape(-1, 1)
    y = yearly_avg['Stony_coral'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    rsquared = model.score(X, y)
    
    # Calculate Mann-Kendall Trend Test
    mk_result = stats.kendalltau(yearly_avg['Year'], yearly_avg['Stony_coral'])
    
    # Plot original data with enhanced styling
    ax1.scatter(yearly_avg['Year'], yearly_avg['Stony_coral'], color=COLORS['coral'], s=100, 
                label='Annual Mean', zorder=5, edgecolor='white', linewidth=1,
                alpha=0.8)
    
    # Plot linear trend with enhanced styling
    ax1.plot(yearly_avg['Year'], y_pred, color=COLORS['dark_blue'], linestyle='-', linewidth=3, 
             label=f'Linear Trend (Slope: {slope:.4f}%/year)', zorder=4,
             path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()])
    
    # Add LOWESS smoothing trend with enhanced styling
    lowess_frac = 0.5  # This controls the smoothness (0.5 = 50% of data used for each estimation)
    lowess_result = lowess(yearly_avg['Stony_coral'], yearly_avg['Year'], frac=lowess_frac)
    ax1.plot(yearly_avg['Year'], lowess_result[:, 1], color=COLORS['reef_green'], linestyle='-', linewidth=3, 
             label=f'LOWESS Trend', zorder=3,
             path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()])
    
    # Calculate 5-year moving average with enhanced styling
    yearly_avg['5yr_MA'] = yearly_avg['Stony_coral'].rolling(window=5, center=True).mean()
    ax1.plot(yearly_avg['Year'], yearly_avg['5yr_MA'], color=COLORS['sand'], linestyle='--', linewidth=2.5,
             label='5-Year Moving Average', zorder=2,
             path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()])
    
    # Add key events as vertical lines
    key_events = {
        1998: "1998 Bleaching",
        2005: "2005 Bleaching",
        2010: "2010 Cold Snap",
        2014: "2014-15 Bleaching",
        2017: "Hurricane Irma"
    }
    
    for year, event in key_events.items():
        if year in yearly_avg['Year'].values:
            ax1.axvline(x=year, color='gray', linestyle=':', alpha=0.5, zorder=1)
            
            # Find the y position based on the linear trend (to avoid overlapping)
            idx = yearly_avg[yearly_avg['Year'] == year].index[0]
            y_pos = y_pred[idx] - 0.5
            
            # Add text annotation
            ax1.text(year, y_pos, event, rotation=90, ha='center', va='top', 
                    fontsize=10, color=COLORS['text'], fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot attributes
    ax1.set_title('Stony Coral Cover Trend Analysis (1996-2023)', 
                 fontweight='bold', fontsize=20, pad=15,
                 path_effects=[pe.withStroke(linewidth=1, foreground='white')])
    ax1.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Stony Coral Cover (%)', fontweight='bold', fontsize=14)
    
    # Enhance legend
    legend = ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
                      fontsize=12, edgecolor='lightgray')
    legend.get_frame().set_linewidth(1.5)
    
    # Add statistical annotations with enhanced styling
    stats_text = (
        f"Linear Regression:\n"
        f"Slope = {slope:.4f} % per year\n"
        f"R² = {rsquared:.4f}\n"
        f"\nMann-Kendall Test:\n"
        f"tau = {mk_result.correlation:.4f}\n"
        f"p-value = {mk_result.pvalue:.4f}"
    )
    
    # Add significance interpretation
    if mk_result.pvalue < 0.05:
        if mk_result.correlation < 0:
            trend_interpretation = "Significant DECREASING trend"
            interp_color = 'red'
        else:
            trend_interpretation = "Significant INCREASING trend"
            interp_color = 'green'
    else:
        trend_interpretation = "No significant trend"
        interp_color = COLORS['text']
    
    stats_text += f"\n\nInterpretation: {trend_interpretation}"
    
    # Add the stats text to the plot with enhanced styling
    stats_box = ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, 
                       fontsize=13, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, 
                                edgecolor=COLORS['dark_blue'], linewidth=2))
    
    # Highlight the interpretation with color
    interp_box = ax1.text(0.02, 0.68, f"Interpretation: {trend_interpretation}", 
                        transform=ax1.transAxes, fontsize=13, fontweight='bold',
                        verticalalignment='top', horizontalalignment='left',
                        color=interp_color,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                                  edgecolor=interp_color, linewidth=1.5))
    
    # Plot 2: Detrended Analysis
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['background'])
    
    # Calculate detrended data
    yearly_avg['detrended'] = yearly_avg['Stony_coral'] - y_pred
    
    # Plot detrended data with enhanced styling
    bars = ax2.bar(yearly_avg['Year'], yearly_avg['detrended'], 
                  color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['detrended']],
                  alpha=0.8, width=0.7, edgecolor='white', linewidth=1)
    
    # Add a horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, zorder=5)
    
    # Add labels for significant residuals
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.5:  # Only label significant residuals
            va = 'bottom' if height > 0 else 'top'
            y_offset = 0.2 if height > 0 else -0.2
            ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                    f'{height:.2f}', ha='center', va=va, 
                    fontsize=9, fontweight='bold', color=COLORS['text'],
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
    
    # Add grid for better readability
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot attributes
    ax2.set_title('Detrended Stony Coral Cover (Residuals)', 
                 fontweight='bold', fontsize=16, pad=15)
    ax2.set_xlabel('Year', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Residual (%)', fontweight='bold', fontsize=12)
    
    # Add explanation of residuals
    ax2.text(0.02, 0.97, 
            "Residuals show the difference between\nactual values and the linear trend.",
            transform=ax2.transAxes, fontsize=11, fontstyle='italic',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Calculate and add autocorrelation
    from statsmodels.tsa.stattools import acf
    
    # Plot 3: Autocorrelation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['background'])
    
    # Calculate autocorrelation
    detrended = yearly_avg['detrended'].dropna()
    lag_acf = acf(detrended, nlags=min(12, len(detrended)-1))
    
    # Plot autocorrelation with enhanced styling
    lags = range(len(lag_acf))
    bars = ax3.bar(lags, lag_acf, color=COLORS['dark_blue'], alpha=0.8, width=0.6,
                  edgecolor='white', linewidth=1)
    
    # Color the lag 0 differently (it's always 1.0)
    bars[0].set_color(COLORS['coral'])
    
    # Add horizontal line at y=0
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add confidence intervals (95%)
    conf_level = 1.96 / np.sqrt(len(detrended))
    ax3.axhline(y=conf_level, linestyle='--', color='red', alpha=0.7,
               label=f'95% Confidence\nInterval (±{conf_level:.2f})')
    ax3.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.7)
    
    # Fill the confidence interval area for better visualization
    ax3.fill_between([-0.5, len(lag_acf)-0.5], -conf_level, conf_level, 
                    color='gray', alpha=0.1)
    
    # Add labels for significant autocorrelations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if abs(height) > conf_level and i > 0:  # Only label significant autocorrelations (excluding lag 0)
            va = 'bottom' if height > 0 else 'top'
            y_offset = 0.05 if height > 0 else -0.05
            ax3.text(i, height + y_offset * (1 if height > 0 else -1),
                    f'{height:.2f}', ha='center', va=va, 
                    fontsize=9, fontweight='bold', color=COLORS['text'])
    
    # Add grid for better readability
    ax3.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot attributes
    ax3.set_title('Autocorrelation of Detrended Data', 
                 fontweight='bold', fontsize=16, pad=15)
    ax3.set_xlabel('Lag (Years)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Autocorrelation', fontweight='bold', fontsize=12)
    ax3.set_xlim(-0.5, len(lag_acf)-0.5)
    
    # Add legend
    ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9,
              edgecolor='lightgray')
    
    # Add explanation of autocorrelation
    ax3.text(0.02, 0.97, 
            "Autocorrelation shows whether past values\ninfluence future values. Significant values\noutside the confidence interval indicate\ntemporal dependence in the data.",
            transform=ax3.transAxes, fontsize=11, fontstyle='italic',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add a title for the entire figure
    fig.suptitle('STATISTICAL TREND ANALYSIS OF STONY CORAL COVER', 
                 fontsize=22, fontweight='bold', y=0.98,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "stony_coral_trend_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Statistical trend analysis saved.")
    
    # Perform regional trend analysis
    print("Performing regional trend analysis...")
    
    # Create a figure for regional trend analysis with enhanced styling
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True, facecolor=COLORS['background'])
    
    # Define regions and their colors
    regions = ['UK', 'MK', 'LK']
    region_names = {'UK': 'Upper Keys', 'MK': 'Middle Keys', 'LK': 'Lower Keys'}
    region_colors = {'UK': COLORS['dark_blue'], 'MK': COLORS['ocean_blue'], 'LK': COLORS['light_blue']}
    
    # Analyze each region
    for i, region in enumerate(regions):
        axes[i].set_facecolor(COLORS['background'])
        
        region_data = df[df['Subregion'] == region]
        yearly_region_avg = region_data.groupby('Year')['Stony_coral'].mean().reset_index()
        
        # Linear regression for regional data
        X_region = yearly_region_avg['Year'].values.reshape(-1, 1)
        y_region = yearly_region_avg['Stony_coral'].values
        
        if len(X_region) > 1:  # Ensure we have data to fit
            model_region = LinearRegression().fit(X_region, y_region)
            y_region_pred = model_region.predict(X_region)
            
            # Calculate statistics
            slope_region = model_region.coef_[0]
            rsquared_region = model_region.score(X_region, y_region)
            
            # Mann-Kendall test
            mk_result_region = stats.kendalltau(yearly_region_avg['Year'], yearly_region_avg['Stony_coral'])
            
            # Plot data and trend with enhanced styling
            scatter = axes[i].scatter(yearly_region_avg['Year'], yearly_region_avg['Stony_coral'], 
                          color=region_colors[region], s=80, alpha=0.8, edgecolor='white', linewidth=1)
            
            trend_line = axes[i].plot(yearly_region_avg['Year'], y_region_pred, color='black', linewidth=2.5,
                                    path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.3), pe.Normal()])
            
            # Add statistics annotation
            if mk_result_region.pvalue < 0.05:
                if mk_result_region.correlation < 0:
                    trend_text = "DECREASING trend"
                    trend_color = 'red'
                else:
                    trend_text = "INCREASING trend"
                    trend_color = 'green'
            else:
                trend_text = "No significant trend"
                trend_color = COLORS['text']
            
            # Format the statistics text
            stats_text = (
                f"Slope: {slope_region:.4f} % per year\n"
                f"R²: {rsquared_region:.4f}\n"
                f"MK tau: {mk_result_region.correlation:.3f}\n"
                f"p-value: {mk_result_region.pvalue:.4f}"
            )
            
            # Add the stats box with enhanced styling
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                       fontsize=12, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                                edgecolor=region_colors[region], linewidth=2))
            
            # Add the trend interpretation with enhanced styling
            axes[i].text(0.05, 0.66, trend_text, transform=axes[i].transAxes, 
                       fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left',
                       color=trend_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                edgecolor=trend_color, linewidth=1.5))
            
            # Add annotation for average level
            avg_level = np.mean(y_region)
            axes[i].axhline(y=avg_level, color=region_colors[region], linestyle='--', alpha=0.7,
                          linewidth=2, label=f'Average: {avg_level:.2f}%')
            
            # Add legend
            axes[i].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9,
                         edgecolor='lightgray')
            
        else:
            axes[i].text(0.5, 0.5, "Insufficient data for analysis", transform=axes[i].transAxes,
                       fontsize=14, ha='center', fontweight='bold')
        
        # Set plot attributes
        axes[i].set_title(f"{region_names[region]}", fontweight='bold', fontsize=18, pad=15,
                        color=region_colors[region],
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        axes[i].set_xlabel('Year', fontweight='bold', fontsize=14)
        
        # Add grid
        axes[i].grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
    # Set common y-label
    fig.text(0.01, 0.5, 'Stony Coral Cover (%)', va='center', rotation='vertical', 
             fontsize=16, fontweight='bold')
    
    # Add title for the entire figure
    fig.suptitle('REGIONAL TREND ANALYSIS OF STONY CORAL COVER', 
                 fontsize=22, fontweight='bold', y=0.98,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about regional differences
    note_text = ("Note: Different regions show distinct patterns of coral cover change, reflecting\n"
                "localized influences of environmental factors, stressors, and management actions.")
    
    fig.text(0.5, 0.02, note_text, ha='center', va='center', 
             fontsize=11, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0.02, 0.05, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "stony_coral_regional_trend_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional trend analysis saved.")

# Main function to execute the analysis
def main():
    """
    Main function to execute the stony coral cover analysis.
    """
    print("\n=== CREMP Stony Coral Cover Analysis ===")
    print("Starting analysis...")
    
    # Load and preprocess data
    taxa_groups_df, stations_df = load_and_preprocess_data()
    
    # Execute analysis functions in a logical chronological order
    
    # 1. First, plot the overall trend to give a broad overview
    plot_overall_trend(taxa_groups_df)
    
    # 2. Plot trend with environmental events for context
    plot_trend_with_events(taxa_groups_df)
    
    # 3. Show yearly changes to understand the pattern better
    plot_yearly_change(taxa_groups_df)
    
    # 4. Perform statistical analysis to quantify the trends
    perform_trend_analysis(taxa_groups_df)
    
    # 5. Now explore the spatial patterns (regional and habitat differences)
    plot_trends_by_region(taxa_groups_df)
    plot_regional_comparison(taxa_groups_df)
    analyze_site_performance(taxa_groups_df, stations_df)
    create_site_map_visualization(taxa_groups_df, stations_df)
    
    # 6. Then explore ecosystem patterns (habitat differences and other taxa)
    plot_trends_by_habitat(taxa_groups_df)
    plot_habitat_comparison(taxa_groups_df)
    plot_coral_vs_other_taxa(taxa_groups_df)
    
    # 7. Detailed visualizations for specific aspects
    create_temporal_heatmap(taxa_groups_df)
    analyze_rate_of_change(taxa_groups_df)
    plot_event_impact_comparison(taxa_groups_df)
    
    print("\nAnalysis complete! All results saved as individual plots in the '{}' directory.".format(results_dir))

if __name__ == "__main__":
    main()