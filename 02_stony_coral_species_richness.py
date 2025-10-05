"""
02_stony_coral_species_richness.py - Analysis of Stony Coral Species Richness Evolution

This script analyzes the evolution of stony coral species richness across different
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
results_dir = "02_Results"
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
    Load and preprocess the CREMP dataset for species richness analysis.
    
    Returns:
        tuple: (species_df, stations_df) - Preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Stony Coral Species dataset
        species_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
        print(f"Species data loaded successfully with {len(species_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Print column names to verify data structure
        print("\nSpecies data columns:", species_df.columns.tolist()[:10], "...")
        
        # Convert date column to datetime format
        species_df['Date'] = pd.to_datetime(species_df['Date'])
        
        # Extract just the year for easier grouping
        species_df['Year'] = species_df['Year'].astype(int)
        
        # Get list of all coral species columns (excluding metadata columns)
        metadata_cols = ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID', 'Surveyed_all_years', 'points']
        species_cols = [col for col in species_df.columns if col not in metadata_cols]
        
        print(f"\nIdentified {len(species_cols)} coral species in the dataset")
        
        # Calculate species richness (number of species present at each station)
        # A species is considered present if its cover value is greater than 0
        species_df['species_richness'] = species_df[species_cols].apply(
            lambda row: sum(row > 0), axis=1
        )
        
        print(f"\nData loaded: {len(species_df)} records from {species_df['Year'].min()} to {species_df['Year'].max()}")
        
        return species_df, stations_df, species_cols
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to plot overall trend of species richness over time
def plot_overall_trend(df):
    """
    Plot the overall trend of stony coral species richness over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Creating overall trend plot...")
    
    # Group by year and calculate mean species richness
    yearly_avg = df.groupby('Year')['species_richness'].agg(['mean', 'std', 'count']).reset_index()
    yearly_avg['se'] = yearly_avg['std'] / np.sqrt(yearly_avg['count'])
    yearly_avg['ci_95'] = 1.96 * yearly_avg['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the mean line with enhanced styling
    ax.plot(yearly_avg['Year'], yearly_avg['mean'], marker='o', color=COLORS['coral'], 
            linewidth=3.5, markersize=10, label='Mean Species Richness',
            path_effects=[pe.SimpleLineShadow(offset=(2, -2), alpha=0.3), pe.Normal()])
    
    # Add 95% confidence interval band with enhanced styling
    ax.fill_between(yearly_avg['Year'], 
                    yearly_avg['mean'] - yearly_avg['ci_95'],
                    yearly_avg['mean'] + yearly_avg['ci_95'],
                    color=COLORS['coral'], alpha=0.2, label='95% Confidence Interval')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot aesthetics
    ax.set_title('EVOLUTION OF STONY CORAL SPECIES RICHNESS (1996-2023)', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Number of Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    
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
            linewidth=2, label=f'Linear Trend (Slope: {slope:.4f} species/year)',
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
    
    # Update legend with the new trend line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, facecolor='white', framealpha=0.9, 
             fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Removed summary textbox and background text to reduce clutter
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_trend.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Overall species richness trend plot saved.")

# Function to plot trends by region
def plot_trends_by_region(df):
    """
    Plot species richness trends separated by region (Upper Keys, Middle Keys, Lower Keys).
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Creating regional trends plot...")
    
    # Group by year and region, calculate mean species richness
    region_yearly_avg = df.groupby(['Year', 'Subregion'])['species_richness'].mean().reset_index()
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
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
    
    # Plot each region with enhanced styling
    for region, color in region_colors.items():
        region_data = region_yearly_avg[region_yearly_avg['Subregion'] == region]
        if not region_data.empty:
            ax.plot(region_data['Year'], region_data['species_richness'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=region_names.get(region, region),
                    path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
            
            # Add trend line for each region
            X = region_data['Year'].values.reshape(-1, 1)
            y = region_data['species_richness'].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            slope = model.coef_[0]
            
            ax.plot(region_data['Year'], y_pred, '--', color=color, alpha=0.7, linewidth=1.5)
            
            # Add slope annotation
            last_year = region_data['Year'].max()
            last_y_pred = y_pred[-1]
            ax.annotate(
                f"Slope: {slope:.3f}",
                xy=(last_year, last_y_pred),
                xytext=(last_year + 0.5, last_y_pred),
                fontsize=9,
                color=color,
                fontweight='bold',
                va='center'
            )
    
    # Set plot aesthetics
    ax.set_title('Stony Coral Species Richness by Region (1996-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Number of Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set axis limits
    ax.set_xlim(region_yearly_avg['Year'].min() - 0.5, region_yearly_avg['Year'].max() + 2)
    ax.set_ylim(0, region_yearly_avg['species_richness'].max() * 1.1)
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper right', edgecolor=COLORS['grid'],
                      title='Region')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_by_region.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional species richness trends plot saved.")

# Function to plot trends by habitat type
def plot_trends_by_habitat(df):
    """
    Plot species richness trends separated by habitat type.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Creating habitat-based trends plot...")
    
    # Group by year and habitat, calculate mean species richness
    habitat_yearly_avg = df.groupby(['Year', 'Habitat'])['species_richness'].mean().reset_index()
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
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
    
    # Plot each habitat with enhanced styling
    for habitat, color in habitat_colors.items():
        habitat_data = habitat_yearly_avg[habitat_yearly_avg['Habitat'] == habitat]
        if not habitat_data.empty:
            ax.plot(habitat_data['Year'], habitat_data['species_richness'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=habitat_names.get(habitat, habitat),
                    path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
            
            # Add trend line for each habitat
            if len(habitat_data) > 2:  # Only add trend line if we have enough data points
                X = habitat_data['Year'].values.reshape(-1, 1)
                y = habitat_data['species_richness'].values
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                slope = model.coef_[0]
                
                ax.plot(habitat_data['Year'], y_pred, '--', color=color, alpha=0.7, linewidth=1.5)
                
                # Add slope annotation
                last_year = habitat_data['Year'].max()
                last_y_pred = y_pred[-1]
                ax.annotate(
                    f"Slope: {slope:.3f}",
                    xy=(last_year, last_y_pred),
                    xytext=(last_year + 0.5, last_y_pred),
                    fontsize=9,
                    color=color,
                    fontweight='bold',
                    va='center'
                )
    
    # Set plot aesthetics
    ax.set_title('Stony Coral Species Richness by Habitat Type (1996-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Number of Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set axis limits
    ax.set_xlim(habitat_yearly_avg['Year'].min() - 0.5, habitat_yearly_avg['Year'].max() + 2)
    ax.set_ylim(0, habitat_yearly_avg['species_richness'].max() * 1.1)
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper right', edgecolor=COLORS['grid'],
                      title='Habitat Type')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_by_habitat.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Habitat-based species richness trends plot saved.")

# Function to create a heatmap of species richness trends
def create_temporal_heatmap(df):
    """
    Create a heatmap visualization of species richness changes over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Creating temporal heatmap...")
    
    # Define habitat and region color mapping
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    region_colors = {
        'UK': COLORS['dark_blue'],   # Upper Keys
        'MK': COLORS['ocean_blue'],  # Middle Keys
        'LK': COLORS['light_blue']   # Lower Keys
    }
    
    # Get a sample of sites to visualize (to avoid overcrowding)
    # Focus on sites that have been surveyed for the longest periods
    site_survey_counts = df.groupby('Site_name')['Year'].nunique().sort_values(ascending=False)
    # Reduce number of sites to avoid overcrowding - only select sites with most years of data
    sites_to_include = site_survey_counts[site_survey_counts > 15].index.tolist()[:8]  # Reduced to 8 sites for better visibility
    
    # Filter data for these sites and pivot to create a matrix suitable for a heatmap
    heatmap_data = df[df['Site_name'].isin(sites_to_include)]
    
    # Calculate average for each site and year
    heatmap_pivot = heatmap_data.groupby(['Site_name', 'Year'])['species_richness'].mean().reset_index()
    heatmap_pivot = heatmap_pivot.pivot(index='Site_name', columns='Year', values='species_richness')
    
    # Sort sites by their habitat type and region for better visualization
    site_info = df[df['Site_name'].isin(sites_to_include)][['Site_name', 'Habitat', 'Subregion']].drop_duplicates()
    # Sort first by habitat, then by region, then by site name for better organization
    site_order = site_info.sort_values(['Habitat', 'Subregion', 'Site_name']).Site_name.tolist()
    heatmap_pivot = heatmap_pivot.reindex(site_order)
    
    # Create a simplified site name mapping to avoid long names
    site_name_mapping = {}
    for i, site in enumerate(heatmap_pivot.index):
        # Create a shorter name by removing common words and truncating
        short_name = site.replace("Reef", "").replace("Coral", "").strip()
        if len(short_name) > 15:  # If still too long, truncate
            short_name = short_name[:15] + "..."
        site_name_mapping[site] = f"{i+1}. {short_name}"
    
    # Rename the index with shorter names
    heatmap_pivot.index = [site_name_mapping.get(site, site) for site in heatmap_pivot.index]
    
    # Create the heatmap with enhanced styling - adjusted figure size for better proportions
    plt.figure(figsize=(18, 12), facecolor=COLORS['background'])
    
    # Create a GridSpec layout with better proportions
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 4])
    
    # Create the metadata axis
    ax_meta = plt.subplot(gs[0])
    ax_meta.set_facecolor(COLORS['background'])
    
    # Create the heatmap axis
    ax = plt.subplot(gs[1])
    ax.set_facecolor(COLORS['background'])
    
    # Generate the heatmap with improved styling
    heatmap = sns.heatmap(
        heatmap_pivot, 
        cmap="YlOrRd",  # Changed to a more visible colormap
        ax=ax, 
        vmin=0, 
        vmax=max(15, heatmap_pivot.max().max()),  # Set reasonable limits
        linewidths=1.0,  # Increased line width for better cell separation
        linecolor='white',
        cbar_kws={
            'label': 'Number of Coral Species', 
            'shrink': 0.8,
            'aspect': 20,
            'pad': 0.01
        },
        annot=True,  # Add value annotations
        annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'},  # Increased font size
        fmt='.1f',  # Format to 1 decimal place
        square=True  # Make cells square for better proportions
    )
    
    # Improve the colorbar appearance
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Number of Coral Species', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set plot aesthetics
    ax.set_title('Stony Coral Species Richness by Site (1996-2023)', 
                fontweight='bold', pad=20, fontsize=20,
                color=COLORS['dark_blue'])
    
    # Improve X and Y labels
    ax.set_xlabel('Year', labelpad=15, fontweight='bold', fontsize=14)
    ax.set_ylabel('Site', labelpad=15, fontweight='bold', fontsize=14)
    
    # Improve tick labels - increased rotation for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12, fontweight='bold')
    
    # Remove axes elements for the metadata axis
    ax_meta.set_xticks([])
    ax_meta.set_yticks([])
    for spine in ax_meta.spines.values():
        spine.set_visible(False)
    
    # Set up the metadata axis with more space
    ax_meta.set_ylim(0, len(heatmap_pivot.index))
    ax_meta.set_title('Site Information', fontweight='bold', fontsize=14, pad=20)
    
    # Create a mapping from short names back to original names
    reverse_mapping = {v: k for k, v in site_name_mapping.items()}
    
    # Create legend elements for habitat and region types
    legend_elements = []
    
    # Add habitat legend elements
    for habitat, color in habitat_colors.items():
        habitat_name = {
            'OS': 'Offshore Shallow',
            'OD': 'Offshore Deep',
            'P': 'Patch Reef',
            'HB': 'Hardbottom',
            'BCP': 'Backcountry Patch'
        }.get(habitat, habitat)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', alpha=0.8, 
                                    label=f'Habitat: {habitat_name}'))
    
    # Add region legend elements
    for region, color in region_colors.items():
        region_name = {
            'UK': 'Upper Keys',
            'MK': 'Middle Keys',
            'LK': 'Lower Keys'
        }.get(region, region)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', alpha=0.8, 
                                    label=f'Region: {region_name}'))
    
    # Add habitat and region information with improved spacing
    for i, site_short in enumerate(heatmap_pivot.index):
        # Get the original site name
        original_site = reverse_mapping[site_short]
        site_info_row = site_info[site_info['Site_name'] == original_site]
        
        if not site_info_row.empty:
            site_habitat = site_info_row['Habitat'].values[0]
            site_region = site_info_row['Subregion'].values[0]
            
            # Map codes to full names
            habitat_name = {
                'OS': 'Offshore Shallow',
                'OD': 'Offshore Deep',
                'P': 'Patch Reef',
                'HB': 'Hardbottom',
                'BCP': 'Backcountry Patch'
            }.get(site_habitat, site_habitat)
            
            region_name = {
                'UK': 'Upper Keys',
                'MK': 'Middle Keys',
                'LK': 'Lower Keys'
            }.get(site_region, site_region)
            
            # Add colored boxes for habitat and region
            habitat_color = habitat_colors.get(site_habitat, 'gray')
            region_color = region_colors.get(site_region, 'gray')
            
            # Position for the boxes and text - increased spacing
            y_pos = len(heatmap_pivot.index) - i - 0.5
            
            # Add habitat indicator
            ax_meta.add_patch(plt.Rectangle(
                (0.05, y_pos - 0.3), 0.2, 0.6, facecolor=habitat_color, alpha=0.8, 
                edgecolor='white', linewidth=1.5, zorder=2
            ))
            
            # Add region indicator
            ax_meta.add_patch(plt.Rectangle(
                (0.35, y_pos - 0.3), 0.2, 0.6, facecolor=region_color, alpha=0.8, 
                edgecolor='white', linewidth=1.5, zorder=2
            ))
            
            # Simplified text labels with better spacing
            ax_meta.text(0.65, y_pos, f"H: {site_habitat} | R: {site_region}", 
                         fontsize=10, ha='left', va='center', fontweight='bold', zorder=3)
    
    # Add the legend to the figure with better positioning
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, 
                 fontsize=11, frameon=True, facecolor='white', framealpha=0.9,
                 bbox_to_anchor=(0.5, 0.01))
    
    # Add a note about the data source
    plt.figtext(0.5, -0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
               ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.subplots_adjust(wspace=0.1)  # Slightly increased space between subplots
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal species richness heatmap saved.")

# Function to analyze most common and rare species
def analyze_species_composition(df, species_cols):
    """
    Analyze the composition of coral species, identifying most common and rare species.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species data
        species_cols (list): List of column names representing coral species
    """
    print("Analyzing species composition...")
    
    # Calculate the prevalence of each species (percentage of stations where it's present)
    species_prevalence = {}
    for species in species_cols:
        # Count stations where the species is present (cover > 0)
        present_count = (df[species] > 0).sum()
        total_stations = len(df)
        prevalence = (present_count / total_stations) * 100
        species_prevalence[species] = prevalence
    
    # Convert to DataFrame for easier manipulation
    prevalence_df = pd.DataFrame(list(species_prevalence.items()), columns=['Species', 'Prevalence'])
    prevalence_df = prevalence_df.sort_values('Prevalence', ascending=False).reset_index(drop=True)
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), facecolor=COLORS['background'])
    fig.subplots_adjust(hspace=0.3)
    
    # Plot top 15 most common species
    top_species = prevalence_df.head(15).copy()
    # Clean species names for display
    top_species['Species'] = top_species['Species'].str.replace('_', ' ')
    
    # Create horizontal bar chart for top species
    bars1 = ax1.barh(top_species['Species'], top_species['Prevalence'], 
                    color=COLORS['coral'], alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels to the bars
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Set plot aesthetics for top species
    ax1.set_title('15 Most Common Stony Coral Species', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    ax1.set_xlabel('Prevalence (%)', fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_ylabel('Species', fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_xlim(0, top_species['Prevalence'].max() * 1.15)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='x')
    ax1.set_facecolor(COLORS['background'])
    
    # Plot 15 rarest species (excluding those with zero prevalence)
    rare_species = prevalence_df[prevalence_df['Prevalence'] > 0].tail(15).copy()
    # Clean species names for display
    rare_species['Species'] = rare_species['Species'].str.replace('_', ' ')
    
    # Create horizontal bar chart for rare species
    bars2 = ax2.barh(rare_species['Species'], rare_species['Prevalence'], 
                    color=COLORS['ocean_blue'], alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels to the bars
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold')
    
    # Set plot aesthetics for rare species
    ax2.set_title('15 Rarest Stony Coral Species', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    ax2.set_xlabel('Prevalence (%)', fontweight='bold', fontsize=12, labelpad=10)
    ax2.set_ylabel('Species', fontweight='bold', fontsize=12, labelpad=10)
    ax2.set_xlim(0, rare_species['Prevalence'].max() * 1.3)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='x')
    ax2.set_facecolor(COLORS['background'])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    # Removed summary textbox and background text to reduce clutter
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_composition.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species composition analysis saved.")

# Function to analyze species richness change over time periods
def analyze_richness_change(df):
    """
    Analyze how species richness has changed between early monitoring period and recent years.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Analyzing species richness change over time...")
    
    # Define early and recent time periods
    early_period = (1996, 2000)  # First 5 years of monitoring
    recent_period = (2018, 2023)  # Last 5 years of data
    
    # Filter data for these periods
    early_data = df[(df['Year'] >= early_period[0]) & (df['Year'] <= early_period[1])]
    recent_data = df[(df['Year'] >= recent_period[0]) & (df['Year'] <= recent_period[1])]
    
    # Calculate average species richness by site for each period
    early_richness = early_data.groupby(['Site_name', 'Habitat', 'Subregion'])['species_richness'].mean().reset_index()
    recent_richness = recent_data.groupby(['Site_name', 'Habitat', 'Subregion'])['species_richness'].mean().reset_index()
    
    # Merge the two periods
    merged_data = pd.merge(early_richness, recent_richness, 
                          on=['Site_name', 'Habitat', 'Subregion'], 
                          suffixes=('_early', '_recent'))
    
    # Calculate change in species richness
    merged_data['richness_change'] = merged_data['species_richness_recent'] - merged_data['species_richness_early']
    merged_data['percent_change'] = (merged_data['richness_change'] / merged_data['species_richness_early']) * 100
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), facecolor=COLORS['background'])
    fig.subplots_adjust(hspace=0.3)
    
    # Define habitat color mapping
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    # Plot 1: Top and bottom performers (absolute change)
    # Sort by absolute change
    top_bottom_sites = merged_data.sort_values('richness_change')
    
    # Get top 5 and bottom 5 sites
    bottom_5 = top_bottom_sites.head(5)
    top_5 = top_bottom_sites.tail(5)
    plot_data = pd.concat([bottom_5, top_5])
    
    # Create bar colors based on habitat
    bar_colors = [habitat_colors.get(habitat, 'gray') for habitat in plot_data['Habitat']]
    
    # Create horizontal bar chart
    bars1 = ax1.barh(plot_data['Site_name'], plot_data['richness_change'], 
                    color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels to the bars
    for bar in bars1:
        width = bar.get_width()
        label_x = width + 0.1 if width >= 0 else width - 0.1
        ha = 'left' if width >= 0 else 'right'
        ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha=ha, va='center', fontweight='bold')
    
    # Add a vertical line at x=0
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set plot aesthetics
    ax1.set_title(f'Sites with Largest Changes in Species Richness ({early_period[0]}-{early_period[1]} vs {recent_period[0]}-{recent_period[1]})', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    ax1.set_xlabel('Change in Number of Coral Species', fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_ylabel('Site Name', fontweight='bold', fontsize=12, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='x')
    ax1.set_facecolor(COLORS['background'])
    
    # Plot 2: Scatter plot of early vs recent richness
    for habitat, color in habitat_colors.items():
        habitat_data = merged_data[merged_data['Habitat'] == habitat]
        if not habitat_data.empty:
            ax2.scatter(habitat_data['species_richness_early'], 
                       habitat_data['species_richness_recent'],
                       color=color, alpha=0.7, s=80, edgecolor='white', linewidth=1,
                       label=f'{habitat}')
    
    # Add diagonal line (no change line)
    max_val = max(merged_data['species_richness_early'].max(), merged_data['species_richness_recent'].max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Change')
    
    # Set plot aesthetics
    ax2.set_title(f'Species Richness Comparison: {early_period[0]}-{early_period[1]} vs {recent_period[0]}-{recent_period[1]}', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    ax2.set_xlabel(f'Species Richness ({early_period[0]}-{early_period[1]})', fontweight='bold', fontsize=12, labelpad=10)
    ax2.set_ylabel(f'Species Richness ({recent_period[0]}-{recent_period[1]})', fontweight='bold', fontsize=12, labelpad=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.set_facecolor(COLORS['background'])
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    
    # Add legend
    legend = ax2.legend(title='Habitat Type', frameon=True, facecolor='white', 
                       framealpha=0.9, fontsize=10, loc='upper left')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    
    # Add a summary textbox
    avg_change = merged_data['richness_change'].mean()
    pct_sites_decreased = (merged_data['richness_change'] < 0).mean() * 100
    max_increase = merged_data['richness_change'].max()
    max_decrease = merged_data['richness_change'].min()
    
    summary_text = (
        f"SUMMARY:\n"
        f"• Average change in species richness: {avg_change:.2f} species\n"
        f"• {pct_sites_decreased:.1f}% of sites showed a decrease in species richness\n"
        f"• Maximum increase: {max_increase:.1f} species\n"
        f"• Maximum decrease: {max_decrease:.1f} species"
    )
    
    # Removed summary textbox to reduce clutter
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_change.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species richness change analysis saved.")

# Function to analyze species richness by depth range
def analyze_depth_richness_patterns(df, stations_df):
    """
    Analyze how coral species richness varies across depth gradients.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
        stations_df (DataFrame): DataFrame with station metadata including depth
    """
    print("Analyzing depth distribution patterns...")
    
    # Make a copy to avoid modifying the original dataframe
    analysis_df = df.copy()
    
    # Merge depth information if not already present
    if 'Depth_ft' not in analysis_df.columns and 'StationID' in analysis_df.columns:
        try:
            # Convert station IDs to string if needed
            analysis_df['StationID'] = analysis_df['StationID'].astype(str)
            stations_df['StationID'] = stations_df['StationID'].astype(str)
            
            # Merge with station data to get depth information
            analysis_df = pd.merge(analysis_df, stations_df[['StationID', 'Depth_ft']], 
                         on='StationID', how='left')
            
            print(f"Merged depth information. {analysis_df['Depth_ft'].isna().sum()} records have missing depth data.")
        except Exception as e:
            print(f"Error merging depth data: {str(e)}")
            print("Continuing with available data...")
    
    # Check if we have depth data to analyze
    if 'Depth_ft' not in analysis_df.columns or analysis_df['Depth_ft'].isna().all():
        print("No depth data available for analysis. Skipping depth analysis.")
        return
    
    # Create depth categories for analysis
    try:
        analysis_df['Depth_Range'] = pd.cut(
            analysis_df['Depth_ft'],
            bins=[0, 10, 20, 30, 100],
            labels=['0-10 ft', '11-20 ft', '21-30 ft', '30+ ft']
        )
        
        # Drop records with missing depth range
        analysis_df = analysis_df.dropna(subset=['Depth_Range'])
        
        if len(analysis_df) == 0:
            print("No valid depth data after categorization. Skipping depth analysis.")
            return
            
        print(f"Created depth ranges with {len(analysis_df)} valid records.")
    except Exception as e:
        print(f"Error creating depth categories: {str(e)}")
        print("Skipping depth analysis.")
        return
    
    # Calculate average species richness by depth range
    depth_richness = analysis_df.groupby('Depth_Range', observed=False)['species_richness'].agg(['mean', 'std', 'count']).reset_index()
    
    # If we have too few data points, skip the analysis
    if len(depth_richness) < 2:
        print("Insufficient depth range categories for analysis. Skipping depth analysis.")
        return
    
    depth_richness['se'] = depth_richness['std'] / np.sqrt(depth_richness['count'])
    depth_richness['ci_95'] = 1.96 * depth_richness['se']
    
    # Calculate average species richness by depth range and year for trend analysis
    depth_year_richness = analysis_df.groupby(['Year', 'Depth_Range'], observed=False)['species_richness'].mean().reset_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), facecolor=COLORS['background'])
    fig.subplots_adjust(hspace=0.3)
    
    # Plot 1: Bar chart of average species richness by depth range
    # Define depth range color gradient (shallow to deep)
    depth_colors = [
        COLORS['light_blue'],  # 0-10 ft
        COLORS['ocean_blue'],  # 11-20 ft
        COLORS['dark_blue'],   # 21-30 ft
        '#01233D'              # 30+ ft (darker blue)
    ]
    
    # Make sure we have enough colors for all depth ranges
    if len(depth_richness) > len(depth_colors):
        # Generate additional colors if needed
        depth_colors = sns.color_palette("Blues", len(depth_richness))
    
    # Create bar chart for depth ranges
    bars = ax1.bar(
        depth_richness['Depth_Range'],
        depth_richness['mean'],
        yerr=depth_richness['ci_95'],
        color=depth_colors[:len(depth_richness)],  # Use only as many colors as we have depth ranges
        alpha=0.8,
        edgecolor='white',
        linewidth=1,
        capsize=8,
        error_kw={'ecolor': 'gray', 'lw': 1.5, 'alpha': 0.8}
    )
    
    # Add value labels to the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = depth_richness['count'].iloc[i]
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f'{height:.1f}\n(n={count})',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    # Perform statistical test (ANOVA) to determine if differences are significant
    # Group data by depth range for ANOVA
    depth_groups = []
    depth_ranges = []
    
    for depth_range in analysis_df['Depth_Range'].dropna().unique():
        depth_data = analysis_df[analysis_df['Depth_Range'] == depth_range]['species_richness'].dropna()
        if len(depth_data) > 0:
            depth_groups.append(depth_data)
            depth_ranges.append(depth_range)
    
    if len(depth_groups) >= 2:  # Need at least 2 groups for ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*depth_groups)
            
            # Add statistical test result to the plot
            significance_text = f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}"
            if p_value < 0.05:
                significance_text += " (Significant)"
            
            ax1.text(
                0.5, 0.95,
                significance_text,
                transform=ax1.transAxes,
                ha='center',
                va='top',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['coral'], boxstyle='round,pad=0.5')
            )
        except Exception as e:
            print(f"Error performing ANOVA: {str(e)}")
    
    # Set plot aesthetics
    ax1.set_title('Stony Coral Species Richness by Depth Range', 
                 fontweight='bold', fontsize=18, pad=15,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax1.set_xlabel('Depth Range', fontweight='bold', fontsize=14, labelpad=10)
    ax1.set_ylabel('Number of Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
    ax1.set_facecolor(COLORS['background'])
    
    # Plot 2: Line chart showing trends over time by depth range
    # Check if we have enough data for time trends
    if len(depth_year_richness) < 5:
        ax2.text(0.5, 0.5, "Insufficient data for temporal trend analysis", 
               transform=ax2.transAxes, ha='center', va='center', 
               fontsize=14, fontweight='bold')
    else:
        # Get unique depth ranges
        unique_depth_ranges = sorted(analysis_df['Depth_Range'].dropna().unique())
        
        # Plot each depth range with enhanced styling
        for i, depth_range in enumerate(unique_depth_ranges):
            if str(depth_range) != 'nan':  # Skip NaN values
                range_data = depth_year_richness[depth_year_richness['Depth_Range'] == depth_range]
                if not range_data.empty and len(range_data) > 1:  # Need at least 2 points for a line
                    color_idx = min(i, len(depth_colors) - 1)  # Ensure we don't go out of bounds
                    ax2.plot(
                        range_data['Year'], 
                        range_data['species_richness'], 
                        marker='o', 
                        color=depth_colors[color_idx], 
                        linewidth=2.5, 
                        markersize=7, 
                        label=depth_range,
                        path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()]
                    )
                    
                    # Add trend line for each depth range
                    if len(range_data) > 2:  # Only add trend line if we have enough data points
                        try:
                            X = range_data['Year'].values.reshape(-1, 1)
                            y = range_data['species_richness'].values
                            model = LinearRegression().fit(X, y)
                            y_pred = model.predict(X)
                            slope = model.coef_[0]
                            
                            ax2.plot(range_data['Year'], y_pred, '--', 
                                  color=depth_colors[color_idx], 
                                  alpha=0.7, linewidth=1.5)
                            
                            # Add slope annotation
                            last_year = range_data['Year'].max()
                            last_y_pred = y_pred[-1]
                            ax2.annotate(
                                f"Slope: {slope:.3f}",
                                xy=(last_year, last_y_pred),
                                xytext=(last_year + 0.5, last_y_pred),
                                fontsize=9,
                                color=depth_colors[color_idx],
                                fontweight='bold',
                                va='center'
                            )
                        except Exception as e:
                            print(f"Error calculating trend line for {depth_range}: {str(e)}")
    
    # Set plot aesthetics
    ax2.set_title('Trends in Stony Coral Species Richness by Depth Range (1996-2023)', 
                 fontweight='bold', fontsize=18, pad=15,
                 color=COLORS['dark_blue'],
                 path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax2.set_ylabel('Number of Coral Species', fontweight='bold', fontsize=14, labelpad=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.set_facecolor(COLORS['background'])
    
    # Set axis limits for the trend plot if we have data
    if not depth_year_richness.empty:
        ax2.set_xlim(depth_year_richness['Year'].min() - 0.5, depth_year_richness['Year'].max() + 2)
        y_min = max(0, depth_year_richness['species_richness'].min() * 0.9)
        y_max = depth_year_richness['species_richness'].max() * 1.1
        ax2.set_ylim(y_min, y_max)
    
    # Enhance the legend
    handles, labels = ax2.get_legend_handles_labels()
    if handles:  # Only add legend if we have data
        legend = ax2.legend(
            handles=handles,
            labels=labels,
            title='Depth Range', 
            frameon=True, 
            facecolor='white', 
            framealpha=0.9, 
            fontsize=12, 
            loc='upper right', 
            edgecolor=COLORS['grid']
        )
        legend.get_frame().set_linewidth(1.5)
        legend.get_title().set_fontweight('bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_by_depth.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Depth analysis saved.")

# Function to perform statistical correlation analysis
def perform_correlation_analysis(df):
    """
    Perform statistical correlation analysis to identify factors related to species richness.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with species richness data
    """
    print("Performing correlation analysis...")
    
    # Create a copy of the dataframe to avoid modifying the original
    analysis_df = df.copy()
    
    # Create time-related variables that might correlate with richness
    analysis_df['Years_Since_Start'] = analysis_df['Year'] - analysis_df['Year'].min()
    
    # Add decade categorical variable for analysis
    analysis_df['Decade'] = (analysis_df['Year'] // 10) * 10
    
    # Calculate average richness by year for time trend analysis
    yearly_avg = analysis_df.groupby('Year')['species_richness'].mean().reset_index()
    yearly_avg['Years_Since_Start'] = yearly_avg['Year'] - yearly_avg['Year'].min()
    
    # Create figure with multiple subplots for different analyses
    fig = plt.figure(figsize=(18, 15), facecolor=COLORS['background'])
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    gs.update(wspace=0.25, hspace=0.3)
    
    # Plot 1: Correlation between time and species richness (time series analysis)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['background'])
    
    # Scatter plot with regression line
    sns.regplot(
        x='Years_Since_Start', 
        y='species_richness', 
        data=yearly_avg,
        scatter_kws={'color': COLORS['coral'], 's': 100, 'alpha': 0.7, 'edgecolor': 'white'},
        line_kws={'color': COLORS['dark_blue'], 'lw': 2},
        ax=ax1
    )
    
    # Calculate Pearson correlation
    corr, p_value = stats.pearsonr(yearly_avg['Years_Since_Start'], yearly_avg['species_richness'])
    
    # Add correlation statistics
    stat_text = f"Pearson correlation: r = {corr:.3f}\np-value: {p_value:.4f}"
    if p_value < 0.05:
        stat_text += " (Significant)"
    
    ax1.text(
        0.05, 0.95,
        stat_text,
        transform=ax1.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['ocean_blue'], boxstyle='round,pad=0.5')
    )
    
    # Set plot aesthetics
    ax1.set_title('Correlation: Time vs. Species Richness', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    
    ax1.set_xlabel('Years Since Monitoring Started (1996)', fontweight='bold', fontsize=12, labelpad=10)
    ax1.set_ylabel('Average Species Richness', fontweight='bold', fontsize=12, labelpad=10)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 2: Species richness by decade (boxplot analysis)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['background'])
    
    # Define decade color gradient
    decade_colors = {
        1990: COLORS['light_blue'],
        2000: COLORS['ocean_blue'],
        2010: COLORS['dark_blue'],
        2020: COLORS['coral']
    }
    
    # Create custom color palette
    decades = sorted(analysis_df['Decade'].unique())
    palette = [decade_colors.get(decade, 'gray') for decade in decades]
    
    # Create boxplot for decades with enhanced styling and proper hue parameter
    boxplot = sns.boxplot(
        x='Decade',
        y='species_richness',
        hue='Decade',  # Add hue parameter to avoid warning
        palette=palette,
        data=analysis_df,
        width=0.6,
        linewidth=1.5,
        fliersize=5,
        legend=False,  # Don't show the legend since it's redundant
        ax=ax2
    )
    
    # Instead of swarmplot, use stripplot with smaller size to avoid overcrowding
    sns.stripplot(
        x='Decade',
        y='species_richness',
        data=analysis_df,
        size=2,  # Smaller point size
        color='black',
        alpha=0.3,
        jitter=True,  # Add jitter to avoid overlapping
        ax=ax2
    )
    
    # Perform ANOVA to determine if differences between decades are significant
    decades = []
    decade_groups = []
    
    for decade in sorted(analysis_df['Decade'].unique()):
        decade_data = analysis_df[analysis_df['Decade'] == decade]['species_richness'].dropna()
        if len(decade_data) > 0:
            decades.append(decade)
            decade_groups.append(decade_data)
    
    if len(decade_groups) >= 2:  # Need at least 2 groups for ANOVA
        f_stat, p_value = stats.f_oneway(*decade_groups)
        
        # Add statistical test result to the plot
        significance_text = f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}"
        if p_value < 0.05:
            significance_text += " (Significant)"
        
        ax2.text(
            0.5, 0.05,
            significance_text,
            transform=ax2.transAxes,
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['coral'], boxstyle='round,pad=0.5')
        )
    
    # Set plot aesthetics
    ax2.set_title('Species Richness Distribution by Decade', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    
    ax2.set_xlabel('Decade', fontweight='bold', fontsize=12, labelpad=10)
    ax2.set_ylabel('Species Richness', fontweight='bold', fontsize=12, labelpad=10)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
    
    # Plot 3: Correlation matrix of species richness with other factors
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS['background'])
    
    # Create correlation dataframe with relevant variables
    # Check if Depth_ft exists, if not skip it
    corr_columns = ['species_richness', 'Year', 'Years_Since_Start']
    if 'Depth_ft' in analysis_df.columns:
        corr_columns.append('Depth_ft')
    
    corr_df = analysis_df[corr_columns].copy()
    corr_df = corr_df.dropna()  # Remove NaN values for correlation calculation
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create heatmap of correlation matrix
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=coral_cmap,
        linewidths=1,
        linecolor='white',
        square=True,
        fmt=".3f",
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax3
    )
    
    # Set plot aesthetics
    ax3.set_title('Correlation Matrix: Species Richness vs. Other Factors', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    
    # Plot 4: Species richness by habitat and region (multi-factor analysis)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS['background'])
    
    # Create multi-factor bubble plot
    # Calculate average species richness by habitat and region
    hab_reg_richness = analysis_df.groupby(['Habitat', 'Subregion'])['species_richness'].mean().reset_index()
    hab_reg_count = analysis_df.groupby(['Habitat', 'Subregion'])['species_richness'].count().reset_index()
    hab_reg_richness['count'] = hab_reg_count['species_richness']
    
    # Define habitat and region color mappings
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    region_markers = {
        'UK': 'o',  # Upper Keys - circle
        'MK': 's',  # Middle Keys - square
        'LK': '^'   # Lower Keys - triangle
    }
    
    region_names = {
        'UK': 'Upper Keys',
        'MK': 'Middle Keys',
        'LK': 'Lower Keys'
    }
    
    habitat_names = {
        'OS': 'Offshore Shallow',
        'OD': 'Offshore Deep',
        'P': 'Patch Reef',
        'HB': 'Hardbottom',
        'BCP': 'Backcountry Patch'
    }
    
    # Jitter the x positions to avoid overlap
    habitats = sorted(hab_reg_richness['Habitat'].unique())
    x_positions = {hab: i+1 for i, hab in enumerate(habitats)}
    
    # Plot bubbles for each habitat-region combination
    legend_elements = []
    
    # First, add habitat color legend elements
    for hab, name in habitat_names.items():
        if hab in habitats:  # Only add if this habitat exists in the data
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=habitat_colors.get(hab, 'gray'),
                        markersize=10, label=name)
            )
    
    # Then add region marker legend elements
    for reg, name in region_names.items():
        if reg in hab_reg_richness['Subregion'].unique():  # Only add if this region exists in the data
            legend_elements.append(
                plt.Line2D([0], [0], marker=region_markers.get(reg, 'o'), color='black',
                        markersize=10, linestyle='None', label=name)
            )
    
    # Define jitter amount for region separation
    jitter = 0.2
    
    # Plot each habitat-region combination
    for _, row in hab_reg_richness.iterrows():
        habitat = row['Habitat']
        region = row['Subregion']
        richness = row['species_richness']
        count = row['count']
        
        # Determine x position with jitter based on region
        if region == 'UK':
            x_pos = x_positions[habitat] - jitter
        elif region == 'MK':
            x_pos = x_positions[habitat]
        else:  # 'LK'
            x_pos = x_positions[habitat] + jitter
        
        # Scale bubble size by count (number of observations)
        size = 100 + (count * 5)  # Base size + scaling factor
        
        # Plot the bubble
        ax4.scatter(
            x_pos, richness,
            s=size,
            color=habitat_colors.get(habitat, 'gray'),
            marker=region_markers.get(region, 'o'),
            alpha=0.7,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add label for each bubble
        ax4.text(
            x_pos, richness + 0.3,
            f"{region}",
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )
    
    # Set plot aesthetics
    ax4.set_title('Species Richness by Habitat Type and Region', 
                 fontweight='bold', fontsize=16, pad=15,
                 color=COLORS['dark_blue'])
    
    ax4.set_xlabel('Habitat Type', fontweight='bold', fontsize=12, labelpad=10)
    ax4.set_ylabel('Average Species Richness', fontweight='bold', fontsize=12, labelpad=10)
    ax4.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--', axis='y')
    
    # Set x-axis ticks and labels
    ax4.set_xticks(list(x_positions.values()))
    ax4.set_xticklabels([habitat_names.get(h, h) for h in x_positions.keys()], rotation=45, ha='right')
    
    # Set y-axis limits with some padding
    if not hab_reg_richness.empty:  # Check if dataframe is not empty
        y_min = hab_reg_richness['species_richness'].min() * 0.9
        y_max = hab_reg_richness['species_richness'].max() * 1.1
        ax4.set_ylim(y_min, y_max)
    
    # Add legend
    if legend_elements:  # Only add legend if there are elements
        ax4.legend(
            handles=legend_elements,
            title="Legend",
            loc='upper right',
            frameon=True,
            facecolor='white',
            framealpha=0.9,
            fontsize=9
        )
    
    # Add title for the entire figure
    fig.suptitle(
        'COMPREHENSIVE STATISTICAL ANALYSIS OF STONY CORAL SPECIES RICHNESS',
        fontweight='bold',
        fontsize=22,
        color=COLORS['dark_blue'],
        y=0.98,
        path_effects=[pe.withStroke(linewidth=2, foreground='white')]
    )
    
    # Add a note about the data source
    # fig.text(
    #     0.5, 0.01,
    #     "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)\nStatistical Analysis: Pearson correlation, ANOVA, and multi-factor comparison",
    #     ha='center',
    #     va='center',
    #     fontsize=10,
    #     fontstyle='italic',
    #     color=COLORS['text']
    # )
    
    # Use fig.tight_layout with padding to avoid warnings
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    plt.savefig(os.path.join(results_dir, "stony_coral_species_richness_correlation_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Correlation analysis saved.")

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    species_df, stations_df, species_cols = load_and_preprocess_data()
    
    # Create overall trend plot
    plot_overall_trend(species_df)
    
    # Create regional trends plot
    plot_trends_by_region(species_df)
    
    # Create habitat-based trends plot
    plot_trends_by_habitat(species_df)
    
    # Create temporal heatmap
    create_temporal_heatmap(species_df)
    
    # Analyze species composition
    analyze_species_composition(species_df, species_cols)
    
    # Analyze species richness change over time
    analyze_richness_change(species_df)
    
    # New analyses
    # Analyze depth patterns
    analyze_depth_richness_patterns(species_df, stations_df)
    
    # Perform statistical correlation analysis
    perform_correlation_analysis(species_df)
    
    print("\nAll analyses completed successfully!")