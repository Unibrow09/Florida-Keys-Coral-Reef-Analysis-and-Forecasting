"""
03_octocoral_density_analysis.py - Analysis of Octocoral Density Variations

This script analyzes the variations in octocoral density across different
stations and over time during the CREMP study period. It explores trends over time
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
results_dir = "03_Results"
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

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP dataset for octocoral density analysis.
    
    Returns:
        tuple: (octo_df, stations_df) - Preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Octocoral Density dataset
        octo_df = pd.read_csv("CREMP_CSV_files/CREMP_OCTO_Summaries_2023_Density.csv")
        print(f"Octocoral density data loaded successfully with {len(octo_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Print column names to verify data structure
        print("\nOctocoral density data columns:", octo_df.columns.tolist())
        
        # Convert date column to datetime format
        octo_df['Date'] = pd.to_datetime(octo_df['Date'])
        
        # Extract just the year for easier grouping
        octo_df['Year'] = octo_df['Year'].astype(int)
        
        # Get list of all octocoral species columns (excluding metadata columns)
        metadata_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                         'Site_name', 'StationID', 'Total_Octocorals']
        species_cols = [col for col in octo_df.columns if col not in metadata_cols]
        
        print(f"\nIdentified {len(species_cols)} octocoral species in the dataset")
        
        # Calculate total density for each station if not already present
        if 'Total_Octocorals' in octo_df.columns:
            # Fill missing values in Total_Octocorals by summing species columns
            octo_df['Total_Octocorals'] = octo_df[species_cols].sum(axis=1, skipna=True)
        else:
            # Create Total_Octocorals column by summing all species columns
            octo_df['Total_Octocorals'] = octo_df[species_cols].sum(axis=1, skipna=True)
        
        print(f"\nData loaded: {len(octo_df)} records from {octo_df['Year'].min()} to {octo_df['Year'].max()}")
        
        return octo_df, stations_df, species_cols
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to plot overall trend of octocoral density over time
def plot_overall_trend(df):
    """
    Plot the overall trend of octocoral density over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Creating overall trend plot...")
    
    # Group by year and calculate mean octocoral density
    yearly_avg = df.groupby('Year')['Total_Octocorals'].agg(['mean', 'std', 'count']).reset_index()
    yearly_avg['se'] = yearly_avg['std'] / np.sqrt(yearly_avg['count'])
    yearly_avg['ci_95'] = 1.96 * yearly_avg['se']
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the mean line with enhanced styling
    ax.plot(yearly_avg['Year'], yearly_avg['mean'], marker='o', color=COLORS['coral'], 
            linewidth=3.5, markersize=10, label='Mean Octocoral Density',
            path_effects=[pe.SimpleLineShadow(offset=(2, -2), alpha=0.3), pe.Normal()])
    
    # Add 95% confidence interval band with enhanced styling
    ax.fill_between(yearly_avg['Year'], 
                    yearly_avg['mean'] - yearly_avg['ci_95'],
                    yearly_avg['mean'] + yearly_avg['ci_95'],
                    color=COLORS['coral'], alpha=0.2, label='95% Confidence Interval')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set plot aesthetics
    ax.set_title('EVOLUTION OF OCTOCORAL DENSITY (2011-2023)', 
                fontweight='bold', fontsize=20, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
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
            linewidth=2, label=f'Linear Trend (Slope: {slope:.4f} colonies/m²/year)',
            path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
    
    # Update legend with the new trend line
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, facecolor='white', framealpha=0.9, 
             fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    
    # Add a summary textbox
    summary_text = (
        f"SUMMARY:\n"
        f"• Overall trend: {slope:.4f} colonies/m² per year\n"
        f"• Maximum density: {yearly_avg['mean'].max():.2f} colonies/m² ({yearly_avg.loc[yearly_avg['mean'].idxmax(), 'Year']})\n"
        f"• Minimum density: {yearly_avg['mean'].min():.2f} colonies/m² ({yearly_avg.loc[yearly_avg['mean'].idxmin(), 'Year']})\n"
        f"• Current level (2023): {yearly_avg.loc[yearly_avg['Year'] == yearly_avg['Year'].max(), 'mean'].values[0]:.2f} colonies/m²"
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
    plt.savefig(os.path.join(results_dir, "octocoral_density_overall_trend.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Overall octocoral density trend plot saved.")

# Function to plot trends by region
def plot_trends_by_region(df):
    """
    Plot octocoral density trends separated by region (Upper Keys, Middle Keys, Lower Keys).
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Creating regional trends plot...")
    
    # Group by year and region, calculate mean octocoral density
    region_yearly_avg = df.groupby(['Year', 'Subregion'])['Total_Octocorals'].mean().reset_index()
    
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
            ax.plot(region_data['Year'], region_data['Total_Octocorals'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=region_names.get(region, region),
                    path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
            
            # Add trend line for each region
            X = region_data['Year'].values.reshape(-1, 1)
            y = region_data['Total_Octocorals'].values
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
    ax.set_title('Octocoral Density by Region (2011-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set axis limits
    ax.set_xlim(region_yearly_avg['Year'].min() - 0.5, region_yearly_avg['Year'].max() + 2)
    ax.set_ylim(0, region_yearly_avg['Total_Octocorals'].max() * 1.1)
    
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
    plt.savefig(os.path.join(results_dir, "octocoral_density_by_region.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Regional octocoral density trends plot saved.")

# Function to plot trends by habitat type
def plot_trends_by_habitat(df):
    """
    Plot octocoral density trends separated by habitat type.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Creating habitat-based trends plot...")
    
    # Group by year and habitat, calculate mean octocoral density
    habitat_yearly_avg = df.groupby(['Year', 'Habitat'])['Total_Octocorals'].mean().reset_index()
    
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
            ax.plot(habitat_data['Year'], habitat_data['Total_Octocorals'], marker='o', color=color, 
                    linewidth=2.5, markersize=7, label=habitat_names.get(habitat, habitat),
                    path_effects=[pe.SimpleLineShadow(offset=(1, -1), alpha=0.2), pe.Normal()])
            
            # Add trend line for each habitat
            if len(habitat_data) > 2:  # Only add trend line if we have enough data points
                X = habitat_data['Year'].values.reshape(-1, 1)
                y = habitat_data['Total_Octocorals'].values
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
    ax.set_title('Octocoral Density by Habitat Type (2011-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set axis limits
    ax.set_xlim(habitat_yearly_avg['Year'].min() - 0.5, habitat_yearly_avg['Year'].max() + 2)
    ax.set_ylim(0, habitat_yearly_avg['Total_Octocorals'].max() * 1.1)
    
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
    plt.savefig(os.path.join(results_dir, "octocoral_density_by_habitat.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Habitat-based octocoral density trends plot saved.")

# Function to create a temporal heatmap of octocoral density
def create_temporal_heatmap(df):
    """
    Create a heatmap visualization of octocoral density changes over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Creating temporal heatmap...")
    
    # Get a sample of sites to visualize (to avoid overcrowding)
    # Focus on sites that have been surveyed for the longest periods
    site_survey_counts = df.groupby('Site_name')['Year'].nunique().sort_values(ascending=False)
    sites_to_include = site_survey_counts[site_survey_counts > 5].index.tolist()[:15]  # Top 15 most surveyed sites
    
    # Filter data for these sites and pivot to create a matrix suitable for a heatmap
    heatmap_data = df[df['Site_name'].isin(sites_to_include)]
    
    # Calculate average for each site and year
    heatmap_pivot = heatmap_data.groupby(['Site_name', 'Year'])['Total_Octocorals'].mean().reset_index()
    heatmap_pivot = heatmap_pivot.pivot(index='Site_name', columns='Year', values='Total_Octocorals')
    
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
        cmap=octocoral_cmap, 
        ax=ax, 
        vmin=0, 
        vmax=max(20, heatmap_pivot.max().max()),  # Set reasonable limits
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={
            'label': 'Octocoral Density (colonies/m²)', 
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
    cbar.set_label('Octocoral Density (colonies/m²)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set plot aesthetics
    ax.set_title('Temporal Evolution of Octocoral Density by Site (2011-2023)', 
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
    
    plt.savefig(os.path.join(results_dir, "octocoral_density_temporal_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal heatmap saved.")

# Function to analyze species composition
def analyze_species_composition(df, species_cols):
    """
    Analyze and visualize the composition of octocoral species.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names for octocoral species
    """
    print("Analyzing octocoral species composition...")
    
    # Calculate the mean density for each species across all stations and years
    species_means = df[species_cols].mean().sort_values(ascending=False)
    
    # Select top 15 most abundant species for visualization
    top_species = species_means.head(15)
    
    # Create a figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create horizontal bar chart
    bars = ax.barh(top_species.index, top_species.values, 
             color=plt.cm.viridis(np.linspace(0, 0.8, len(top_species))),
             height=0.7, alpha=0.8)
    
    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', fontweight='bold', fontsize=10)
    
    # Set plot aesthetics
    ax.set_title('Most Abundant Octocoral Species (2011-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Species', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_species_composition.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Octocoral species composition analysis saved.")

# Function to create a heatmap of octocoral density trends
def create_temporal_heatmap(df):
    """
    Create a heatmap visualization of octocoral density changes over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Creating temporal heatmap...")
    
    # Get a sample of sites to visualize (to avoid overcrowding)
    # Focus on sites that have been surveyed for the longest periods
    site_survey_counts = df.groupby('Site_name')['Year'].nunique().sort_values(ascending=False)
    sites_to_include = site_survey_counts[site_survey_counts > 3].index.tolist()[:15]  # Top 15 most surveyed sites
    
    # Filter data for these sites and pivot to create a matrix suitable for a heatmap
    heatmap_data = df[df['Site_name'].isin(sites_to_include)]
    
    # Calculate average for each site and year
    heatmap_pivot = heatmap_data.groupby(['Site_name', 'Year'])['Total_Octocorals'].mean().reset_index()
    heatmap_pivot = heatmap_pivot.pivot(index='Site_name', columns='Year', values='Total_Octocorals')
    
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
        cmap=octocoral_cmap, 
        ax=ax, 
        vmin=0, 
        vmax=max(15, heatmap_pivot.max().max()),  # Set reasonable limits
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={
            'label': 'Octocoral Density (colonies/m²)', 
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
    cbar.set_label('Octocoral Density (colonies/m²)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set plot aesthetics
    ax.set_title('Temporal Evolution of Octocoral Density by Site (2011-2023)', 
                fontweight='bold', pad=20, fontsize=20,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Improve X and Y labels
    ax.set_xlabel('Year', labelpad=15, fontweight='bold', fontsize=14)
    ax.set_ylabel('Site Name', labelpad=15, fontweight='bold', fontsize=14)
    
    # Improve tick labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_density_temporal_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Temporal heatmap of octocoral density saved.")

# Function to plot species composition
def plot_species_composition(df, species_cols):
    """
    Plot the composition of octocoral species across the study period.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names representing octocoral species
    """
    print("Creating species composition plot...")
    
    # Calculate mean density for each species by year
    species_yearly = df.groupby('Year')[species_cols].mean().reset_index()
    
    # Melt the dataframe for easier plotting
    species_melted = pd.melt(species_yearly, id_vars=['Year'], value_vars=species_cols, 
                            var_name='Species', value_name='Density')
    
    # Replace underscores with spaces in species names for better readability
    species_melted['Species'] = species_melted['Species'].str.replace('_', ' ')
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create a stacked area plot
    sns.lineplot(data=species_melted, x='Year', y='Density', hue='Species', 
                marker='o', linewidth=2.5, markersize=7)
    
    # Set plot aesthetics
    ax.set_title('Octocoral Species Composition (2011-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=10, loc='upper right', edgecolor=COLORS['grid'],
                      title='Octocoral Species')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_species_composition.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Octocoral species composition plot saved.")

# Function to plot spatial distribution of octocoral density
def plot_spatial_distribution(df, stations_df):
    """
    Plot the spatial distribution of octocoral density across different stations.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        stations_df (DataFrame): DataFrame with station metadata including coordinates
    """
    print("Creating spatial distribution plot...")
    
    # Merge octocoral data with station coordinates
    # First, calculate average density for each station across all years
    station_avg = df.groupby('StationID')['Total_Octocorals'].mean().reset_index()
    
    # Merge with stations data to get coordinates
    if 'StationID' in stations_df.columns:
        merged_data = pd.merge(station_avg, stations_df, on='StationID', how='inner')
    else:
        print("Warning: Cannot create spatial distribution plot - StationID not found in stations data")
        return
    
    # Check if we have latitude and longitude columns
    if 'Latitude' not in merged_data.columns or 'Longitude' not in merged_data.columns:
        print("Warning: Cannot create spatial distribution plot - coordinates not found")
        return
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Create a scatter plot with point size proportional to octocoral density
    scatter = ax.scatter(merged_data['Longitude'], merged_data['Latitude'], 
                        c=merged_data['Total_Octocorals'], cmap=octocoral_cmap,
                        s=merged_data['Total_Octocorals'] * 10, # Scale point size
                        alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label('Octocoral Density (colonies/m²)', fontsize=12, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    
    # Set plot aesthetics
    ax.set_title('Spatial Distribution of Octocoral Density', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Longitude', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Latitude', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_spatial_distribution.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Spatial distribution of octocoral density plot saved.")

# Function to analyze rate of change in octocoral density
def analyze_rate_of_change(df):
    """
    Analyze and visualize the rate of change in octocoral density over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Analyzing rate of change...")
    
    # Group by year and calculate mean octocoral density
    yearly_avg = df.groupby('Year')['Total_Octocorals'].mean().reset_index()
    
    # Calculate year-over-year change
    yearly_avg['Change'] = yearly_avg['Total_Octocorals'].diff()
    yearly_avg['Percent_Change'] = yearly_avg['Total_Octocorals'].pct_change() * 100
    
    # Filter out the first year which will have NaN for change
    yearly_avg = yearly_avg.dropna()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, facecolor=COLORS['background'])
    fig.subplots_adjust(hspace=0.3)
    
    # Plot absolute change
    bars1 = ax1.bar(yearly_avg['Year'], yearly_avg['Change'], 
             color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['Change']], 
             alpha=0.7, width=0.7)
    
    ax1.set_title('Absolute Change in Octocoral Density (Year-over-Year)', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax1.set_ylabel('Change in Density (colonies/m²)', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add data labels to the bars
    for bar in bars1:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            y_pos = height - 0.1
        else:
            va = 'bottom'
            y_pos = height + 0.1
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{height:.2f}', ha='center', va=va, fontweight='bold')
    
    # Plot percent change
    bars2 = ax2.bar(yearly_avg['Year'], yearly_avg['Percent_Change'], 
             color=[COLORS['coral'] if x < 0 else COLORS['reef_green'] for x in yearly_avg['Percent_Change']], 
             alpha=0.7, width=0.7)
    
    ax2.set_title('Percent Change in Octocoral Density (Year-over-Year)', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax2.set_xlabel('Year', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Percent Change (%)', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add data labels to the bars
    for bar in bars2:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            y_pos = height - 1
        else:
            va = 'bottom'
            y_pos = height + 1
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{height:.1f}%', ha='center', va=va, fontweight='bold')
    
    # Add key events as vertical lines
    key_events = {
        2014: "2014-2015 Global Bleaching Event",
        2017: "Hurricane Irma",
        2019: "Stony Coral Tissue Loss Disease Peak"
    }
    
    for year, event in key_events.items():
        if year in yearly_avg['Year'].values:
            # Add vertical line to both subplots
            ax1.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
            ax2.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
            
            # Add text annotation to the top subplot only to avoid clutter
            y_pos = ax1.get_ylim()[1] * 0.9
            ax1.annotate(event, xy=(year, y_pos), xytext=(year, y_pos),
                        ha='center', va='center', fontsize=9, rotation=90,
                        color='gray', fontweight='bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_density_rate_of_change.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Rate of change analysis saved.")

# Function to analyze octocoral density by depth range
def analyze_depth_patterns(df, stations_df):
    """
    Analyze how octocoral density varies across different depth gradients.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        stations_df (DataFrame): DataFrame with station metadata including depth information
    """
    print("Analyzing depth distribution patterns...")
    
    # Make a copy of the input DataFrame to avoid modifying the original
    analysis_df = df.copy()
    
    # Check if depth information is already in the dataset
    if 'Depth_ft' not in analysis_df.columns:
        # Merge with stations data to get depth information
        if 'StationID' in stations_df.columns and 'Depth_ft' in stations_df.columns:
            # Convert StationID to string type for both DataFrames before merging
            analysis_df['StationID'] = analysis_df['StationID'].astype(str)
            stations_df['StationID'] = stations_df['StationID'].astype(str)
            
            # Merge depth information
            try:
                analysis_df = pd.merge(analysis_df, 
                                       stations_df[['StationID', 'Depth_ft']], 
                                       on='StationID', how='left')
                print(f"Merged depth information. {analysis_df['Depth_ft'].isna().sum()} records have missing depth data.")
            except Exception as e:
                print(f"Error merging depth data: {str(e)}")
                return
        else:
            print("Warning: Cannot perform depth analysis - depth information not found in stations data")
            return
    
    # Create depth categories
    try:
        # Ensure depth is numeric
        analysis_df['Depth_ft'] = pd.to_numeric(analysis_df['Depth_ft'], errors='coerce')
        
        # Create depth range categories
        analysis_df['Depth_Range'] = pd.cut(
            analysis_df['Depth_ft'],
            bins=[0, 10, 20, 30, 100],  # Adjust bins as needed
            labels=['0-10 ft', '11-20 ft', '21-30 ft', '30+ ft']
        )
        
        # Drop records with missing depth ranges
        analysis_df = analysis_df.dropna(subset=['Depth_Range'])
        print(f"Created depth ranges with {len(analysis_df)} valid records.")
    except Exception as e:
        print(f"Error creating depth categories: {str(e)}")
        return
    
    # Calculate average octocoral density by depth range
    depth_density = analysis_df.groupby('Depth_Range', observed=True)['Total_Octocorals'].agg(['mean', 'median', 'std', 'count']).reset_index()
    depth_density['se'] = depth_density['std'] / np.sqrt(depth_density['count'])
    depth_density['ci_95'] = 1.96 * depth_density['se']
    
    # Check if we have enough data points for analysis
    if len(depth_density) < 2:
        print("Not enough depth categories with data for analysis")
        return
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=COLORS['background'])
    fig.subplots_adjust(wspace=0.3)
    
    # Plot 1: Bar chart of density by depth range
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(depth_density)))
    bars = ax1.bar(depth_density['Depth_Range'], depth_density['mean'], 
                  yerr=depth_density['ci_95'], capsize=10,
                  color=colors, alpha=0.8, width=0.7)
    
    # Add data labels to the bars
    for bar, count in zip(bars, depth_density['count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{height:.2f}\nn={count}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Style the first plot
    ax1.set_title('Octocoral Density by Depth Range', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax1.set_xlabel('Depth Range', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax1.set_ylim(0, depth_density['mean'].max() * 1.2)
    
    # Plot 2: Line chart showing trend with depth
    sorted_depth = depth_density.sort_values(by='Depth_Range')
    depth_order = {depth: i for i, depth in enumerate(sorted_depth['Depth_Range'])}
    
    # Create a gradient color for the line
    colors = plt.cm.ocean(np.linspace(0.2, 0.8, len(depth_density)))
    
    # Plot the line chart
    ax2.plot(range(len(sorted_depth)), sorted_depth['mean'], marker='o', markersize=10,
            color=COLORS['ocean_blue'], linewidth=2.5, zorder=3)
    
    # Add confidence interval shading
    ax2.fill_between(range(len(sorted_depth)),
                    sorted_depth['mean'] - sorted_depth['ci_95'],
                    sorted_depth['mean'] + sorted_depth['ci_95'],
                    color=COLORS['ocean_blue'], alpha=0.2, zorder=2)
    
    # Add individual data points as a scatter plot
    for i, depth_range in enumerate(sorted_depth['Depth_Range']):
        # Get all density values for this depth range
        densities = analysis_df[analysis_df['Depth_Range'] == depth_range]['Total_Octocorals']
        
        # Add jittered scatter points with a single color
        jitter = np.random.normal(0, 0.05, size=len(densities))
        ax2.scatter([i + j for j in jitter], densities, 
                   color=colors[i], alpha=0.5, s=20, zorder=1)
    
    # Style the second plot
    ax2.set_title('Octocoral Density Trend with Depth', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax2.set_xticks(range(len(sorted_depth)))
    ax2.set_xticklabels(sorted_depth['Depth_Range'])
    ax2.set_xlabel('Depth Range (Increasing Depth →)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.set_ylim(0, analysis_df['Total_Octocorals'].max() * 1.05)
    
    # Perform ANOVA to test for significant differences among depth ranges
    try:
        # List to hold data for each depth range
        depth_data = []
        depth_names = []
        
        for depth_range in depth_density['Depth_Range']:
            group_data = analysis_df[analysis_df['Depth_Range'] == depth_range]['Total_Octocorals']
            if len(group_data) > 0:
                depth_data.append(group_data)
                depth_names.append(depth_range)
        
        if len(depth_data) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_value = stats.f_oneway(*depth_data)
            
            # Add statistical test results to the plot
            fig.text(0.5, 0.96, 
                    f'ANOVA Results: F={f_stat:.2f}, p={p_value:.4f} ' + 
                    ('(Significant)' if p_value < 0.05 else '(Not Significant)'),
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor=COLORS['dark_blue']))
    except Exception as e:
        print(f"Error performing statistical test: {str(e)}")
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_density_by_depth.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Create a box plot of octocoral density by depth range
    plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['background'])
    
    # Create the box plot with enhanced styling
    box_plot = sns.boxplot(x='Depth_Range', y='Total_Octocorals', data=analysis_df,
                         palette='viridis', ax=ax, width=0.6, fliersize=5,
                         hue='Depth_Range', legend=False,
                         boxprops={'alpha': 0.8})
    
    # Add individual data points
    sns.stripplot(x='Depth_Range', y='Total_Octocorals', data=analysis_df,
                 size=5, hue='Depth_Range', palette='viridis', alpha=0.4, jitter=True, ax=ax, legend=False)
    
    # Style the box plot
    ax.set_title('Distribution of Octocoral Density by Depth Range', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Depth Range', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add a note about the data source
    plt.figtext(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_density_depth_boxplot.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Depth-based analysis saved.")
    
    return depth_density  # Return the summary data for potential further analysis

# Function to analyze seasonal patterns in octocoral density
def analyze_seasonal_patterns(df):
    """
    Analyze seasonal patterns in octocoral density.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
    """
    print("Analyzing seasonal patterns...")
    
    # Ensure we have a Date column in datetime format
    if 'Date' not in df.columns:
        print("Warning: Cannot perform seasonal analysis - Date column not found")
        return
    
    # Extract month and season
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    
    # Define seasons based on months
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df['Season'] = df['Month'].map(season_map)
    
    # Group by season and calculate statistics
    seasonal_stats = df.groupby('Season')['Total_Octocorals'].agg(['mean', 'median', 'std', 'count']).reset_index()
    seasonal_stats['se'] = seasonal_stats['std'] / np.sqrt(seasonal_stats['count'])
    seasonal_stats['ci_95'] = 1.96 * seasonal_stats['se']
    
    # Sort in natural season order
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_stats['Season'] = pd.Categorical(seasonal_stats['Season'], categories=season_order, ordered=True)
    seasonal_stats = seasonal_stats.sort_values('Season')
    
    # Group by month and calculate statistics
    monthly_stats = df.groupby(['Month', 'Month_Name'])['Total_Octocorals'].agg(['mean', 'std', 'count']).reset_index()
    monthly_stats['se'] = monthly_stats['std'] / np.sqrt(monthly_stats['count'])
    monthly_stats['ci_95'] = 1.96 * monthly_stats['se']
    monthly_stats = monthly_stats.sort_values('Month')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.subplots_adjust(wspace=0.3)
    
    # Plot 1: Seasonal patterns
    season_colors = {
        'Winter': '#A1D6E2',  # Light blue
        'Spring': '#81C784',  # Green
        'Summer': '#FFC107',  # Yellow
        'Fall': '#E57373'     # Red
    }
    
    # Create a bar chart for seasonal patterns
    for i, season in enumerate(seasonal_stats['Season']):
        ax1.bar(i, seasonal_stats.loc[seasonal_stats['Season'] == season, 'mean'].values[0],
               yerr=seasonal_stats.loc[seasonal_stats['Season'] == season, 'ci_95'].values[0],
               color=season_colors[season], alpha=0.8, width=0.7, capsize=10,
               label=season)
    
    # Add data labels
    for i, row in enumerate(seasonal_stats.itertuples()):
        ax1.text(i, row.mean + row.ci_95 + 0.3, f'{row.mean:.2f}\nn={row.count}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Style the first plot
    ax1.set_title('Octocoral Density by Season', 
                 fontweight='bold', fontsize=18, color=COLORS['dark_blue'])
    ax1.set_xlabel('Season', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Mean Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(seasonal_stats)))
    ax1.set_xticklabels(seasonal_stats['Season'])
    ax1.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Plot 2: Monthly patterns
    monthly_colors = plt.cm.viridis(np.linspace(0, 1, 12))
    
    # Create a bar chart for monthly patterns
    bars = ax2.bar(range(len(monthly_stats)), monthly_stats['mean'], 
                  yerr=monthly_stats['ci_95'], capsize=7,
                  color=monthly_colors, alpha=0.8, width=0.7)
    
    # Add data labels (only show count to avoid clutter)
    for i, (bar, count) in enumerate(zip(bars, monthly_stats['count'])):
        height = bar.get_height()
        if count > 10:  # Only show labels for months with sufficient data
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f'n={count}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    # Style the second plot
    ax2.set_title('Octocoral Density by Month', 
                 fontweight='bold', fontsize=18, color=COLORS['dark_blue'])
    ax2.set_xlabel('Month', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Mean Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(monthly_stats)))
    ax2.set_xticklabels(monthly_stats['Month_Name'])
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Perform statistical test (ANOVA) to check if the differences are significant
    try:
        # Prepare data for ANOVA (by season)
        season_groups = [df[df['Season'] == season]['Total_Octocorals'] for season in season_order if season in df['Season'].unique()]
        
        if len(season_groups) >= 2 and all(len(group) > 0 for group in season_groups):
            f_stat, p_value = stats.f_oneway(*season_groups)
            
            # Add statistical results to the plot
            fig.text(0.5, 0.95, 
                    f'ANOVA Results: F={f_stat:.2f}, p={p_value:.4f} ' + 
                    ('(Significant seasonal differences)' if p_value < 0.05 else '(No significant seasonal differences)'),
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor=COLORS['dark_blue']))
    except Exception as e:
        print(f"Error performing statistical test: {str(e)}")
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_seasonal_patterns.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Seasonal analysis saved.")
    
    return seasonal_stats  # Return the seasonal stats for potential further analysis

# Function to analyze species diversity across stations
def analyze_species_diversity(df, species_cols):
    """
    Analyze the diversity of octocoral species across stations and regions.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names for octocoral species
    """
    print("Analyzing species diversity...")
    
    # Create a copy of the dataframe for analysis
    analysis_df = df.copy()
    
    # Calculate richness (number of species present at each station)
    richness_data = []
    
    for _, row in analysis_df.iterrows():
        # Count species with non-zero values
        species_present = sum(row[species_cols] > 0)
        richness_data.append({
            'StationID': row['StationID'],
            'Site_name': row['Site_name'],
            'Habitat': row['Habitat'],
            'Subregion': row['Subregion'],
            'Year': row['Year'],
            'Richness': species_present,
            'Total_Density': row['Total_Octocorals']
        })
    
    # Create a DataFrame from the richness data
    richness_df = pd.DataFrame(richness_data)
    
    # Calculate Shannon diversity index (H) and Simpson index (D) for each row
    for idx, row in analysis_df.iterrows():
        # Get species data
        species_data = row[species_cols].astype(float).values
        
        # Skip calculation if no species present
        if np.sum(species_data) == 0:
            shannon = 0
            simpson = 0
        else:
            # Calculate proportions
            proportions = species_data / np.sum(species_data)
            
            # Remove zeros to avoid log(0)
            proportions = proportions[proportions > 0]
            
            # Calculate Shannon index
            shannon = 0
            for p in proportions:
                if p > 0:
                    shannon -= p * np.log(p)
            
            # Calculate Simpson index
            simpson = 1 - np.sum(proportions ** 2)
        
        # Store in richness_df
        richness_df.loc[idx, 'Shannon_Index'] = shannon
        richness_df.loc[idx, 'Simpson_Index'] = simpson
    
    # Calculate evenness (Shannon index / log of richness)
    richness_df['Evenness'] = 0.0  # Initialize with 0.0 as float
    
    # Avoid division by zero or log(1) = 0
    for idx, row in richness_df.iterrows():
        if row['Richness'] > 1:
            richness_df.loc[idx, 'Evenness'] = float(row['Shannon_Index'] / np.log(row['Richness']))
    
    # Group by region and habitat to get average diversity metrics
    region_diversity = richness_df.groupby('Subregion').agg({
        'Richness': 'mean',
        'Shannon_Index': 'mean',
        'Simpson_Index': 'mean',
        'Evenness': 'mean'
    }).reset_index()
    
    habitat_diversity = richness_df.groupby('Habitat').agg({
        'Richness': 'mean',
        'Shannon_Index': 'mean',
        'Simpson_Index': 'mean',
        'Evenness': 'mean'
    }).reset_index()
    
    # Create visualizations for diversity metrics
    # 1. Plot richness by region and habitat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.subplots_adjust(wspace=0.3)
    
    # Region colors for consistency
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
    
    # Habitat colors for consistency
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
    
    # Plot richness by region
    bar_colors = [region_colors.get(region, 'gray') for region in region_diversity['Subregion']]
    ax1.bar(region_diversity['Subregion'], region_diversity['Richness'], color=bar_colors, alpha=0.8)
    
    # Add region names instead of codes
    ax1.set_xticks(range(len(region_diversity)))
    ax1.set_xticklabels([region_names.get(region, region) for region in region_diversity['Subregion']])
    
    ax1.set_title('Mean Octocoral Species Richness by Region', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax1.set_xlabel('Region', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add data labels
    for i, v in enumerate(region_diversity['Richness']):
        ax1.text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Plot richness by habitat
    bar_colors = [habitat_colors.get(habitat, 'gray') for habitat in habitat_diversity['Habitat']]
    ax2.bar(habitat_diversity['Habitat'], habitat_diversity['Richness'], color=bar_colors, alpha=0.8)
    
    # Add habitat names instead of codes
    ax2.set_xticks(range(len(habitat_diversity)))
    ax2.set_xticklabels([habitat_names.get(habitat, habitat) for habitat in habitat_diversity['Habitat']])
    
    ax2.set_title('Mean Octocoral Species Richness by Habitat', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax2.set_xlabel('Habitat', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add data labels
    for i, v in enumerate(habitat_diversity['Richness']):
        ax2.text(i, v + 0.2, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_species_richness.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # 2. Create a multi-metric diversity plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=COLORS['background'])
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Region-based diversity metrics
    metrics = ['Richness', 'Shannon_Index', 'Simpson_Index', 'Evenness']
    metric_titles = ['Species Richness', 'Shannon Diversity Index', 'Simpson Diversity Index', 'Species Evenness']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot the metric by region
        for j, region in enumerate(region_diversity['Subregion']):
            ax.bar(j, region_diversity.loc[region_diversity['Subregion'] == region, metric].values[0],
                  color=region_colors.get(region, 'gray'), alpha=0.8)
        
        # Set labels and styling
        ax.set_title(f'{title} by Region', fontweight='bold', fontsize=14, color=COLORS['dark_blue'])
        ax.set_xlabel('Region', fontweight='bold', fontsize=12)
        ax.set_ylabel(title, fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(region_diversity)))
        ax.set_xticklabels([region_names.get(region, region) for region in region_diversity['Subregion']])
        ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add data labels
        for j, v in enumerate(region_diversity[metric]):
            ax.text(j, v + (ax.get_ylim()[1] * 0.03), f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)
    
    # Add a overall title for the figure
    fig.suptitle('Octocoral Diversity Metrics Analysis', 
                fontsize=20, fontweight='bold', color=COLORS['dark_blue'],
                y=0.98, va='top',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_diversity_metrics.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # 3. Create a temporal trend of species richness
    # Group by year and calculate mean richness
    yearly_richness = richness_df.groupby('Year').agg({
        'Richness': ['mean', 'std', 'count'],
        'Shannon_Index': ['mean', 'std'],
        'Simpson_Index': ['mean', 'std'],
        'Evenness': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-level columns
    yearly_richness.columns = ['_'.join(col).strip('_') for col in yearly_richness.columns.values]
    
    # Calculate confidence intervals
    yearly_richness['Richness_ci'] = 1.96 * yearly_richness['Richness_std'] / np.sqrt(yearly_richness['Richness_count'])
    
    # Create a temporal trend plot
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the mean richness line
    ax.plot(yearly_richness['Year'], yearly_richness['Richness_mean'], 
            marker='o', color=COLORS['coral'], linewidth=2.5, markersize=8,
            label='Mean Species Richness')
    
    # Add confidence interval
    ax.fill_between(yearly_richness['Year'],
                   yearly_richness['Richness_mean'] - yearly_richness['Richness_ci'],
                   yearly_richness['Richness_mean'] + yearly_richness['Richness_ci'],
                   color=COLORS['coral'], alpha=0.2)
    
    # Add trend line
    X = yearly_richness['Year'].values.reshape(-1, 1)
    y = yearly_richness['Richness_mean'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate slope and plot trend line
    slope = model.coef_[0]
    ax.plot(yearly_richness['Year'], y_pred, '--', color=COLORS['dark_blue'], 
            linewidth=2, label=f'Linear Trend (Slope: {slope:.2f} species/year)')
    
    # Style the plot
    ax.set_title('Temporal Trend in Octocoral Species Richness (2011-2023)', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Set axis limits
    ax.set_xlim(yearly_richness['Year'].min() - 0.5, yearly_richness['Year'].max() + 0.5)
    y_min = max(0, yearly_richness['Richness_mean'].min() - yearly_richness['Richness_ci'].max() - 1)
    ax.set_ylim(y_min, yearly_richness['Richness_mean'].max() + yearly_richness['Richness_ci'].max() + 1)
    
    # Enhance the legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                      fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
    legend.get_frame().set_linewidth(1.5)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_richness_temporal_trend.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species diversity analysis saved.")
    
    return richness_df  # Return the richness DataFrame for further analysis

# Function to perform comparative analysis between octocorals and environmental factors
def analyze_environmental_correlations(df, species_cols, stations_df):
    """
    Analyze correlations between octocoral density and environmental factors.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names for octocoral species
        stations_df (DataFrame): DataFrame with station metadata including environmental data
    """
    print("Analyzing environmental correlations...")
    
    # Check if we have environmental data
    environmental_cols = ['Depth_ft', 'Temperature_C', 'Salinity_ppt', 'pH', 'Turbidity_NTU']
    available_env_cols = [col for col in environmental_cols if col in stations_df.columns]
    
    if not available_env_cols:
        print("Warning: No environmental data columns found for correlation analysis")
        return
    
    # Merge octocoral data with environmental data
    merged_df = df.copy()
    
    # Convert StationID to same type before merging
    if 'StationID' in stations_df.columns:
        merged_df['StationID'] = merged_df['StationID'].astype(str)
        stations_df['StationID'] = stations_df['StationID'].astype(str)
        
        # Get environmental variables from stations data
        env_data = stations_df[['StationID'] + available_env_cols].drop_duplicates()
        
        # Merge with octocoral data
        merged_df = pd.merge(merged_df, env_data, on='StationID', how='left')
    else:
        print("Warning: Cannot merge environmental data - StationID not found in stations data")
        return
    
    # Calculate correlation between octocoral density and environmental factors
    correlation_data = []
    
    # Overall correlation with total octocoral density
    for env_col in available_env_cols:
        valid_data = merged_df.dropna(subset=[env_col, 'Total_Octocorals'])
        if len(valid_data) > 10:  # Ensure we have enough data points
            corr, p_value = stats.pearsonr(valid_data[env_col], valid_data['Total_Octocorals'])
            correlation_data.append({
                'Environmental_Factor': env_col,
                'Target': 'Total Octocoral Density',
                'Correlation': corr,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Correlations for the top 5 most abundant species
    top_species = df[species_cols].mean().sort_values(ascending=False).head(5).index.tolist()
    
    for species in top_species:
        for env_col in available_env_cols:
            valid_data = merged_df.dropna(subset=[env_col, species])
            if len(valid_data) > 10:  # Ensure we have enough data points
                corr, p_value = stats.pearsonr(valid_data[env_col], valid_data[species])
                correlation_data.append({
                    'Environmental_Factor': env_col,
                    'Target': species.replace('_', ' '),
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_data)
    
    if len(corr_df) == 0:
        print("Warning: Not enough data for correlation analysis")
        return
    
    # Create visualizations
    # 1. Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Pivot the data for the heatmap
    heatmap_data = corr_df.pivot(index='Target', columns='Environmental_Factor', values='Correlation')
    
    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
               linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'},
               fmt='.2f', ax=ax)
    
    # Style the heatmap
    ax.set_title('Correlation Between Octocoral Density and Environmental Factors', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    ax.set_xlabel('Environmental Factor', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel('Octocoral Species / Total Density', fontweight='bold', fontsize=14, labelpad=10)
    
    # Add asterisks for significant correlations
    for i, target in enumerate(heatmap_data.index):
        for j, factor in enumerate(heatmap_data.columns):
            is_significant = corr_df[(corr_df['Target'] == target) & 
                                     (corr_df['Environmental_Factor'] == factor)]['Significant'].values
            
            if len(is_significant) > 0 and is_significant[0]:
                ax.text(j + 0.5, i + 0.85, '*', fontsize=20, ha='center', va='center',
                       color='black', path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note about significance
    ax.text(0.5, -0.1, '* indicates statistically significant correlation (p < 0.05)',
           ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_environmental_correlations.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # 2. Scatter plots for significant correlations
    significant_correlations = corr_df[corr_df['Significant']].sort_values(by='Correlation', ascending=False)
    
    if len(significant_correlations) > 0:
        # Limit to top 4 strongest correlations (positive or negative)
        if len(significant_correlations) > 4:
            # Get top 2 positive and top 2 negative correlations
            positive_corr = significant_correlations[significant_correlations['Correlation'] > 0].head(2)
            negative_corr = significant_correlations[significant_correlations['Correlation'] < 0].head(2)
            top_correlations = pd.concat([positive_corr, negative_corr])
        else:
            top_correlations = significant_correlations
        
        # Create scatter plots for these correlations
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLORS['background'])
        axs = axs.flatten()
        
        for i, (_, row) in enumerate(top_correlations.iterrows()):
            if i >= 4:  # Limit to 4 plots
                break
            
            ax = axs[i]
            ax.set_facecolor(COLORS['background'])
            
            env_factor = row['Environmental_Factor']
            target = row['Target']
            corr = row['Correlation']
            p_value = row['P_Value']
            
            # Get the data for this correlation
            if target == 'Total Octocoral Density':
                y_col = 'Total_Octocorals'
            else:
                # Convert back to column name format
                y_col = target.replace(' ', '_')
            
            # Create the scatter plot with regression line
            valid_data = merged_df.dropna(subset=[env_factor, y_col])
            sns.regplot(x=env_factor, y=y_col, data=valid_data, 
                       scatter_kws={'alpha': 0.6, 's': 50, 'color': COLORS['dark_blue']},
                       line_kws={'color': COLORS['coral'], 'linewidth': 2},
                       ax=ax)
            
            # Style the plot
            ax.set_title(f'{target} vs {env_factor}\nr = {corr:.2f}, p = {p_value:.4f}', 
                        fontweight='bold', fontsize=14, color=COLORS['dark_blue'])
            
            ax.set_xlabel(env_factor, fontweight='bold', fontsize=12)
            ax.set_ylabel(target, fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add density coloring for better visualization
            if i == 0:  # Add a legend for the first plot only
                legend_elements = [
                    Patch(facecolor=COLORS['dark_blue'], alpha=0.6, label='Data Points'),
                    Patch(facecolor=COLORS['coral'], alpha=0.8, label='Regression Line')
                ]
                ax.legend(handles=legend_elements, loc='best')
        
        # Adjust the layout if we have fewer than 4 plots
        for i in range(len(top_correlations), 4):
            axs[i].set_visible(False)
        
        fig.suptitle('Significant Correlations Between Octocoral Density and Environmental Factors',
                    fontsize=18, fontweight='bold', color=COLORS['dark_blue'],
                    y=0.98, va='top',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "octocoral_environmental_scatterplots.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    
    print("Environmental correlation analysis saved.")
    
    return corr_df  # Return the correlation data for potential further analysis

# Function to perform multivariate analysis of octocoral community
def perform_multivariate_analysis(df, species_cols):
    """
    Perform multivariate analysis to identify patterns in octocoral community structure.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names for octocoral species
    """
    print("Performing multivariate analysis...")
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from scipy.cluster.hierarchy import dendrogram, linkage
    except ImportError:
        print("Warning: Required libraries for multivariate analysis not available")
        return
    
    # Create a copy of the dataframe with just the species data
    species_data = df[species_cols].copy()
    
    # Remove any rows with all zeros (no octocorals)
    species_data = species_data.loc[~(species_data == 0).all(axis=1)]
    
    # Fill NaN values with 0
    species_data.fillna(0, inplace=True)
    
    # Check if we have enough data
    if len(species_data) < 10:
        print("Warning: Not enough data for multivariate analysis")
        return
    
    # Get metadata for plotting
    metadata = df.loc[species_data.index, ['Habitat', 'Subregion', 'Year']].copy()
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(species_data)
    
    # Create a DataFrame for the scaled data
    scaled_df = pd.DataFrame(X_scaled, index=species_data.index, columns=species_data.columns)
    
    # Perform Principal Component Analysis (PCA)
    n_components = min(5, len(species_data.columns), len(species_data) - 1)  # Limit to 5 or less
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df = pd.concat([pca_df, metadata], axis=1)
    
    # Calculate the explained variance for each component
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Plot the PCA results
    # 1. Scatter plot of the first two principal components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
    fig.subplots_adjust(wspace=0.3)
    
    # Define region and habitat colors for consistency
    region_colors = {
        'UK': COLORS['dark_blue'],   # Upper Keys
        'MK': COLORS['ocean_blue'],  # Middle Keys
        'LK': COLORS['light_blue']   # Lower Keys
    }
    
    habitat_colors = {
        'OS': COLORS['coral'],      # Offshore Shallow
        'OD': COLORS['sand'],       # Offshore Deep
        'P': COLORS['reef_green'],  # Patch Reef
        'HB': COLORS['ocean_blue'], # Hardbottom
        'BCP': COLORS['dark_blue']  # Backcountry Patch
    }
    
    # Plot PCA by region
    for region in pca_df['Subregion'].unique():
        subset = pca_df[pca_df['Subregion'] == region]
        ax1.scatter(subset['PC1'], subset['PC2'], 
                   c=region_colors.get(region, 'gray'), 
                   alpha=0.7, s=50, label=region,
                   edgecolor='white', linewidth=0.5)
    
    ax1.set_title('PCA of Octocoral Community by Region', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1f}% variance)', fontweight='bold', fontsize=12)
    ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1f}% variance)', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax1.legend(title='Region', frameon=True)
    
    # Plot PCA by habitat
    for habitat in pca_df['Habitat'].unique():
        subset = pca_df[pca_df['Habitat'] == habitat]
        ax2.scatter(subset['PC1'], subset['PC2'], 
                   c=habitat_colors.get(habitat, 'gray'), 
                   alpha=0.7, s=50, label=habitat,
                   edgecolor='white', linewidth=0.5)
    
    ax2.set_title('PCA of Octocoral Community by Habitat', 
                 fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax2.set_xlabel(f'PC1 ({explained_variance[0]:.1f}% variance)', fontweight='bold', fontsize=12)
    ax2.set_ylabel(f'PC2 ({explained_variance[1]:.1f}% variance)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax2.legend(title='Habitat', frameon=True)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.savefig(os.path.join(results_dir, "octocoral_pca_analysis.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # 2. Plot the explained variance for each component
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    bars = ax.bar(range(1, n_components + 1), explained_variance, 
            color=plt.cm.viridis(np.linspace(0, 0.8, n_components)), alpha=0.8)
    
    # Add cumulative variance line
    cumulative_variance = np.cumsum(explained_variance)
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cumulative_variance, 'o-', color=COLORS['coral'], 
            linewidth=2.5, markersize=8)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 105)
    
    # Add data labels
    for i, (bar, pct) in enumerate(zip(bars, explained_variance)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    for i, cum_pct in enumerate(cumulative_variance):
        ax2.text(i + 1, cum_pct + 2, f'{cum_pct:.1f}%', ha='center', va='bottom', 
                color=COLORS['coral'], fontweight='bold')
    
    ax.set_title('Explained Variance by Principal Component', 
                fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax.set_xlabel('Principal Component', fontweight='bold', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontweight='bold', fontsize=12)
    ax.set_xticks(range(1, n_components + 1))
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    plt.savefig(os.path.join(results_dir, "octocoral_pca_variance.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # 3. Plot the loadings (contribution of each species to the first two PCs)
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=species_data.columns
    )
    
    # Get sorted indices based on absolute contribution to PC1
    pc1_sorted_idx = loadings['PC1'].abs().sort_values(ascending=False).index
    
    # Extract top contributing species for PC1 and PC2
    top_pc1_species = list(pc1_sorted_idx[:min(15, len(pc1_sorted_idx))])
    top_pc2_species = list(loadings['PC2'].abs().sort_values(ascending=False).index[:min(15, len(loadings))])
    
    # Combine unique species from both lists
    top_loadings = list(set(top_pc1_species + top_pc2_species))
    
    # Create a new DataFrame with just these species
    top_loadings_df = loadings.loc[top_loadings, ['PC1', 'PC2']]
    
    # Create a loading plot
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot loadings as vectors
    for i, species in enumerate(top_loadings_df.index):
        x = top_loadings_df.loc[species, 'PC1']
        y = top_loadings_df.loc[species, 'PC2']
        
        # Draw vector
        ax.arrow(0, 0, x, y, head_width=0.02, head_length=0.05, 
                fc=plt.cm.viridis(i/len(top_loadings_df)), ec=plt.cm.viridis(i/len(top_loadings_df)),
                alpha=0.7, width=0.005)
        
        # Label species (adjust text position for better visibility)
        distance = np.sqrt(x**2 + y**2)
        label_x = x * 1.1
        label_y = y * 1.1
        
        # Make species name more readable
        species_label = species.replace('_', ' ')
        
        ax.text(label_x, label_y, species_label, 
               fontsize=9, ha='center', va='center', 
               fontweight='bold', color=plt.cm.viridis(i/len(top_loadings_df)),
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Add circle for reference
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Style the plot
    ax.set_title('Species Contributions to Principal Components', 
                fontweight='bold', fontsize=16, color=COLORS['dark_blue'])
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1f}% variance)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1f}% variance)', fontweight='bold', fontsize=12)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add zero lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.savefig(os.path.join(results_dir, "octocoral_pca_loadings.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Multivariate analysis saved.")
    
    return pca_df  # Return the PCA results for potential further analysis

# Function to analyze species similarity and co-occurrence patterns
def analyze_species_similarity(df, species_cols):
    """
    Analyze the similarity between octocoral species and their co-occurrence patterns.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with octocoral density data
        species_cols (list): List of column names for octocoral species
    """
    print("Analyzing species similarity patterns...")
    
    # Create a copy of the dataframe with just the species data
    species_data = df[species_cols].copy()
    
    # Remove any rows with all zeros (no octocorals)
    species_data = species_data.loc[~(species_data == 0).all(axis=1)]
    
    # Fill NaN values with 0
    species_data.fillna(0, inplace=True)
    
    # Check if we have enough data
    if len(species_data) < 10 or len(species_cols) < 5:
        print("Warning: Not enough data for species similarity analysis")
        return
    
    # Calculate species correlation matrix
    species_corr = species_data.corr(method='pearson')
    
    # Filter to only include species that have some correlation data
    valid_species = species_corr.columns[~species_corr.isna().all()]
    species_corr = species_corr.loc[valid_species, valid_species]
    
    # Create a heatmap of species correlations
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=COLORS['background'])
    
    # Generate more readable species names for plotting
    plot_labels = [species.replace('_', ' ') for species in species_corr.columns]
    
    # Calculate a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(species_corr, dtype=bool))
    
    # Create the heatmap
    sns.heatmap(species_corr, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0,
               linewidths=0.5, cbar_kws={'label': 'Pearson Correlation Coefficient'},
               xticklabels=plot_labels, yticklabels=plot_labels,
               ax=ax)
    
    # Style the heatmap
    ax.set_title('Octocoral Species Co-occurrence Patterns', 
                fontweight='bold', fontsize=18, pad=20,
                color=COLORS['dark_blue'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Add a note about the data source
    fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
             ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "octocoral_species_correlations.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Perform hierarchical clustering to identify groups of similar species
    # Get the most common species (to reduce the dimensionality)
    common_species = species_data.columns[species_data.sum() > species_data.sum().quantile(0.75)].tolist()
    
    if len(common_species) < 5:
        # If we don't have enough species with the 75th percentile threshold, take the top 15
        common_species = species_data.sum().sort_values(ascending=False).head(15).index.tolist()
    
    # Filter the data to include only common species
    common_species_data = species_data[common_species]
    species_corr_common = common_species_data.corr(method='pearson')
    
    # Convert the correlation matrix to a distance matrix (1 - abs(correlation))
    # This ensures that both positive and negative correlations are considered as similarity
    distance_matrix = 1 - np.abs(species_corr_common)
    
    # Perform hierarchical clustering
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert distance matrix to condensed form
        condensed_dist = squareform(distance_matrix)
        
        # Generate linkage matrix
        Z = linkage(condensed_dist, method='average')
        
        # Plot dendrogram
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['background'])
        
        # Create more readable labels
        dendrogram_labels = [species.replace('_', ' ') for species in species_corr_common.columns]
        
        # Plot dendrogram with enhanced styling
        dendrogram(
            Z,
            labels=dendrogram_labels,
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax,
            color_threshold=0.5 * max(Z[:, 2]),  # Color threshold at 50% of max distance
            above_threshold_color='gray'
        )
        
        # Style the dendrogram
        ax.set_title('Hierarchical Clustering of Octocoral Species', 
                    fontweight='bold', fontsize=18, pad=20,
                    color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        ax.set_xlabel('Species', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Distance (1 - |correlation|)', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "octocoral_species_dendrogram.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    except Exception as e:
        print(f"Error creating dendrogram: {str(e)}")
    
    # Network visualization of species co-occurrence
    try:
        # Threshold the correlation matrix to focus on strong relationships
        threshold = 0.3  # Adjust threshold as needed
        species_network = species_corr_common.copy()
        species_network[np.abs(species_network) < threshold] = 0
        
        # Create a network graph
        import networkx as nx
        
        # Create a graph from the thresholded correlation matrix
        G = nx.from_pandas_adjacency(species_network)
        
        # Set edge weights based on correlation strength
        for u, v, d in G.edges(data=True):
            d['weight'] = np.abs(species_network.loc[u, v])
            d['color'] = 'green' if species_network.loc[u, v] > 0 else 'red'
        
        # Create the network visualization
        fig, ax = plt.subplots(figsize=(14, 14), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['background'])
        
        # Use a force-directed layout for the network
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw nodes with a size proportional to their total correlation strength
        node_sizes = [500 * (np.abs(species_network[node]).sum() / len(species_network)) for node in G.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
        
        # Draw edges with width proportional to correlation strength
        # Positive correlations in green, negative in red
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_widths = [2 * G[u][v]['weight'] for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, ax=ax)
        
        # Create more readable labels
        labels = {node: node.replace('_', ' ') for node in G.nodes()}
        
        # Add node labels with white background for readability
        for node, (x, y) in pos.items():
            ax.text(x, y, labels[node],
                   fontsize=10, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                   fontweight='bold')
        
        # Style the network plot
        ax.set_title('Octocoral Species Co-occurrence Network', 
                    fontweight='bold', fontsize=18, pad=20,
                    color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Add a legend explaining the network elements
        legend_elements = [
            Patch(facecolor='skyblue', alpha=0.8, label='Species'),
            Patch(facecolor='green', alpha=0.6, label='Positive Correlation'),
            Patch(facecolor='red', alpha=0.6, label='Negative Correlation')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                 title='Network Elements', title_fontsize=14)
        
        # Add a note about the correlation threshold
        ax.text(0.5, -0.05, f'Only correlations with |r| > {threshold} are shown',
               ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        # Remove axis
        ax.axis('off')
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        plt.tight_layout()

        plt.close()
    except Exception as e:
        print(f"Error creating network visualization: {str(e)}")
    
    print("Species similarity analysis saved.")
    
    return species_corr  # Return the correlation matrix for potential further analysis

# Main execution
if __name__ == "__main__":
    try:
        # Load and preprocess data
        octo_df, stations_df, species_cols = load_and_preprocess_data()
        
        # Basic analyses
        # Plot overall trend
        plot_overall_trend(octo_df)
        
        # Plot trends by region
        plot_trends_by_region(octo_df)
        
        # Plot trends by habitat
        plot_trends_by_habitat(octo_df)
        
        # Create temporal heatmap
        create_temporal_heatmap(octo_df)
        
        # Plot species composition
        plot_species_composition(octo_df, species_cols)
        
        # Analyze rate of change
        analyze_rate_of_change(octo_df)
        
        # Plot spatial distribution if coordinates are available
        try:
            plot_spatial_distribution(octo_df, stations_df)
        except Exception as e:
            print(f"Could not create spatial distribution plot: {str(e)}")
        
        # Advanced analyses
        # Analyze depth patterns
        analyze_depth_patterns(octo_df, stations_df)
        
        # Analyze seasonal patterns
        analyze_seasonal_patterns(octo_df)
        
        # Analyze species diversity
        analyze_species_diversity(octo_df, species_cols)
        
        # Analyze environmental correlations
        analyze_environmental_correlations(octo_df, species_cols, stations_df)
        
        # Perform multivariate analysis
        perform_multivariate_analysis(octo_df, species_cols)
        
        # Analyze species similarity
        analyze_species_similarity(octo_df, species_cols)
        
        print("\nAll octocoral density analysis plots have been generated successfully!")
        print(f"Results saved in the '{results_dir}' directory.")
        
    except Exception as e:
        print(f"Error in octocoral density analysis: {str(e)}")
        import traceback
        traceback.print_exc()