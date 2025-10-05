"""
05_coral_species_spatial_patterns.py - Analysis of Spatial Patterns in Coral Species Distribution

This script analyzes the spatial distribution patterns of different coral species across the 
Florida Keys reef system and how these patterns have changed over time (1996-2023).

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
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
import matplotlib.patheffects as pe  # For enhanced visual effects
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Create results directory if it doesn't exist
results_dir = "05_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

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

# Species of interest for detailed analysis
# Selected based on ecological significance and abundance
FOCUS_SPECIES = [
    'Orbicella_annularis_complex',  # Boulder star coral - framework builder
    'Siderastrea_siderea',          # Massive starlet coral - stress-tolerant
    'Porites_astreoides',           # Mustard hill coral - opportunistic
    'Montastraea_cavernosa',        # Great star coral - common massive coral
    'Acropora_cervicornis',         # Staghorn coral - endangered, fast-growing
    'Acropora_palmata',             # Elkhorn coral - endangered, framework builder
    'Pseudodiploria_strigosa',      # Symmetrical brain coral - resilient
    'Colpophyllia_natans',          # Boulder brain coral - large massive coral
    'Undaria_agaricites_complex',   # Lettuce coral - common on reefs
    'Millepora_alcicornis'          # Fire coral - not a true coral but common
]

# Species groups for ecological analysis
SPECIES_GROUPS = {
    'Framework builders': ['Orbicella_annularis_complex', 'Montastraea_cavernosa', 
                          'Colpophyllia_natans', 'Pseudodiploria_strigosa', 
                          'Pseudodiploria_clivosa', 'Diploria_labyrinthiformis'],
    'Endangered species': ['Acropora_cervicornis', 'Acropora_palmata', 'Dendrogyra_cylindrus'],
    'Opportunistic species': ['Porites_astreoides', 'Porites_porites_complex', 
                             'Siderastrea_siderea', 'Siderastrea_radians', 
                             'Undaria_agaricites_complex'],
    'Fire corals': ['Millepora_alcicornis', 'Millepora_complanata']
}

# Region and habitat mapping for better display
REGION_NAMES = {
    'UK': 'Upper Keys',
    'MK': 'Middle Keys',
    'LK': 'Lower Keys'
}

HABITAT_NAMES = {
    'OS': 'Offshore Shallow',
    'OD': 'Offshore Deep',
    'P': 'Patch Reef',
    'HB': 'Hardbottom',
    'BCP': 'Backcountry Patch'
}

# Function to load and preprocess data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP stony coral species and station datasets.
    
    Returns:
        tuple: (species_df, stations_df) - Preprocessed DataFrames
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the Stony Coral Species dataset
        species_df = pd.read_csv("CREMP_CSV_files/CREMP_Pcover_2023_StonyCoralSpecies.csv")
        print(f"Stony coral species data loaded successfully with {len(species_df)} rows")
        
        # Load the Stations dataset (contains station metadata)
        stations_df = pd.read_csv("CREMP_CSV_files/CREMP_Stations_2023.csv")
        print(f"Stations data loaded successfully with {len(stations_df)} rows")
        
        # Convert date column to datetime format
        if 'Date' in species_df.columns:
            species_df['Date'] = pd.to_datetime(species_df['Date'])
        
        # Convert proportion to percentage (if needed)
        # Check the first few rows to determine if conversion is needed
        if species_df.iloc[0, 10:].max() <= 1.0:  # Assuming columns from 10 onwards are species data
            print("Converting proportions to percentages...")
            # Convert all species columns (from column 10 onwards) to percentages
            for col in species_df.columns[10:]:
                if col in species_df.columns and pd.api.types.is_numeric_dtype(species_df[col]):
                    species_df[col] = species_df[col] * 100
        
        # Add a total stony coral cover column for reference
        species_cols = [col for col in species_df.columns if col not in 
                        ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                        'Site_name', 'StationID', 'Surveyed_all_years', 'points']]
        
        species_df['Total_Stony_Coral'] = species_df[species_cols].sum(axis=1)
        
        # Merge with station data to get coordinates
        # First, ensure StationID is the same type in both DataFrames
        species_df['StationID'] = species_df['StationID'].astype(str)
        stations_df['StationID'] = stations_df['StationID'].astype(str)
        
        # Merge on StationID
        merged_df = pd.merge(species_df, 
                            stations_df[['StationID', 'latDD', 'lonDD', 'Depth_ft']], 
                            on='StationID', how='left')
        
        print(f"Merged data shape: {merged_df.shape}")
        
        return merged_df, stations_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise 

def create_species_richness_map(df, stations_df):
    """
    Create a map visualization showing the spatial distribution of coral species richness
    across monitoring sites.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
        stations_df (DataFrame): DataFrame with station metadata
    """
    print("Creating species richness map...")
    
    # Calculate species richness (number of species present) at each site
    # First, identify columns containing species data
    species_cols = [col for col in df.columns if col not in 
                   ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                    'Site_name', 'StationID', 'Surveyed_all_years', 'points', 
                    'latDD', 'lonDD', 'Depth_ft', 'Total_Stony_Coral']]
    
    # Create a function to count species present at each station (>0% cover)
    def count_species(row):
        return sum(row[col] > 0 for col in species_cols)
    
    # Get recent data (last 3 years) for current status
    recent_years = df['Year'].max() - 2
    recent_data = df[df['Year'] >= recent_years].copy()  # Create explicit copy
    
    # Calculate species richness for each station in recent data
    recent_data.loc[:, 'Species_Richness'] = recent_data.apply(count_species, axis=1)  # Use .loc to avoid SettingWithCopyWarning
    
    # Aggregate to site level (average across stations and years)
    site_richness = recent_data.groupby(['Site_name', 'Habitat', 'Subregion', 'latDD', 'lonDD'])['Species_Richness'].mean().reset_index()
    
    # Create the map figure
    fig, ax = plt.subplots(figsize=(18, 12), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define boundaries for the Florida Keys area
    min_lon, max_lon = site_richness['lonDD'].min() - 0.2, site_richness['lonDD'].max() + 0.2
    min_lat, max_lat = site_richness['latDD'].min() - 0.1, site_richness['latDD'].max() + 0.1
    
    # Plot simplified Florida Keys outline
    keys_outline_x = np.linspace(min_lon, max_lon, 100)
    keys_outline_y = np.sin((keys_outline_x - min_lon) / (max_lon - min_lon) * np.pi) * 0.05 + min_lat + 0.1
    ax.plot(keys_outline_x, keys_outline_y, 'k-', linewidth=2, alpha=0.5)
    
    # Fill land area with a light color
    ax.fill_between(keys_outline_x, keys_outline_y, min_lat, color='lightgray', alpha=0.3)
    
    # Color mapping for habitats
    habitat_colors = {
        'OS': COLORS['coral'],
        'OD': COLORS['sand'],
        'P': COLORS['reef_green'],
        'HB': COLORS['ocean_blue'],
        'BCP': COLORS['dark_blue']
    }
    
    # Create a colormap for species richness
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=site_richness['Species_Richness'].min(), 
                    vmax=site_richness['Species_Richness'].max())
    
    # Plot each site as a marker on the map
    for habitat, color in habitat_colors.items():
        habitat_sites = site_richness[site_richness['Habitat'] == habitat]
        
        if not habitat_sites.empty:
            scatter = ax.scatter(
                habitat_sites['lonDD'], 
                habitat_sites['latDD'],
                s=habitat_sites['Species_Richness'] * 15,  # Scale marker size by richness
                c=habitat_sites['Species_Richness'],
                cmap='YlOrRd',
                norm=norm,
                alpha=0.8,
                edgecolor='black',
                linewidth=1,
                marker='^' if habitat in ['OS', 'OD'] else 'o',  # Different markers for different habitat types
                label=habitat
            )
    
    # Add site labels for selected sites (top species richness)
    top_sites = site_richness.nlargest(5, 'Species_Richness')
    for _, site in top_sites.iterrows():
        ax.annotate(
            f"{site['Site_name']}\n({int(site['Species_Richness'])} species)",
            xy=(site['lonDD'], site['latDD']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )
    
    # Add colorbar for species richness
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Coral Species Richness', fontweight='bold', fontsize=14)
    
    # Add legend for habitat types
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    legend_elements = []
    for habitat, name in HABITAT_NAMES.items():
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
    ax.set_title('Spatial Distribution of Coral Species Richness (2021-2023)', 
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
    plt.savefig(os.path.join(results_dir, "coral_species_richness_map.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species richness map saved.")

def analyze_species_habitat_association(df):
    """
    Analyze and visualize the habitat associations of different coral species.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Analyzing species-habitat associations...")
    
    # Get recent data (last 5 years) for current associations
    recent_years = df['Year'].max() - 4
    recent_data = df[df['Year'] >= recent_years]
    
    # Identify focal species present in the data
    species_cols = [col for col in FOCUS_SPECIES if col in df.columns]
    
    # Calculate average abundance by habitat
    habitat_species = recent_data.groupby('Habitat')[species_cols].mean().reset_index()
    
    # Transform data to long format for easier plotting
    habitat_species_long = pd.melt(habitat_species, id_vars=['Habitat'], 
                                  value_vars=species_cols, 
                                  var_name='Species', value_name='Cover_Percent')
    
    # Clean species names for display
    def clean_species_name(name):
        return name.replace('_', ' ').replace('complex', '').strip()
    
    habitat_species_long['Species_Display'] = habitat_species_long['Species'].apply(clean_species_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define habitat color mapping
    habitat_colors = {
        'OS': COLORS['coral'],
        'OD': COLORS['sand'],
        'P': COLORS['reef_green'],
        'HB': COLORS['ocean_blue'],
        'BCP': COLORS['dark_blue']
    }
    
    # Convert habitat codes to full names
    habitat_species_long['Habitat_Display'] = habitat_species_long['Habitat'].map(HABITAT_NAMES)
    
    # Create grouped bar chart
    sns.barplot(x='Species_Display', y='Cover_Percent', hue='Habitat_Display', 
               data=habitat_species_long, palette=habitat_colors.values(), ax=ax)
    
    # Enhance the plot
    ax.set_title('Coral Species Distribution by Habitat Type (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Coral Species', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Cover (%)', fontweight='bold', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    ax.legend(title='Habitat Type', frameon=True, facecolor='white', framealpha=0.9,
             loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Calculate and add statistical significance
    # Perform ANOVA for each species to check if habitat differences are significant
    from scipy import stats
    
    significance_results = []
    
    for species in species_cols:
        species_data = []
        habitat_groups = []
        
        for habitat in recent_data['Habitat'].unique():
            habitat_data = recent_data[recent_data['Habitat'] == habitat][species].dropna()
            if len(habitat_data) > 0:
                species_data.append(habitat_data)
                habitat_groups.append(habitat)
        
        if len(species_data) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_value = stats.f_oneway(*species_data)
            significance_results.append({
                'Species': species,
                'F_statistic': f_stat,
                'P_value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Add more bottom padding for the explanatory notes
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Increase bottom margin to avoid text overlap
    
    # Add explanatory notes
    note_text = (
        "Note: * indicates species with statistically significant (p<0.05)\n"
        "differences in abundance between habitat types (ANOVA test)."
    )
    
    fig.text(0.5, 0.01, note_text, ha='center', va='center', 
             fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "species_habitat_association.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Create a heatmap for species-habitat association
    # Reshape data into a pivot table format
    pivot_data = habitat_species_long.pivot(index='Habitat_Display', 
                                           columns='Species_Display', 
                                           values='Cover_Percent')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Generate the heatmap with improved styling
    sns.heatmap(
        pivot_data, 
        cmap=coral_cmap,
        annot=True,  # Add value annotations
        fmt=".2f",  # Format to 2 decimal places
        linewidths=0.5,
        cbar_kws={'label': 'Average Cover (%)', 'shrink': 0.8},
        ax=ax
    )
    
    # Set plot titles and labels
    ax.set_title('Heatmap of Coral Species Abundance by Habitat Type (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    
    # Improve colorbar appearance
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Average Cover (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "species_habitat_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species-habitat association analysis saved.")

def analyze_species_region_distribution(df):
    """
    Analyze and visualize the regional distribution of coral species.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Analyzing species distribution by region...")
    
    # Get recent data (last 5 years) for current distributions
    recent_years = df['Year'].max() - 4
    recent_data = df[df['Year'] >= recent_years]
    
    # Identify focal species present in the data
    species_cols = [col for col in FOCUS_SPECIES if col in df.columns]
    
    # Calculate average abundance by region
    region_species = recent_data.groupby('Subregion')[species_cols].mean().reset_index()
    
    # Transform data to long format for easier plotting
    region_species_long = pd.melt(region_species, id_vars=['Subregion'], 
                                 value_vars=species_cols, 
                                 var_name='Species', value_name='Cover_Percent')
    
    # Clean species names for display
    def clean_species_name(name):
        return name.replace('_', ' ').replace('complex', '').strip()
    
    region_species_long['Species_Display'] = region_species_long['Species'].apply(clean_species_name)
    region_species_long['Region_Display'] = region_species_long['Subregion'].map(REGION_NAMES)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define region color mapping
    region_colors = {
        'UK': COLORS['dark_blue'],   # Upper Keys
        'MK': COLORS['ocean_blue'],  # Middle Keys
        'LK': COLORS['light_blue']   # Lower Keys
    }
    
    # Create grouped bar chart
    sns.barplot(x='Species_Display', y='Cover_Percent', hue='Region_Display', 
               data=region_species_long, palette=[region_colors[k] for k in sorted(region_colors.keys())], ax=ax)
    
    # Enhance the plot
    ax.set_title('Coral Species Distribution by Region (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Coral Species', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Cover (%)', fontweight='bold', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    ax.legend(title='Region', frameon=True, facecolor='white', framealpha=0.9,
             loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Calculate and add statistical significance
    # Perform ANOVA for each species to check if regional differences are significant
    from scipy import stats
    
    significance_results = []
    
    for species in species_cols:
        species_data = []
        region_groups = []
        
        for region in recent_data['Subregion'].unique():
            region_data = recent_data[recent_data['Subregion'] == region][species].dropna()
            if len(region_data) > 0:
                species_data.append(region_data)
                region_groups.append(region)
        
        if len(species_data) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_value = stats.f_oneway(*species_data)
            significance_results.append({
                'Species': species,
                'F_statistic': f_stat,
                'P_value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Add more bottom padding for explanatory notes
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Increase bottom margin to avoid text overlap
    
    # Add explanatory notes
    note_text = (
        "Note: * indicates species with statistically significant (p<0.05)\n"
        "differences in abundance between regions (ANOVA test)."
    )
    
    fig.text(0.5, 0.01, note_text, ha='center', va='center', 
             fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "species_region_distribution.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Create a supplementary heatmap for species-region association
    # Reshape data into a pivot table format
    pivot_data = region_species_long.pivot(index='Region_Display', 
                                          columns='Species_Display', 
                                          values='Cover_Percent')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Generate the heatmap with improved styling
    sns.heatmap(
        pivot_data, 
        cmap=coral_cmap,
        annot=True,  # Add value annotations
        fmt=".2f",  # Format to 2 decimal places
        linewidths=0.5,
        cbar_kws={'label': 'Average Cover (%)', 'shrink': 0.8},
        ax=ax
    )
    
    # Set plot titles and labels
    ax.set_title('Heatmap of Coral Species Abundance by Region (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    
    # Improve colorbar appearance
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Average Cover (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "species_region_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Species regional distribution analysis saved.")

def analyze_species_temporal_spatial_trends(df):
    """
    Analyze how the spatial distribution of coral species has changed over time.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Analyzing temporal-spatial trends in species distribution...")
    
    # Identify focal species present in the data
    species_cols = [col for col in FOCUS_SPECIES if col in df.columns]
    
    # Define time periods for comparison
    df['Period'] = pd.cut(df['Year'], 
                        bins=[1995, 2000, 2005, 2010, 2015, 2023],
                        labels=['1996-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2023'],
                        right=True)
    
    # Create a figure for each region with panels for different time periods
    for region_code, region_name in REGION_NAMES.items():
        print(f"Analyzing {region_name}...")
        
        # Filter data for the current region
        region_data = df[df['Subregion'] == region_code]
        
        if region_data.empty:
            print(f"No data for {region_name}, skipping...")
            continue
        
        # Create figure with subplots for each time period
        fig, axes = plt.subplots(1, 5, figsize=(22, 8), sharey=True, facecolor=COLORS['background'])
        
        # Define periods to display
        periods = ['1996-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2023']
        
        # For each time period
        for i, period in enumerate(periods):
            axes[i].set_facecolor(COLORS['background'])
            
            # Filter data for this period
            period_data = region_data[region_data['Period'] == period]
            
            if period_data.empty:
                axes[i].text(0.5, 0.5, "No data for this period", 
                           transform=axes[i].transAxes, ha='center', fontsize=12)
                continue
            
            # Calculate average cover for each species and habitat
            habitat_species = period_data.groupby('Habitat')[species_cols].mean().reset_index()
            
            # Convert to long format for stacked bar chart
            habitat_species_long = pd.melt(habitat_species, id_vars=['Habitat'], 
                                         value_vars=species_cols, 
                                         var_name='Species', value_name='Cover_Percent')
            
            # Clean species names
            def clean_species_name(name):
                return name.replace('_', ' ').replace('complex', '').strip()
            
            habitat_species_long['Species_Display'] = habitat_species_long['Species'].apply(clean_species_name)
            
            # Aggregate by habitat to see total
            habitat_totals = habitat_species_long.groupby('Habitat')['Cover_Percent'].sum().sort_values(ascending=False)
            sorted_habitats = habitat_totals.index.tolist()
            
            # Define species color mapping (consistent across all plots)
            species_colors = {}
            all_species = sorted(species_cols)
            color_palette = sns.color_palette("husl", len(all_species))
            
            for s, c in zip(all_species, color_palette):
                species_colors[clean_species_name(s)] = c
            
            # Create stacked bar chart
            habitat_species_pivot = habitat_species_long.pivot_table(
                index='Habitat', columns='Species_Display', values='Cover_Percent', fill_value=0)
            
            # Reindex to sort by total cover
            if all(habitat in habitat_species_pivot.index for habitat in sorted_habitats):
                habitat_species_pivot = habitat_species_pivot.reindex(sorted_habitats)
            
            # Plot stacked bars
            habitat_species_pivot.plot(kind='bar', stacked=True, ax=axes[i], 
                                     colormap='viridis', width=0.7)
            
            # Improve axis labels
            axes[i].set_title(f"{period}", fontweight='bold', fontsize=14)
            axes[i].set_xlabel("Habitat", fontweight='bold')
            
            if i == 0:
                axes[i].set_ylabel("Cover (%)", fontweight='bold')
            
            # Add grid
            axes[i].grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Remove legend from all but the last plot
            if i < len(periods) - 1:
                axes[i].get_legend().remove()
            else:
                # Improve the legend in the last plot
                handles, labels = axes[i].get_legend_handles_labels()
                axes[i].legend(handles, labels, title="Species", loc='upper left', 
                             bbox_to_anchor=(1.05, 1), borderaxespad=0)
        
        # Add an overall title
        fig.suptitle(f'Temporal Changes in Coral Species Composition - {region_name}', 
                    fontsize=20, fontweight='bold', y=0.98,
                    color=COLORS['dark_blue'])
        
        # Add a note about the data
        fig.text(0.5, 0.01, 
                "Note: Stacked bars show average cover (%) of different coral species by habitat type over time.", 
                ha='center', fontsize=10, fontstyle='italic')
        
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
        plt.savefig(os.path.join(results_dir, f"species_temporal_change_{region_code}.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    
    # Create a combined visualization showing temporal changes for key species across regions
    # Select a few key species for this visualization
    key_species = [s for s in ['Orbicella_annularis_complex', 'Acropora_cervicornis', 
                              'Siderastrea_siderea', 'Montastraea_cavernosa'] 
                  if s in species_cols]
    
    if key_species:
        # Calculate temporal trends for each species by region
        region_trends = df.groupby(['Year', 'Subregion'])[key_species].mean().reset_index()
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(len(key_species), 1, figsize=(14, 4*len(key_species)), 
                               sharex=True, facecolor=COLORS['background'])
        
        # Adjust axes for single species case
        if len(key_species) == 1:
            axes = [axes]
        
        # Loop through species
        for i, species in enumerate(key_species):
            axes[i].set_facecolor(COLORS['background'])
            
            # Plot temporal trend for each region
            for region_code, region_name in REGION_NAMES.items():
                region_data = region_trends[region_trends['Subregion'] == region_code]
                if not region_data.empty:
                    axes[i].plot(region_data['Year'], region_data[species], 
                               label=region_name, linewidth=2.5, marker='o', markersize=5)
            
            # Set panel title and labels
            species_display = species.replace('_', ' ').replace('complex', '').strip()
            axes[i].set_title(f'{species_display}', fontweight='bold', fontsize=14)
            axes[i].set_ylabel('Cover (%)', fontweight='bold')
            
            # Add grid
            axes[i].grid(alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add legend (only in the first panel)
            if i == 0:
                axes[i].legend(frameon=True, facecolor='white', framealpha=0.9,
                             title='Region', loc='upper right')
        
        # Set common x-axis label
        axes[-1].set_xlabel('Year', fontweight='bold')
        
        # Add an overall title
        fig.suptitle('Temporal Changes in Key Coral Species by Region (1996-2023)', 
                    fontsize=20, fontweight='bold', y=0.98,
                    color=COLORS['dark_blue'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(results_dir, "key_species_temporal_regional_trends.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
    
    print("Temporal-spatial trend analysis saved.")

def create_species_community_dissimilarity_map(df):
    """
    Create a map showing the dissimilarity in coral community composition between sites.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Creating community dissimilarity map...")
    
    # Get recent data (last 3 years) for current composition
    recent_years = df['Year'].max() - 2
    recent_data = df[df['Year'] >= recent_years]
    
    # Identify all species columns
    species_cols = [col for col in df.columns if col not in 
                   ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                    'Site_name', 'StationID', 'Surveyed_all_years', 'points', 
                    'latDD', 'lonDD', 'Depth_ft', 'Total_Stony_Coral', 'Period']]
    
    # Aggregate data to site level
    site_composition = recent_data.groupby(['Site_name', 'Habitat', 'Subregion', 'latDD', 'lonDD'])[species_cols].mean().reset_index()
    
    # Standardize species data for clustering
    X = site_composition[species_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate pairwise distances between sites based on species composition
    distances = pdist(X_scaled, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')
    
    # Determine clusters (3-5 clusters is usually meaningful for ecological communities)
    num_clusters = 4  # Can be adjusted
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    # Add cluster assignments to the data
    site_composition['Cluster'] = clusters
    
    # Create the map figure
    fig, ax = plt.subplots(figsize=(18, 12), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define boundaries for the Florida Keys area
    min_lon, max_lon = site_composition['lonDD'].min() - 0.2, site_composition['lonDD'].max() + 0.2
    min_lat, max_lat = site_composition['latDD'].min() - 0.1, site_composition['latDD'].max() + 0.1
    
    # Plot simplified Florida Keys outline
    keys_outline_x = np.linspace(min_lon, max_lon, 100)
    keys_outline_y = np.sin((keys_outline_x - min_lon) / (max_lon - min_lon) * np.pi) * 0.05 + min_lat + 0.1
    ax.plot(keys_outline_x, keys_outline_y, 'k-', linewidth=2, alpha=0.5)
    
    # Fill land area with a light color
    ax.fill_between(keys_outline_x, keys_outline_y, min_lat, color='lightgray', alpha=0.3)
    
    # Define colors for different clusters
    cluster_colors = sns.color_palette("Set1", num_clusters)
    
    # Create mapping of cluster to marker shape based on dominant habitat in cluster
    cluster_habitats = {}
    for cluster in range(1, num_clusters + 1):
        cluster_data = site_composition[site_composition['Cluster'] == cluster]
        if not cluster_data.empty:
            habitat_counts = cluster_data['Habitat'].value_counts()
            dominant_habitat = habitat_counts.idxmax() if not habitat_counts.empty else 'unknown'
            cluster_habitats[cluster] = dominant_habitat
    
    # Plot sites colored by cluster
    for cluster in range(1, num_clusters + 1):
        cluster_sites = site_composition[site_composition['Cluster'] == cluster]
        if not cluster_sites.empty:
            # Determine marker shape based on dominant habitat
            dominant_habitat = cluster_habitats.get(cluster, 'unknown')
            marker = '^' if dominant_habitat in ['OS', 'OD'] else 'o'
            
            ax.scatter(
                cluster_sites['lonDD'], 
                cluster_sites['latDD'],
                s=120,  # Marker size
                c=[cluster_colors[cluster-1]],  # Color by cluster
                marker=marker,
                alpha=0.8,
                edgecolor='black',
                linewidth=1,
                label=f'Cluster {cluster}'
            )
    
    # Add site labels for selected sites (one example from each cluster)
    for cluster in range(1, num_clusters + 1):
        cluster_sites = site_composition[site_composition['Cluster'] == cluster]
        if not cluster_sites.empty:
            # Select one representative site (e.g., closest to cluster center)
            site = cluster_sites.iloc[0]
            
            ax.annotate(
                f"{site['Site_name']} (C{cluster})",
                xy=(site['lonDD'], site['latDD']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
    
    # Set plot attributes
    ax.set_title('Coral Community Composition Clusters (2021-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Longitude', fontweight='bold', fontsize=14)
    ax.set_ylabel('Latitude', fontweight='bold', fontsize=14)
    
    # Set axis limits to focus on the Florida Keys
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add legend
    ax.legend(frameon=True, facecolor='white', framealpha=0.9,
             title='Community Cluster')
    
    # Add grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Add region labels
    region_positions = {
        'Upper Keys': (max_lon - 0.3, max_lat - 0.05),
        'Middle Keys': ((max_lon + min_lon) / 2, min_lat + 0.12),
        'Lower Keys': (min_lon + 0.3, min_lat + 0.08)
    }
    
    for region, pos in region_positions.items():
        ax.text(pos[0], pos[1], region, fontsize=14, fontweight='bold', ha='center', va='center',
               color=COLORS['dark_blue'], path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    
    # Add a note explaining clusters
    cluster_descriptions = []
    
    # Analyze each cluster to determine key characteristics
    for cluster in range(1, num_clusters + 1):
        cluster_data = site_composition[site_composition['Cluster'] == cluster]
        if not cluster_data.empty:
            # Find dominant species in this cluster
            cluster_means = cluster_data[species_cols].mean()
            top_species = cluster_means.nlargest(3).index.tolist()
            
            # Format species names for display
            top_species_clean = [sp.replace('_', ' ').replace('complex', '').strip() for sp in top_species]
            
            # Get dominant habitat
            dominant_habitat = cluster_habitats.get(cluster, 'unknown')
            habitat_name = HABITAT_NAMES.get(dominant_habitat, dominant_habitat)
            
            # Create description
            description = f"Cluster {cluster}: Primarily {habitat_name}, dominated by {', '.join(top_species_clean)}"
            cluster_descriptions.append(description)
    
    # Add description text box
    description_text = "\n".join(cluster_descriptions)
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, 
                edgecolor=COLORS['dark_blue'], linewidth=1.5)
    
    ax.text(0.02, 0.02, description_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    # Add information about the clustering method
    method_text = (
        "Clustering Method: Hierarchical clustering based on Euclidean distance\n"
        "between sites in multivariate species space (standardized species abundance data)."
    )
    
    fig.text(0.5, 0.01, method_text, ha='center', va='center', 
             fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(results_dir, "coral_community_clusters.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Create a dendrogram visualization to show the hierarchical relationships
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot the dendrogram
    dendrogram(
        linkage_matrix,
        labels=site_composition['Site_name'].tolist(),
        color_threshold=0.7 * max(linkage_matrix[:,2]),  # Adjust this threshold as needed
        leaf_font_size=10,
        ax=ax
    )
    
    # Enhance the plot
    ax.set_title('Hierarchical Clustering of Coral Communities by Site', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Site', fontweight='bold', fontsize=14)
    ax.set_ylabel('Dissimilarity Distance', fontweight='bold', fontsize=14)
    
    # Add a horizontal line to show the clustering threshold
    ax.axhline(y=linkage_matrix[-(num_clusters-1), 2], c='r', linestyle='--', label=f'Threshold for {num_clusters} clusters')
    
    # Add legend
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "coral_community_dendrogram.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Community dissimilarity map and dendrogram saved.")

def analyze_depth_distribution_patterns(df):
    """
    Analyze how coral species are distributed across depth gradients.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Analyzing depth distribution patterns...")
    
    # Identify focal species present in the data
    species_cols = [col for col in FOCUS_SPECIES if col in df.columns]
    
    # Get recent data (last 5 years) for current patterns
    recent_years = df['Year'].max() - 4
    recent_data = df[df['Year'] >= recent_years].copy()
    
    # Create depth categories
    recent_data['Depth_Range'] = pd.cut(
        recent_data['Depth_ft'],
        bins=[0, 10, 20, 30, 100],
        labels=['0-10 ft', '11-20 ft', '21-30 ft', '30+ ft']
    )
    
    # Calculate average abundance by depth range
    depth_species = recent_data.groupby('Depth_Range', observed=False)[species_cols].mean().reset_index()
    
    # Transform data to long format for easier plotting
    depth_species_long = pd.melt(depth_species, id_vars=['Depth_Range'], 
                               value_vars=species_cols, 
                               var_name='Species', value_name='Cover_Percent')
    
    # Clean species names for display
    def clean_species_name(name):
        return name.replace('_', ' ').replace('complex', '').strip()
    
    depth_species_long['Species_Display'] = depth_species_long['Species'].apply(clean_species_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Define depth range color mapping
    depth_colors = {
        '0-10 ft': COLORS['light_blue'],
        '11-20 ft': COLORS['ocean_blue'],
        '21-30 ft': COLORS['dark_blue'],
        '30+ ft': '#01233D'  # Darker blue
    }
    
    # Create grouped bar chart
    sns.barplot(x='Species_Display', y='Cover_Percent', hue='Depth_Range', 
               data=depth_species_long, palette=depth_colors.values(), ax=ax)
    
    # Enhance the plot
    ax.set_title('Coral Species Distribution by Depth (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    ax.set_xlabel('Coral Species', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Cover (%)', fontweight='bold', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    ax.legend(title='Depth Range', frameon=True, facecolor='white', framealpha=0.9,
             loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
    
    # Calculate and add statistical significance
    # Perform ANOVA for each species to check if depth differences are significant
    from scipy import stats
    
    significance_results = []
    
    for species in species_cols:
        species_data = []
        depth_groups = []
        
        for depth in recent_data['Depth_Range'].dropna().unique():
            depth_data = recent_data[recent_data['Depth_Range'] == depth][species].dropna()
            if len(depth_data) > 0:
                species_data.append(depth_data)
                depth_groups.append(depth)
        
        if len(species_data) >= 2:  # Need at least 2 groups for ANOVA
            f_stat, p_value = stats.f_oneway(*species_data)
            significance_results.append({
                'Species': species,
                'F_statistic': f_stat,
                'P_value': p_value,
                'Significant': p_value < 0.05
            })
    
    # Add more bottom padding for the explanatory notes
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Increase bottom margin to avoid text overlap
    
    # Add explanatory notes
    note_text = (
        "Note: * indicates species with statistically significant (p<0.05)\n"
        "differences in abundance across depth ranges (ANOVA test)."
    )
    
    fig.text(0.5, 0.01, note_text, ha='center', va='center', 
             fontsize=10, fontstyle='italic', color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, "species_depth_distribution.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    # Create a supplementary heatmap
    # Reshape data into a pivot table format
    pivot_data = depth_species_long.pivot(index='Depth_Range', 
                                        columns='Species_Display', 
                                        values='Cover_Percent')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Generate the heatmap with improved styling
    sns.heatmap(
        pivot_data, 
        cmap=coral_cmap,
        annot=True,  # Add value annotations
        fmt=".2f",  # Format to 2 decimal places
        linewidths=0.5,
        cbar_kws={'label': 'Average Cover (%)', 'shrink': 0.8},
        ax=ax
    )
    
    # Set plot titles and labels
    ax.set_title('Heatmap of Coral Species Abundance by Depth (2019-2023)', 
                fontweight='bold', pad=15, fontsize=20)
    
    # Improve colorbar appearance
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Average Cover (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "species_depth_heatmap.png"), 
               bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
    plt.close()
    
    print("Depth distribution analysis saved.")

def analyze_indicator_species(df):
    """
    Identify and analyze indicator species for different habitats and regions.
    
    Args:
        df (DataFrame): Preprocessed DataFrame with coral species data
    """
    print("Analyzing indicator species...")
    
    # Get recent data (last 5 years) for current patterns
    recent_years = df['Year'].max() - 4
    recent_data = df[df['Year'] >= recent_years]
    
    # Identify all species columns
    species_cols = [col for col in df.columns if col not in 
                   ['OID_', 'Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 
                    'Site_name', 'StationID', 'Surveyed_all_years', 'points', 
                    'latDD', 'lonDD', 'Depth_ft', 'Total_Stony_Coral', 'Period']]
    
    # Implement a simplified indicator species analysis
    # We'll use a combination of:
    # 1. Specificity - how much the species is concentrated in the habitat
    # 2. Fidelity - how frequently the species is found in the habitat
    
    # Analyze indicator species for habitats
    habitat_indicators = []
    
    for habitat in recent_data['Habitat'].unique():
        habitat_data = recent_data[recent_data['Habitat'] == habitat]
        non_habitat_data = recent_data[recent_data['Habitat'] != habitat]
        
        if habitat_data.empty or non_habitat_data.empty:
            continue
        
        # Calculate metrics for each species
        for species in species_cols:
            # Skip if species is not present
            if recent_data[species].sum() == 0:
                continue
            
            # Calculate specificity (mean abundance in habitat / mean abundance across all habitats)
            specificity = habitat_data[species].mean() / (recent_data[species].mean() or 1)
            
            # Calculate fidelity (percentage of sites in the habitat where the species is present)
            fidelity = (habitat_data[species] > 0).mean()
            
            # Calculate indicator value (product of specificity and fidelity)
            indicator_value = specificity * fidelity
            
            # Calculate statistical significance using t-test
            if (habitat_data[species].std() > 0 or non_habitat_data[species].std() > 0) and len(habitat_data) > 1 and len(non_habitat_data) > 1:
                t_stat, p_value = stats.ttest_ind(
                    habitat_data[species].dropna(), 
                    non_habitat_data[species].dropna(), 
                    equal_var=False
                )
            else:
                p_value = 1.0
            
            # Store the results if the species has a reasonable presence
            if habitat_data[species].mean() > 0.1 and indicator_value > 1.1:  # Thresholds can be adjusted
                habitat_indicators.append({
                    'Habitat': habitat,
                    'Species': species,
                    'Mean_Abundance': habitat_data[species].mean(),
                    'Specificity': specificity,
                    'Fidelity': fidelity,
                    'Indicator_Value': indicator_value,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
    
    # Convert to DataFrame
    if habitat_indicators:
        habitat_indicators_df = pd.DataFrame(habitat_indicators)
        
        # Sort by indicator value
        habitat_indicators_df = habitat_indicators_df.sort_values(
            ['Habitat', 'Indicator_Value'], ascending=[True, False])
        
        # Clean species names for display
        def clean_species_name(name):
            return name.replace('_', ' ').replace('complex', '').strip()
        
        habitat_indicators_df['Species_Display'] = habitat_indicators_df['Species'].apply(clean_species_name)
        
        # Map habitat codes to full names
        habitat_indicators_df['Habitat_Display'] = habitat_indicators_df['Habitat'].map(HABITAT_NAMES)
        
        # Create a visualizing to display top indicator species for each habitat
        fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['background'])
        
        # Limit to top 5 indicators per habitat for clarity
        top_indicators = habitat_indicators_df.groupby('Habitat').head(5).reset_index(drop=True)
        
        # Color mapping for habitats
        habitat_colors = {
            'OS': COLORS['coral'],
            'OD': COLORS['sand'],
            'P': COLORS['reef_green'],
            'HB': COLORS['ocean_blue'],
            'BCP': COLORS['dark_blue']
        }
        
        # Create a categorical palette where each habitat has a consistent color
        palette = []
        for habitat in top_indicators['Habitat']:
            palette.append(habitat_colors.get(habitat, 'gray'))
        
        # Create bar chart
        sns.barplot(x='Species_Display', y='Indicator_Value', hue='Habitat_Display', 
                   data=top_indicators, palette=habitat_colors.values(), ax=ax)
        
        # Enhance the plot
        ax.set_title('Top Indicator Coral Species by Habitat (2019-2023)', 
                    fontweight='bold', pad=15, fontsize=20)
        ax.set_xlabel('Coral Species', fontweight='bold', fontsize=14)
        ax.set_ylabel('Indicator Value', fontweight='bold', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust legend
        ax.legend(title='Habitat Type', frameon=True, facecolor='white', framealpha=0.9,
                 loc='upper right')
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add more bottom padding for explanatory note
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Increase bottom margin to avoid text overlap
        
        # Add explanatory note
        note_text = (
            "Note: Indicator Value combines specificity (concentration in habitat) and fidelity (frequency in habitat).\n"
            "* indicates statistically significant indicator species (p<0.05, t-test)."
        )
        
        fig.text(0.5, 0.01, note_text, ha='center', va='center', 
                 fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(results_dir, "habitat_indicator_species.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        # Similarly, analyze indicator species for regions
        region_indicators = []
        
        for region in recent_data['Subregion'].unique():
            region_data = recent_data[recent_data['Subregion'] == region]
            non_region_data = recent_data[recent_data['Subregion'] != region]
            
            if region_data.empty or non_region_data.empty:
                continue
            
            # Calculate metrics for each species
            for species in species_cols:
                # Skip if species is not present
                if recent_data[species].sum() == 0:
                    continue
                
                # Calculate specificity
                specificity = region_data[species].mean() / (recent_data[species].mean() or 1)
                
                # Calculate fidelity
                fidelity = (region_data[species] > 0).mean()
                
                # Calculate indicator value
                indicator_value = specificity * fidelity
                
                # Calculate statistical significance
                if (region_data[species].std() > 0 or non_region_data[species].std() > 0) and len(region_data) > 1 and len(non_region_data) > 1:
                    t_stat, p_value = stats.ttest_ind(
                        region_data[species].dropna(), 
                        non_region_data[species].dropna(), 
                        equal_var=False
                    )
                else:
                    p_value = 1.0
                
                # Store the results if the species has a reasonable presence
                if region_data[species].mean() > 0.1 and indicator_value > 1.1:
                    region_indicators.append({
                        'Region': region,
                        'Species': species,
                        'Mean_Abundance': region_data[species].mean(),
                        'Specificity': specificity,
                        'Fidelity': fidelity,
                        'Indicator_Value': indicator_value,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05
                    })
        
        # Convert to DataFrame
        if region_indicators:
            region_indicators_df = pd.DataFrame(region_indicators)
            
            # Sort by indicator value
            region_indicators_df = region_indicators_df.sort_values(
                ['Region', 'Indicator_Value'], ascending=[True, False])
            
            # Clean species names for display
            region_indicators_df['Species_Display'] = region_indicators_df['Species'].apply(clean_species_name)
            
            # Map region codes to full names
            region_indicators_df['Region_Display'] = region_indicators_df['Region'].map(REGION_NAMES)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Limit to top 5 indicators per region for clarity
            top_region_indicators = region_indicators_df.groupby('Region').head(5).reset_index(drop=True)
            
            # Color mapping for regions
            region_colors = {
                'UK': COLORS['dark_blue'],
                'MK': COLORS['ocean_blue'],
                'LK': COLORS['light_blue']
            }
            
            # Create bar chart
            sns.barplot(x='Species_Display', y='Indicator_Value', hue='Region_Display', 
                       data=top_region_indicators, palette=region_colors.values(), ax=ax)
            
            # Enhance the plot
            ax.set_title('Top Indicator Coral Species by Region (2019-2023)', 
                        fontweight='bold', pad=15, fontsize=20)
            ax.set_xlabel('Coral Species', fontweight='bold', fontsize=14)
            ax.set_ylabel('Indicator Value', fontweight='bold', fontsize=14)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust legend
            ax.legend(title='Region', frameon=True, facecolor='white', framealpha=0.9,
                     loc='upper right')
            
            # Add grid for better readability
            ax.grid(axis='y', alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add more bottom padding for explanatory note
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Increase bottom margin to avoid text overlap
            
            # Add explanatory note
            note_text = (
                "Note: Indicator Value combines specificity (concentration in region) and fidelity (frequency in region).\n"
                "* indicates statistically significant indicator species (p<0.05, t-test)."
            )
            
            fig.text(0.5, 0.01, note_text, ha='center', va='center', 
                     fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "region_indicator_species.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
    
    print("Indicator species analysis saved.")

def main():
    """
    Main function to execute the coral species spatial pattern analysis.
    """
    print("\n=== CREMP Coral Species Spatial Pattern Analysis ===")
    print("Starting analysis...")
    
    # Load and preprocess data
    df, stations_df = load_and_preprocess_data()
    
    # Execute analysis functions in a logical order
    
    # 1. First, create a map visualization of species richness
    create_species_richness_map(df, stations_df)
    
    # 2. Analyze species distribution patterns by habitat
    analyze_species_habitat_association(df)
    
    # 3. Analyze species distribution patterns by region
    analyze_species_region_distribution(df)
    
    # 4. Analyze species distribution patterns by depth
    analyze_depth_distribution_patterns(df)
    
    # 5. Identify and analyze indicator species
    analyze_indicator_species(df)
    
    # 6. Analyze temporal changes in spatial distribution
    analyze_species_temporal_spatial_trends(df)
    
    # 7. Create community dissimilarity map
    create_species_community_dissimilarity_map(df)
    
    print("\nAnalysis complete! All results saved in the '{}' directory.".format(results_dir))

if __name__ == "__main__":
    main() 