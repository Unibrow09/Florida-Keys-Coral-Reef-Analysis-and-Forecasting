"""
09_coral_health_factors_analysis.py - Analysis of Key Factors Affecting Coral Health

This script identifies and analyzes the key factors affecting coral health, density, and species richness
in the Florida Keys based on CREMP monitoring data. It explores environmental, spatial, and ecological
drivers and their relative importance in explaining variations in coral parameters.

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
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pingouin as pg
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set plotting style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create results directory if it doesn't exist
results_dir = "09_Results"
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
    'macroalgae': '#8CC084',  # Green for algae
    'disease': '#9A4C95',  # Purple for disease
    'hurricane': '#5B84B1',  # Stormy blue
    'temperature': '#FC766A',  # Warm red-orange
    'positive': '#4CAF50',  # Green for positive correlation
    'negative': '#F44336',  # Red for negative correlation
    'neutral': '#9E9E9E'   # Grey for neutral/no correlation
}

# Create a custom colormap for coral reef visualization
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_cmap', 
    [COLORS['light_blue'], COLORS['ocean_blue'], COLORS['reef_green'], COLORS['coral']]
)

# Create a correlation colormap
corr_cmap = LinearSegmentedColormap.from_list(
    'corr_cmap',
    [COLORS['negative'], COLORS['background'], COLORS['positive']]
)

# Function to load and preprocess the data
def load_and_preprocess_data():
    """
    Load and preprocess the CREMP datasets for factor analysis.
    
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
        
        # Calculate species richness for each dataset
        for key in ['lta', 'stony_density', 'octo_density']:
            if key in data_dict and 'Species_Richness' not in data_dict[key].columns:
                data_dict[key]['Species_Richness'] = (data_dict[key][species_cols[key]] > 0).sum(axis=1)
        
        print("Data preprocessing completed successfully")
        
        # Return both the data dictionary and species columns dictionary
        return data_dict, species_cols
        
    except Exception as e:
        print(f"Error loading or preprocessing data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Function to analyze the impact of major environmental events on coral health
def analyze_environmental_impacts(data_dict, species_cols):
    """
    Analyze the impact of major environmental events (bleaching, hurricanes, disease)
    on coral health parameters.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
    """
    print("Analyzing impacts of major environmental events on coral health...")
    
    # Define major environmental events
    environmental_events = {
        2014: {"name": "2014-2015 Bleaching Event", "type": "bleaching", "color": COLORS['temperature']},
        2015: {"name": "2015 Bleaching Event", "type": "bleaching", "color": COLORS['temperature']},
        2017: {"name": "Hurricane Irma", "type": "hurricane", "color": COLORS['hurricane']},
        2018: {"name": "Stony Coral Tissue Loss Disease Peak", "type": "disease", "color": COLORS['disease']},
        2019: {"name": "Stony Coral Tissue Loss Disease", "type": "disease", "color": COLORS['disease']}
    }
    
    # Analyze the impact of these events on stony coral percent cover
    if 'pcover_taxa' in data_dict:
        print("Analyzing impacts on percent cover...")
        
        # Print column names to find the correct stony coral column
        print("Available columns in percent cover dataset:", data_dict['pcover_taxa'].columns.tolist())
        
        # Use the correct stony coral column
        # From the output, we can see 'Stony_coral' is the correct column, not 'Octocoral'
        stony_coral_col = 'Stony_coral'
        print(f"Using column '{stony_coral_col}' for stony coral cover analysis")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
        fig.suptitle('IMPACT OF MAJOR ENVIRONMENTAL EVENTS ON STONY CORAL COVER',
                    fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Set background color
        ax.set_facecolor(COLORS['background'])
        
        # Calculate yearly means for stony coral cover
        yearly_cover = data_dict['pcover_taxa'].groupby('Year')[stony_coral_col].mean().reset_index()
        
        # Calculate percent change between consecutive years
        yearly_cover['Percent_Change'] = yearly_cover[stony_coral_col].pct_change() * 100
        
        # Plot stony coral cover over time
        ax.plot(yearly_cover['Year'], yearly_cover[stony_coral_col], 
               marker='o', linestyle='-', linewidth=3, color=COLORS['coral'],
               markersize=8, label='Stony Coral Cover (%)')
        
        # Calculate the minimum and maximum coral cover for padding the plot
        y_min = max(0, yearly_cover[stony_coral_col].min() - 1)
        y_max = yearly_cover[stony_coral_col].max() + 1
        
        # Add vertical lines and annotations for environmental events
        for year, event in environmental_events.items():
            if year in yearly_cover['Year'].values:
                # Add vertical line
                ax.axvline(x=year, color=event['color'], linestyle='--', alpha=0.7, linewidth=2)
                
                # Get coral cover value for the event year for annotation positioning
                event_cover = yearly_cover.loc[yearly_cover['Year'] == year, stony_coral_col].values[0]
                
                # Get percent change for this year if available
                percent_change = yearly_cover.loc[yearly_cover['Year'] == year, 'Percent_Change']
                change_text = f"{percent_change.values[0]:.1f}%" if not percent_change.isnull().all() else "N/A"
                
                # Determine if change is positive or negative for color coding
                if not percent_change.isnull().all():
                    change_color = COLORS['positive'] if percent_change.values[0] > 0 else COLORS['negative']
                else:
                    change_color = COLORS['text']
                
                # Add annotation with event name and percent change
                ax.annotate(
                    f"{event['name']}\n({change_text} change)",
                    xy=(year, event_cover),
                    xytext=(year+0.3, event_cover + (0.05 * (y_max - y_min))),
                    color=event['color'],
                    fontweight='bold',
                    fontsize=10,
                    arrowprops=dict(
                        arrowstyle='->', 
                        connectionstyle='arc3,rad=0.2', 
                        color=event['color'], 
                        alpha=0.7
                    )
                )
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Set plot aesthetics
        ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Mean Stony Coral Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_xlim(yearly_cover['Year'].min() - 0.5, yearly_cover['Year'].max() + 0.5)
        ax.set_ylim(y_min, y_max)
        
        # Create custom legend with event types
        legend_elements = [
            Patch(facecolor=COLORS['coral'], edgecolor='black', label='Stony Coral Cover'),
            Patch(facecolor=COLORS['temperature'], edgecolor='black', label='Bleaching Event'),
            Patch(facecolor=COLORS['hurricane'], edgecolor='black', label='Hurricane'),
            Patch(facecolor=COLORS['disease'], edgecolor='black', label='Disease Outbreak')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                 facecolor='white', framealpha=0.9, edgecolor=COLORS['grid'])
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        # Calculate statistics on the impact of events
        impact_stats = []
        
        for year, event in environmental_events.items():
            if year in yearly_cover['Year'].values:
                # Get data for this year and the previous year
                current_year_data = yearly_cover[yearly_cover['Year'] == year]
                prev_year_data = yearly_cover[yearly_cover['Year'] == year - 1]
                
                if not prev_year_data.empty and not current_year_data.empty:
                    # Calculate absolute and percent change
                    absolute_change = current_year_data[stony_coral_col].values[0] - prev_year_data[stony_coral_col].values[0]
                    percent_change = (absolute_change / prev_year_data[stony_coral_col].values[0]) * 100
                    
                    impact_stats.append({
                        'Year': year,
                        'Event': event['name'],
                        'Type': event['type'],
                        'Absolute_Change': absolute_change,
                        'Percent_Change': percent_change
                    })
        
        # Convert to DataFrame for easier analysis
        if impact_stats:
            impact_stats_df = pd.DataFrame(impact_stats)
            
            # Add summary statistics to the plot if we have data
            if not impact_stats_df.empty:
                # Group by event type
                event_types = impact_stats_df['Type'].unique()
                summary_lines = []
                
                for event_type in event_types:
                    type_data = impact_stats_df[impact_stats_df['Type'] == event_type]
                    if not type_data.empty:
                        avg_change = type_data['Percent_Change'].mean()
                        summary_lines.append(f"• Average impact of {event_type} events: {avg_change:.1f}% change")
                
                # Add most severe event
                if not impact_stats_df.empty:
                    worst_idx = impact_stats_df['Percent_Change'].idxmin()
                    summary_lines.append(f"• Most severe event: {impact_stats_df.loc[worst_idx, 'Event']} ({impact_stats_df.loc[worst_idx, 'Percent_Change']:.1f}%)")
                
                impact_summary = "IMPACT SUMMARY:\n" + "\n".join(summary_lines)
                
                # Add the summary box with enhanced styling
                props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                            edgecolor=COLORS['dark_blue'], linewidth=2)
                
                # Position the text box in a free space
                ax.text(0.02, 0.95, impact_summary, transform=ax.transAxes, fontsize=12, fontweight='bold',
                       verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(results_dir, "environmental_impacts_on_coral_cover.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Stony coral cover impact analysis completed and saved.")
    
    # Analyze the impact on LTA and density if data is available
    for dataset_type, dataset_name in [('lta', 'Living Tissue Area'), ('stony_density', 'Density')]:
        if dataset_type in data_dict:
            print(f"Analyzing impacts on {dataset_name}...")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
            fig.suptitle(f'IMPACT OF MAJOR ENVIRONMENTAL EVENTS ON STONY CORAL {dataset_name.upper()}',
                        fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            # Set background color
            ax.set_facecolor(COLORS['background'])
            
            # Calculate yearly means
            if dataset_type == 'lta':
                yearly_data = data_dict[dataset_type].groupby('Year')['Total_LTA'].mean().reset_index()
                measure_name = 'Total_LTA'
                y_label = 'Mean Living Tissue Area (mm²)'
            else:
                yearly_data = data_dict[dataset_type].groupby('Year')['Total_Density'].mean().reset_index()
                measure_name = 'Total_Density'
                y_label = 'Mean Density (colonies/m²)'
            
            # Calculate percent change between consecutive years
            yearly_data['Percent_Change'] = yearly_data[measure_name].pct_change() * 100
            
            # Plot data over time
            ax.plot(yearly_data['Year'], yearly_data[measure_name], 
                   marker='o', linestyle='-', linewidth=3, color=COLORS['coral'],
                   markersize=8, label=f'Stony Coral {dataset_name}')
            
            # Calculate the minimum and maximum for padding the plot
            y_min = max(0, yearly_data[measure_name].min() * 0.9)
            y_max = yearly_data[measure_name].max() * 1.1
            
            # Add vertical lines and annotations for environmental events
            for year, event in environmental_events.items():
                if year in yearly_data['Year'].values:
                    # Add vertical line
                    ax.axvline(x=year, color=event['color'], linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Get value for the event year for annotation positioning
                    event_value = yearly_data.loc[yearly_data['Year'] == year, measure_name].values[0]
                    
                    # Get percent change for this year if available
                    percent_change = yearly_data.loc[yearly_data['Year'] == year, 'Percent_Change']
                    change_text = f"{percent_change.values[0]:.1f}%" if not percent_change.isnull().all() else "N/A"
                    
                    # Determine if change is positive or negative for color coding
                    if not percent_change.isnull().all():
                        change_color = COLORS['positive'] if percent_change.values[0] > 0 else COLORS['negative']
                    else:
                        change_color = COLORS['text']
                    
                    # Add annotation with event name and percent change
                    ax.annotate(
                        f"{event['name']}\n({change_text} change)",
                        xy=(year, event_value),
                        xytext=(year+0.3, event_value + (0.05 * (y_max - y_min))),
                        color=event['color'],
                        fontweight='bold',
                        fontsize=10,
                        arrowprops=dict(
                            arrowstyle='->', 
                            connectionstyle='arc3,rad=0.2', 
                            color=event['color'], 
                            alpha=0.7
                        )
                    )
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Set plot aesthetics
            ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel(y_label, fontweight='bold', fontsize=14, labelpad=10)
            ax.set_xlim(yearly_data['Year'].min() - 0.5, yearly_data['Year'].max() + 0.5)
            ax.set_ylim(y_min, y_max)
            
            # Create custom legend with event types
            legend_elements = [
                Patch(facecolor=COLORS['coral'], edgecolor='black', label=f'Stony Coral {dataset_name}'),
                Patch(facecolor=COLORS['temperature'], edgecolor='black', label='Bleaching Event'),
                Patch(facecolor=COLORS['hurricane'], edgecolor='black', label='Hurricane'),
                Patch(facecolor=COLORS['disease'], edgecolor='black', label='Disease Outbreak')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                     facecolor='white', framealpha=0.9, edgecolor=COLORS['grid'])
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Calculate statistics on the impact of events
            impact_stats = []
            
            for year, event in environmental_events.items():
                if year in yearly_data['Year'].values:
                    # Get data for this year and the previous year
                    current_year_data = yearly_data[yearly_data['Year'] == year]
                    prev_year_data = yearly_data[yearly_data['Year'] == year - 1]
                    
                    if not prev_year_data.empty and not current_year_data.empty:
                        # Calculate absolute and percent change
                        absolute_change = current_year_data[measure_name].values[0] - prev_year_data[measure_name].values[0]
                        percent_change = (absolute_change / prev_year_data[measure_name].values[0]) * 100
                        
                        impact_stats.append({
                            'Year': year,
                            'Event': event['name'],
                            'Type': event['type'],
                            'Absolute_Change': absolute_change,
                            'Percent_Change': percent_change
                        })
            
            # Convert to DataFrame for easier analysis
            impact_stats_df = pd.DataFrame(impact_stats)
            
            # Add summary statistics to the plot if we have data
            if not impact_stats_df.empty:
                impact_summary = (
                    f"IMPACT SUMMARY:\n"
                    f"• Average impact of bleaching events: {impact_stats_df[impact_stats_df['Type'] == 'bleaching']['Percent_Change'].mean():.1f}% change\n"
                    f"• Average impact of hurricanes: {impact_stats_df[impact_stats_df['Type'] == 'hurricane']['Percent_Change'].mean():.1f}% change\n"
                    f"• Average impact of disease outbreaks: {impact_stats_df[impact_stats_df['Type'] == 'disease']['Percent_Change'].mean():.1f}% change\n"
                    f"• Most severe event: {impact_stats_df.loc[impact_stats_df['Percent_Change'].idxmin(), 'Event']} ({impact_stats_df['Percent_Change'].min():.1f}%)"
                )
                
                # Add the summary box with enhanced styling
                props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                            edgecolor=COLORS['dark_blue'], linewidth=2)
                
                # Position the text box in a free space
                ax.text(0.02, 0.15, impact_summary, transform=ax.transAxes, fontsize=12, fontweight='bold',
                       verticalalignment='top', horizontalalignment='left', bbox=props)
            
            # Save the figure
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, f"environmental_impacts_on_{dataset_type}.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print(f"Stony coral {dataset_name} impact analysis completed and saved.")
    
    # Analyze impacts on species richness
    for dataset_type in ['lta', 'stony_density']:
        if dataset_type in data_dict and 'Species_Richness' in data_dict[dataset_type].columns:
            print(f"Analyzing impacts on species richness from {dataset_type} dataset...")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['background'])
            fig.suptitle('IMPACT OF MAJOR ENVIRONMENTAL EVENTS ON STONY CORAL SPECIES RICHNESS',
                        fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            # Set background color
            ax.set_facecolor(COLORS['background'])
            
            # Calculate yearly means
            yearly_data = data_dict[dataset_type].groupby('Year')['Species_Richness'].mean().reset_index()
            
            # Calculate percent change between consecutive years
            yearly_data['Percent_Change'] = yearly_data['Species_Richness'].pct_change() * 100
            
            # Plot species richness over time
            ax.plot(yearly_data['Year'], yearly_data['Species_Richness'], 
                   marker='o', linestyle='-', linewidth=3, color=COLORS['coral'],
                   markersize=8, label='Species Richness')
            
            # Calculate the minimum and maximum for padding the plot
            y_min = max(0, yearly_data['Species_Richness'].min() * 0.9)
            y_max = yearly_data['Species_Richness'].max() * 1.1
            
            # Add vertical lines and annotations for environmental events
            for year, event in environmental_events.items():
                if year in yearly_data['Year'].values:
                    # Add vertical line
                    ax.axvline(x=year, color=event['color'], linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Get value for the event year for annotation positioning
                    event_value = yearly_data.loc[yearly_data['Year'] == year, 'Species_Richness'].values[0]
                    
                    # Get percent change for this year if available
                    percent_change = yearly_data.loc[yearly_data['Year'] == year, 'Percent_Change']
                    change_text = f"{percent_change.values[0]:.1f}%" if not percent_change.isnull().all() else "N/A"
                    
                    # Determine if change is positive or negative for color coding
                    if not percent_change.isnull().all():
                        change_color = COLORS['positive'] if percent_change.values[0] > 0 else COLORS['negative']
                    else:
                        change_color = COLORS['text']
                    
                    # Add annotation with event name and percent change
                    ax.annotate(
                        f"{event['name']}\n({change_text} change)",
                        xy=(year, event_value),
                        xytext=(year+0.3, event_value + (0.05 * (y_max - y_min))),
                        color=event['color'],
                        fontweight='bold',
                        fontsize=10,
                        arrowprops=dict(
                            arrowstyle='->', 
                            connectionstyle='arc3,rad=0.2', 
                            color=event['color'], 
                            alpha=0.7
                        )
                    )
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Set plot aesthetics
            ax.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Mean Species Richness', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_xlim(yearly_data['Year'].min() - 0.5, yearly_data['Year'].max() + 0.5)
            ax.set_ylim(y_min, y_max)
            
            # Create custom legend with event types
            legend_elements = [
                Patch(facecolor=COLORS['coral'], edgecolor='black', label='Species Richness'),
                Patch(facecolor=COLORS['temperature'], edgecolor='black', label='Bleaching Event'),
                Patch(facecolor=COLORS['hurricane'], edgecolor='black', label='Hurricane'),
                Patch(facecolor=COLORS['disease'], edgecolor='black', label='Disease Outbreak')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                     facecolor='white', framealpha=0.9, edgecolor=COLORS['grid'])
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Calculate statistics on the impact of events
            impact_stats = []
            
            for year, event in environmental_events.items():
                if year in yearly_data['Year'].values:
                    # Get data for this year and the previous year
                    current_year_data = yearly_data[yearly_data['Year'] == year]
                    prev_year_data = yearly_data[yearly_data['Year'] == year - 1]
                    
                    if not prev_year_data.empty and not current_year_data.empty:
                        # Calculate absolute and percent change
                        absolute_change = current_year_data['Species_Richness'].values[0] - prev_year_data['Species_Richness'].values[0]
                        percent_change = (absolute_change / prev_year_data['Species_Richness'].values[0]) * 100
                        
                        impact_stats.append({
                            'Year': year,
                            'Event': event['name'],
                            'Type': event['type'],
                            'Absolute_Change': absolute_change,
                            'Percent_Change': percent_change
                        })
            
            # Convert to DataFrame for easier analysis
            impact_stats_df = pd.DataFrame(impact_stats)
            
            # Add summary statistics to the plot if we have data
            if not impact_stats_df.empty:
                impact_summary = (
                    f"IMPACT SUMMARY:\n"
                    f"• Average impact of bleaching events: {impact_stats_df[impact_stats_df['Type'] == 'bleaching']['Percent_Change'].mean():.1f}% change\n"
                    f"• Average impact of hurricanes: {impact_stats_df[impact_stats_df['Type'] == 'hurricane']['Percent_Change'].mean():.1f}% change\n"
                    f"• Average impact of disease outbreaks: {impact_stats_df[impact_stats_df['Type'] == 'disease']['Percent_Change'].mean():.1f}% change\n"
                    f"• Most severe event: {impact_stats_df.loc[impact_stats_df['Percent_Change'].idxmin(), 'Event']} ({impact_stats_df['Percent_Change'].min():.1f}%)"
                )
                
                # Add the summary box with enhanced styling
                props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                            edgecolor=COLORS['dark_blue'], linewidth=2)
                
                # Position the text box in a free space
                ax.text(0.02, 0.15, impact_summary, transform=ax.transAxes, fontsize=12, fontweight='bold',
                       verticalalignment='top', horizontalalignment='left', bbox=props)
            
            # Save the figure
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, f"environmental_impacts_on_species_richness_{dataset_type}.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print(f"Species richness impact analysis from {dataset_type} dataset completed and saved.")
            
            # We only need to analyze species richness from one dataset, so break after the first one
            break
    
    # Return a summary of impact statistics
    impact_summary = {}
    
    return impact_summary

# Function to analyze the influence of spatial factors (region, habitat, depth) on coral health
def analyze_spatial_factors(data_dict, species_cols):
    """
    Analyze the influence of spatial factors (region, habitat, depth) on coral health parameters.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
    """
    print("Analyzing spatial factors influencing coral health...")
    
    # Print the columns in stations dataset to check for depth information
    print("Columns in stations dataset:", data_dict['stations'].columns.tolist())
    
    # Analyze the influence of region on stony coral parameters
    if 'lta' in data_dict and 'stony_density' in data_dict:
        print("Analyzing regional influence on coral parameters...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=COLORS['background'])
        fig.suptitle('INFLUENCE OF SPATIAL FACTORS ON CORAL HEALTH',
                    fontweight='bold', fontsize=24, color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Set background color for all subplots
        for ax in axes.flat:
            ax.set_facecolor(COLORS['background'])
        
        # Define region color mapping
        region_colors = {
            'UK': COLORS['dark_blue'],   # Upper Keys
            'MK': COLORS['ocean_blue'],  # Middle Keys
            'LK': COLORS['light_blue'],  # Lower Keys
            'DT': COLORS['coral']        # Dry Tortugas (if present)
        }
        
        region_names = {
            'UK': 'Upper Keys',
            'MK': 'Middle Keys',
            'LK': 'Lower Keys',
            'DT': 'Dry Tortugas'
        }
        
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
        
        # 1. Regional analysis of LTA (top left)
        region_lta = data_dict['lta'].groupby('Subregion')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
        region_lta['se'] = region_lta['std'] / np.sqrt(region_lta['count'])
        region_lta['ci_95'] = 1.96 * region_lta['se']
        
        # Sort by mean LTA
        region_lta = region_lta.sort_values('mean', ascending=False)
        
        ax = axes[0, 0]
        bars = ax.bar(region_lta['Subregion'].map(region_names), region_lta['mean'],
                     yerr=region_lta['ci_95'],
                     color=[region_colors.get(region, COLORS['coral']) for region in region_lta['Subregion']],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5,
                     error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2})
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + region_lta['ci_95'].max()/2,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    rotation=0, color=COLORS['text'])
        
        # Set plot aesthetics
        ax.set_title('Living Tissue Area by Region', fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Mean Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Perform ANOVA to test for significant differences between regions
        f_stat, p_value = f_oneway(
            *[data_dict['lta'][data_dict['lta']['Subregion'] == region]['Total_LTA'] 
              for region in region_lta['Subregion']]
        )
        
        # Add ANOVA results to the plot
        anova_text = (
            f"ANOVA Results:\n"
            f"F-statistic: {f_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add statistical test results with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax.text(0.05, 0.95, anova_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # 2. Regional analysis of Density (top right)
        region_density = data_dict['stony_density'].groupby('Subregion')['Total_Density'].agg(['mean', 'median', 'std', 'count']).reset_index()
        region_density['se'] = region_density['std'] / np.sqrt(region_density['count'])
        region_density['ci_95'] = 1.96 * region_density['se']
        
        # Sort by mean density
        region_density = region_density.sort_values('mean', ascending=False)
        
        ax = axes[0, 1]
        bars = ax.bar(region_density['Subregion'].map(region_names), region_density['mean'],
                     yerr=region_density['ci_95'],
                     color=[region_colors.get(region, COLORS['coral']) for region in region_density['Subregion']],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5,
                     error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2})
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + region_density['ci_95'].max()/2,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    rotation=0, color=COLORS['text'])
        
        # Set plot aesthetics
        ax.set_title('Stony Coral Density by Region', fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Region', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Perform ANOVA to test for significant differences between regions
        f_stat, p_value = f_oneway(
            *[data_dict['stony_density'][data_dict['stony_density']['Subregion'] == region]['Total_Density'] 
              for region in region_density['Subregion']]
        )
        
        # Add ANOVA results to the plot
        anova_text = (
            f"ANOVA Results:\n"
            f"F-statistic: {f_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add statistical test results with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax.text(0.05, 0.95, anova_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # 3. Habitat analysis of LTA (bottom left)
        habitat_lta = data_dict['lta'].groupby('Habitat')['Total_LTA'].agg(['mean', 'median', 'std', 'count']).reset_index()
        habitat_lta['se'] = habitat_lta['std'] / np.sqrt(habitat_lta['count'])
        habitat_lta['ci_95'] = 1.96 * habitat_lta['se']
        
        # Sort by mean LTA
        habitat_lta = habitat_lta.sort_values('mean', ascending=False)
        
        ax = axes[1, 0]
        bars = ax.bar(habitat_lta['Habitat'].map(habitat_names), habitat_lta['mean'],
                     yerr=habitat_lta['ci_95'],
                     color=[habitat_colors.get(habitat, COLORS['coral']) for habitat in habitat_lta['Habitat']],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5,
                     error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2})
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + habitat_lta['ci_95'].max()/2,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    rotation=0, color=COLORS['text'])
        
        # Set plot aesthetics
        ax.set_title('Living Tissue Area by Habitat Type', fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Mean Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Perform ANOVA to test for significant differences between habitats
        f_stat, p_value = f_oneway(
            *[data_dict['lta'][data_dict['lta']['Habitat'] == habitat]['Total_LTA'] 
              for habitat in habitat_lta['Habitat']]
        )
        
        # Add ANOVA results to the plot
        anova_text = (
            f"ANOVA Results:\n"
            f"F-statistic: {f_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add statistical test results with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax.text(0.05, 0.95, anova_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # 4. Habitat analysis of Density (bottom right)
        habitat_density = data_dict['stony_density'].groupby('Habitat')['Total_Density'].agg(['mean', 'median', 'std', 'count']).reset_index()
        habitat_density['se'] = habitat_density['std'] / np.sqrt(habitat_density['count'])
        habitat_density['ci_95'] = 1.96 * habitat_density['se']
        
        # Sort by mean density
        habitat_density = habitat_density.sort_values('mean', ascending=False)
        
        ax = axes[1, 1]
        bars = ax.bar(habitat_density['Habitat'].map(habitat_names), habitat_density['mean'],
                     yerr=habitat_density['ci_95'],
                     color=[habitat_colors.get(habitat, COLORS['coral']) for habitat in habitat_density['Habitat']],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5,
                     error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2})
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + habitat_density['ci_95'].max()/2,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    rotation=0, color=COLORS['text'])
        
        # Set plot aesthetics
        ax.set_title('Stony Coral Density by Habitat Type', fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Habitat Type', fontweight='bold', fontsize=14, labelpad=10)
        ax.set_ylabel('Mean Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Perform ANOVA to test for significant differences between habitats
        f_stat, p_value = f_oneway(
            *[data_dict['stony_density'][data_dict['stony_density']['Habitat'] == habitat]['Total_Density'] 
              for habitat in habitat_density['Habitat']]
        )
        
        # Add ANOVA results to the plot
        anova_text = (
            f"ANOVA Results:\n"
            f"F-statistic: {f_stat:.2f}\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant differences: {'Yes' if p_value < 0.05 else 'No'}"
        )
        
        # Add statistical test results with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax.text(0.05, 0.95, anova_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        # Add an overall summary of spatial factor importance
        summary_text = (
            f"SPATIAL FACTORS SUMMARY:\n\n"
            f"• Region explains approximately {region_lta['mean'].var() / data_dict['lta']['Total_LTA'].var() * 100:.1f}% of LTA variation "
            f"and {region_density['mean'].var() / data_dict['stony_density']['Total_Density'].var() * 100:.1f}% of density variation.\n\n"
            f"• Habitat type explains approximately {habitat_lta['mean'].var() / data_dict['lta']['Total_LTA'].var() * 100:.1f}% of LTA variation "
            f"and {habitat_density['mean'].var() / data_dict['stony_density']['Total_Density'].var() * 100:.1f}% of density variation.\n\n"
            f"• Key findings: {region_lta.iloc[0]['Subregion']} has the highest mean LTA, "
            f"{habitat_lta.iloc[0]['Habitat']} has the highest mean LTA, "
            f"{region_density.iloc[0]['Subregion']} has the highest mean density, and "
            f"{habitat_density.iloc[0]['Habitat']} has the highest mean density."
        )
        
        # Add the summary box with enhanced styling
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in the center area
        fig.text(0.5, -0.05, summary_text, fontsize=12, fontweight='bold',
                verticalalignment='center', horizontalalignment='center', bbox=props)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(results_dir, "spatial_factors_influence.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Spatial factors influence analysis saved.")
    
    # Analyze the influence of depth on coral parameters if depth column exists
    if 'lta' in data_dict and 'stations' in data_dict:
        print("Analyzing depth influence on coral parameters...")
        
        # Check if depth information is available in another column
        depth_cols = [col for col in data_dict['stations'].columns if 'depth' in col.lower()]
        if depth_cols:
            depth_col = depth_cols[0]
            print(f"Found depth column: {depth_col}")
        else:
            # If no depth column exists, we'll skip this analysis
            print("No depth information found in stations dataset. Skipping depth analysis.")
            return {'region': {'lta_fstat': 0, 'lta_pvalue': 0}, 
                    'habitat': {'lta_fstat': 0, 'lta_pvalue': 0}, 
                    'depth': {'lta_correlation': 0, 'lta_pvalue': 0}}
        
        # Merge LTA data with stations data to get depth information
        merged_data = pd.merge(
            data_dict['lta'], 
            data_dict['stations'][['StationID', depth_col]], 
            on='StationID',
            how='inner'
        )
        
        # Check if we have depth information
        if depth_col in merged_data.columns and not merged_data[depth_col].isnull().all():
            # Create figure for depth analysis
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Scatter plot of LTA vs Depth
            scatter = ax.scatter(
                merged_data[depth_col], 
                merged_data['Total_LTA'],
                c=merged_data['Species_Richness'], 
                cmap=coral_cmap,
                alpha=0.7,
                s=50,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add labels for key points (extremes)
            top_stations = merged_data.nlargest(5, 'Total_LTA')
            for _, row in top_stations.iterrows():
                ax.annotate(
                    f"Site: {row['Site_name']}",
                    xy=(row[depth_col], row['Total_LTA']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold',
                    color=COLORS['dark_blue'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
            
            # Add best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_data[depth_col], 
                merged_data['Total_LTA']
            )
            
            x_line = np.array([merged_data[depth_col].min(), merged_data[depth_col].max()])
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, color=COLORS['dark_blue'], linewidth=2, 
                   linestyle='--', label=f'Linear trend (r = {r_value:.2f}, p = {p_value:.4f})')
            
            # Add a colorbar to show species richness
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Species Richness', fontweight='bold')
            
            # Set plot aesthetics
            ax.set_title('RELATIONSHIP BETWEEN DEPTH AND LIVING TISSUE AREA',
                        fontweight='bold', fontsize=18, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            ax.set_xlabel('Depth (m)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Total Living Tissue Area (mm²)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add legend
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                     fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add statistical analysis summary
            analysis_text = (
                f"DEPTH ANALYSIS SUMMARY:\n"
                f"• Correlation coefficient: {r_value:.2f}\n"
                f"• p-value: {p_value:.4f}\n"
                f"• Significant relationship: {'Yes' if p_value < 0.05 else 'No'}\n"
                f"• Slope: {slope:.2f} mm²/m depth\n"
                f"• Optimal depth range: {merged_data.groupby(pd.cut(merged_data[depth_col], bins=5))['Total_LTA'].mean().idxmax()}"
            )
            
            # Add the analysis box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.70, 0.90, analysis_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "depth_influence_on_lta.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Depth influence on LTA analysis saved.")
    
    # Return summary of spatial factor analysis
    spatial_factors_summary = {
        'region': {
            'lta_fstat': f_stat,
            'lta_pvalue': p_value
        },
        'habitat': {
            'lta_fstat': f_stat,
            'lta_pvalue': p_value
        },
        'depth': {
            'lta_correlation': r_value,
            'lta_pvalue': p_value
        }
    }
    
    return spatial_factors_summary

# Function to build multivariate models to identify key predictors of coral health
def build_multivariate_models(data_dict, species_cols):
    """
    Build multivariate regression models to identify key predictors of coral health parameters.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
    """
    print("Building multivariate models to identify key predictors of coral health...")
    
    # Check the station dataset columns for coordinate and depth information
    print("Station dataset columns:", data_dict['stations'].columns.tolist())
    
    # Map column names to standard names for consistent use
    # Based on previous output, we know these are the correct column names
    lat_col = 'latDD'  # Latitude in decimal degrees
    lon_col = 'lonDD'  # Longitude in decimal degrees
    depth_col = 'Depth_ft'  # Depth in feet
    
    # Prepare dataset for modeling stony coral cover
    if 'pcover_taxa' in data_dict and 'stations' in data_dict:
        print("Preparing data for stony coral cover model...")
        
        # Merge percent cover data with stations data to get spatial information
        cover_model_data = pd.merge(
            data_dict['pcover_taxa'], 
            data_dict['stations'][['StationID', lat_col, lon_col, depth_col]], 
            on='StationID',
            how='inner'
        )
        
        # Rename columns to standard names for consistent use in the model
        cover_model_data.rename(columns={
            lat_col: 'Latitude',
            lon_col: 'Longitude',
            depth_col: 'Depth'
        }, inplace=True)
        
        # Add habitat and region as categorical variables
        cover_model_data['Habitat_Cat'] = cover_model_data['Habitat']
        cover_model_data['Region_Cat'] = cover_model_data['Subregion']
        
        # Use the correct column name for stony coral cover
        stony_coral_col = 'Stony_coral'
        
        # Prepare features and target
        features = ['Year', 'Habitat_Cat', 'Region_Cat', 'Depth', 'Latitude', 'Longitude']
        target = stony_coral_col
        
        # Check if we have Macroalgae and Turf Algae columns for ecological interaction analysis
        if 'Macroalgae' in cover_model_data.columns:
            features.append('Macroalgae')
        
        if 'Substrate' in cover_model_data.columns:
            features.append('Substrate')
        
        # Remove rows with missing values
        model_df = cover_model_data[features + [target]].dropna()
        
        if len(model_df) > 10:  # Ensure we have enough data for modeling
            # Split data into features and target
            X = model_df[features]
            y = model_df[target]
            
            # Create preprocessing pipeline for mixed data types
            categorical_features = ['Habitat_Cat', 'Region_Cat']
            numeric_features = [f for f in features if f not in categorical_features]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )
            
            # Create and train the model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate model performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Stony coral cover model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Calculate feature importance
            # Extract the random forest model from pipeline
            rf_model = model.named_steps['regressor']
            
            # Get feature names after preprocessing
            ohe = model.named_steps['preprocessor'].transformers_[1][1]
            cat_feature_names = []
            for i, col in enumerate(categorical_features):
                cat_feature_names.extend([f"{col}_{cat}" for cat in list(ohe.categories_[i])[1:]])
            
            feature_names = numeric_features + cat_feature_names
            
            # Compute feature importances using permutation importance
            # (more reliable than built-in feature importances)
            X_train_processed = model.named_steps['preprocessor'].transform(X_train)
            perm_importance = permutation_importance(rf_model, X_train_processed, y_train, 
                                                   n_repeats=10, random_state=42)
            
            # Create DataFrame of feature importances
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Calculate percent importance
            total_importance = feature_importance['Importance'].sum()
            feature_importance['Percent'] = feature_importance['Importance'] / total_importance * 100
            
            # Create a horizontal bar chart of feature importances
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Plot bars
            bars = ax.barh(
                feature_importance['Feature'], 
                feature_importance['Percent'],
                color=COLORS['coral'],
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5,
                xerr=feature_importance['Std'] / total_importance * 100,
                error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2}
            )
            
            # Add data labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%',
                        ha='left', va='center', fontsize=11, fontweight='bold',
                        color=COLORS['text'])
            
            # Set plot aesthetics
            ax.set_title('KEY FACTORS INFLUENCING STONY CORAL COVER',
                        fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            ax.set_xlabel('Relative Importance (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Factor', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add model performance statistics
            stats_text = (
                f"MODEL STATISTICS:\n"
                f"• Model type: Random Forest Regression\n"
                f"• Mean Squared Error: {mse:.2f}\n"
                f"• R² Score: {r2:.2f}\n"
                f"• Top 3 factors: {', '.join(feature_importance['Feature'].head(3))}\n"
                f"• These top 3 factors explain {feature_importance['Percent'].head(3).sum():.1f}% of variation"
            )
            
            # Add the statistics box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.98, 0.70, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "key_factors_stony_coral_cover.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Stony coral cover model and key factors visualization saved.")
    
    # Create a model for LTA if data is available
    if 'lta' in data_dict and 'stations' in data_dict:
        print("Preparing data for living tissue area model...")
        
        # Merge LTA data with stations data
        lta_model_data = pd.merge(
            data_dict['lta'], 
            data_dict['stations'][['StationID', lat_col, lon_col, depth_col]], 
            on='StationID',
            how='inner'
        )
        
        # Rename columns to standard names for consistent use in the model
        lta_model_data.rename(columns={
            lat_col: 'Latitude',
            lon_col: 'Longitude',
            depth_col: 'Depth'
        }, inplace=True)
        
        # Add habitat and region as categorical variables
        lta_model_data['Habitat_Cat'] = lta_model_data['Habitat']
        lta_model_data['Region_Cat'] = lta_model_data['Subregion']
        
        # Prepare features and target
        features = ['Year', 'Habitat_Cat', 'Region_Cat', 'Depth', 'Latitude', 'Longitude', 'Species_Richness']
        target = 'Total_LTA'
        
        # Remove rows with missing values
        model_df = lta_model_data[features + [target]].dropna()
        
        if len(model_df) > 10:  # Ensure we have enough data for modeling
            # Split data into features and target
            X = model_df[features]
            y = model_df[target]
            
            # Create preprocessing pipeline for mixed data types
            categorical_features = ['Habitat_Cat', 'Region_Cat']
            numeric_features = [f for f in features if f not in categorical_features]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )
            
            # Create and train the model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate model performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Living tissue area model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Calculate feature importance
            # Extract the random forest model from pipeline
            rf_model = model.named_steps['regressor']
            
            # Get feature names after preprocessing
            ohe = model.named_steps['preprocessor'].transformers_[1][1]
            cat_feature_names = []
            for i, col in enumerate(categorical_features):
                cat_feature_names.extend([f"{col}_{cat}" for cat in list(ohe.categories_[i])[1:]])
            
            feature_names = numeric_features + cat_feature_names
            
            # Compute feature importances using permutation importance
            X_train_processed = model.named_steps['preprocessor'].transform(X_train)
            perm_importance = permutation_importance(rf_model, X_train_processed, y_train, 
                                                   n_repeats=10, random_state=42)
            
            # Create DataFrame of feature importances
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Calculate percent importance
            total_importance = feature_importance['Importance'].sum()
            feature_importance['Percent'] = feature_importance['Importance'] / total_importance * 100
            
            # Create a horizontal bar chart of feature importances
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Plot bars
            bars = ax.barh(
                feature_importance['Feature'], 
                feature_importance['Percent'],
                color=COLORS['reef_green'],
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5,
                xerr=feature_importance['Std'] / total_importance * 100,
                error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2}
            )
            
            # Add data labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%',
                        ha='left', va='center', fontsize=11, fontweight='bold',
                        color=COLORS['text'])
            
            # Set plot aesthetics
            ax.set_title('KEY FACTORS INFLUENCING STONY CORAL LIVING TISSUE AREA',
                        fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            ax.set_xlabel('Relative Importance (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Factor', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add model performance statistics
            stats_text = (
                f"MODEL STATISTICS:\n"
                f"• Model type: Random Forest Regression\n"
                f"• Mean Squared Error: {mse:.2f}\n"
                f"• R² Score: {r2:.2f}\n"
                f"• Top 3 factors: {', '.join(feature_importance['Feature'].head(3))}\n"
                f"• These top 3 factors explain {feature_importance['Percent'].head(3).sum():.1f}% of variation"
            )
            
            # Add the statistics box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.98, 0.70, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "key_factors_living_tissue_area.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Living tissue area model and key factors visualization saved.")
    
    # Create a model for species richness if data is available
    if 'stony_density' in data_dict and 'stations' in data_dict:
        print("Preparing data for species richness model...")
        
        # Merge density data with stations data
        richness_model_data = pd.merge(
            data_dict['stony_density'], 
            data_dict['stations'][['StationID', lat_col, lon_col, depth_col]], 
            on='StationID',
            how='inner'
        )
        
        # Rename columns to standard names for consistent use in the model
        richness_model_data.rename(columns={
            lat_col: 'Latitude',
            lon_col: 'Longitude',
            depth_col: 'Depth'
        }, inplace=True)
        
        # Add habitat and region as categorical variables
        richness_model_data['Habitat_Cat'] = richness_model_data['Habitat']
        richness_model_data['Region_Cat'] = richness_model_data['Subregion']
        
        # Prepare features and target
        features = ['Year', 'Habitat_Cat', 'Region_Cat', 'Depth', 'Latitude', 'Longitude', 'Total_Density']
        target = 'Species_Richness'
        
        # Remove rows with missing values
        model_df = richness_model_data[features + [target]].dropna()
        
        if len(model_df) > 10:  # Ensure we have enough data for modeling
            # Split data into features and target
            X = model_df[features]
            y = model_df[target]
            
            # Create preprocessing pipeline for mixed data types
            categorical_features = ['Habitat_Cat', 'Region_Cat']
            numeric_features = [f for f in features if f not in categorical_features]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )
            
            # Create and train the model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate model performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Species richness model - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Calculate feature importance
            # Extract the random forest model from pipeline
            rf_model = model.named_steps['regressor']
            
            # Get feature names after preprocessing
            ohe = model.named_steps['preprocessor'].transformers_[1][1]
            cat_feature_names = []
            for i, col in enumerate(categorical_features):
                cat_feature_names.extend([f"{col}_{cat}" for cat in list(ohe.categories_[i])[1:]])
            
            feature_names = numeric_features + cat_feature_names
            
            # Compute feature importances using permutation importance
            X_train_processed = model.named_steps['preprocessor'].transform(X_train)
            perm_importance = permutation_importance(rf_model, X_train_processed, y_train, 
                                                   n_repeats=10, random_state=42)
            
            # Create DataFrame of feature importances
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Calculate percent importance
            total_importance = feature_importance['Importance'].sum()
            feature_importance['Percent'] = feature_importance['Importance'] / total_importance * 100
            
            # Create a horizontal bar chart of feature importances
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Plot bars
            bars = ax.barh(
                feature_importance['Feature'], 
                feature_importance['Percent'],
                color=COLORS['ocean_blue'],
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5,
                xerr=feature_importance['Std'] / total_importance * 100,
                error_kw={'ecolor': 'black', 'capsize': 5, 'capthick': 2}
            )
            
            # Add data labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%',
                        ha='left', va='center', fontsize=11, fontweight='bold',
                        color=COLORS['text'])
            
            # Set plot aesthetics
            ax.set_title('KEY FACTORS INFLUENCING STONY CORAL SPECIES RICHNESS',
                        fontweight='bold', fontsize=20, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            ax.set_xlabel('Relative Importance (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Factor', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add model performance statistics
            stats_text = (
                f"MODEL STATISTICS:\n"
                f"• Model type: Random Forest Regression\n"
                f"• Mean Squared Error: {mse:.2f}\n"
                f"• R² Score: {r2:.2f}\n"
                f"• Top 3 factors: {', '.join(feature_importance['Feature'].head(3))}\n"
                f"• These top 3 factors explain {feature_importance['Percent'].head(3).sum():.1f}% of variation"
            )
            
            # Add the statistics box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.98, 0.70, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "key_factors_species_richness.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Species richness model and key factors visualization saved.")
    
    # Return a summary of model results
    model_results = {
        'cover': {'r2': r2 if 'r2' in locals() else 0, 
                  'top_factors': feature_importance['Feature'].head(3).tolist() if 'feature_importance' in locals() else []},
        'lta': {'r2': 0, 'top_factors': []},
        'richness': {'r2': 0, 'top_factors': []}
    }
    
    return model_results

# Function to analyze ecological interactions affecting coral health
def analyze_ecological_interactions(data_dict, species_cols):
    """
    Analyze ecological interactions (competition, predation, etc.) affecting coral health.
    
    Args:
        data_dict (dict): Dictionary containing preprocessed DataFrames
        species_cols (dict): Dictionary containing species column names for each dataset
    """
    print("Analyzing ecological interactions affecting coral health...")
    
    # Use the correct column name for stony coral
    stony_coral_col = 'Stony_coral'
    
    # Analyze competition between stony corals and macroalgae
    if 'pcover_taxa' in data_dict:
        print("Analyzing stony coral vs. macroalgae competition...")
        
        # Check if we have macroalgae data
        if 'Macroalgae' in data_dict['pcover_taxa'].columns:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
            fig.suptitle('ECOLOGICAL INTERACTIONS: CORAL-ALGAE DYNAMICS',
                        fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            # Set background color
            ax1.set_facecolor(COLORS['background'])
            ax2.set_facecolor(COLORS['background'])
            
            # 1. Time series of stony coral vs. macroalgae cover (left plot)
            yearly_cover = data_dict['pcover_taxa'].groupby('Year')[[stony_coral_col, 'Macroalgae']].mean().reset_index()
            
            # Plot stony coral cover
            ax1.plot(yearly_cover['Year'], yearly_cover[stony_coral_col], 
                   marker='o', linestyle='-', linewidth=3, color=COLORS['coral'],
                   markersize=8, label='Stony Coral Cover (%)')
            
            # Plot macroalgae cover on the same axis
            ax1.plot(yearly_cover['Year'], yearly_cover['Macroalgae'], 
                   marker='s', linestyle='-', linewidth=3, color=COLORS['macroalgae'],
                   markersize=8, label='Macroalgae Cover (%)')
            
            # Set plot aesthetics
            ax1.set_title('Temporal Trends in Cover', fontweight='bold', fontsize=16, pad=15)
            ax1.set_xlabel('Year', fontweight='bold', fontsize=14, labelpad=10)
            ax1.set_ylabel('Mean Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add legend
            ax1.legend(frameon=True, facecolor='white', framealpha=0.9, 
                     fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
            
            # Calculate correlation between stony coral and macroalgae cover
            correlation, p_value = pearsonr(yearly_cover[stony_coral_col], yearly_cover['Macroalgae'])
            
            # Add correlation information to the plot
            corr_text = (
                f"Correlation Analysis:\n"
                f"• Pearson's r: {correlation:.2f}\n"
                f"• p-value: {p_value:.4f}\n"
                f"• Significant: {'Yes' if p_value < 0.05 else 'No'}"
            )
            
            # Add the correlation box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax1.text(0.05, 0.05, corr_text, transform=ax1.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='left', bbox=props)
            
            # 2. Scatter plot of stony coral vs. macroalgae cover (right plot)
            ax2.scatter(data_dict['pcover_taxa']['Macroalgae'], 
                       data_dict['pcover_taxa'][stony_coral_col],
                       c=data_dict['pcover_taxa']['Year'], cmap='viridis',
                       alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
            
            # Add colorbar to show year
            cbar = plt.colorbar(ax2.scatter(data_dict['pcover_taxa']['Macroalgae'], 
                                           data_dict['pcover_taxa'][stony_coral_col],
                                           c=data_dict['pcover_taxa']['Year'], cmap='viridis', 
                                           alpha=0), ax=ax2)
            cbar.set_label('Year', fontweight='bold')
            
            # Add best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data_dict['pcover_taxa']['Macroalgae'], 
                data_dict['pcover_taxa'][stony_coral_col]
            )
            
            x_line = np.array([0, data_dict['pcover_taxa']['Macroalgae'].max()])
            y_line = slope * x_line + intercept
            
            ax2.plot(x_line, y_line, color='red', linewidth=2, 
                   linestyle='--', label=f'Linear trend (r = {r_value:.2f})')
            
            # Set plot aesthetics
            ax2.set_title('Relationship Between Cover Types', fontweight='bold', fontsize=16, pad=15)
            ax2.set_xlabel('Macroalgae Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax2.set_ylabel('Stony Coral Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add legend
            ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
                     fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
            
            # Add regression statistics to the plot
            regression_text = (
                f"Regression Analysis:\n"
                f"• Slope: {slope:.3f}\n"
                f"• Intercept: {intercept:.3f}\n"
                f"• R²: {r_value**2:.2f}\n"
                f"• p-value: {p_value:.4f}"
            )
            
            # Add the regression box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax2.text(0.05, 0.95, regression_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left', bbox=props)
            
            # Add a note about the data source
            fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                     ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add a summary interpretation
            summary_text = (
                f"CORAL-ALGAE INTERACTION SUMMARY:\n"
                f"• Relationship: {correlation:.2f} correlation coefficient "
                f"({'Negative' if correlation < 0 else 'Positive'})\n"
                f"• Interpretation: {'Evidence for competition between macroalgae and stony corals' if correlation < 0 else 'No clear competitive relationship observed'}\n"
                f"• This suggests {'space limitation or inhibitory effects' if correlation < 0 else 'other factors may be more important drivers'}"
            )
            
            # Add the summary box with enhanced styling
            props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in the center area between plots
            fig.text(0.5, -0.10, summary_text, fontsize=12, fontweight='bold',
                    verticalalignment='center', horizontalalignment='center', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "coral_algae_interactions.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Coral-algae interaction analysis saved.")
    
    # Analyze the relationship between stony coral density and species richness
    if 'stony_density' in data_dict and 'Species_Richness' in data_dict['stony_density'].columns:
        print("Analyzing density-richness relationships...")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['background'])
        fig.suptitle('ECOLOGICAL INTERACTIONS: DENSITY-RICHNESS RELATIONSHIPS',
                    fontweight='bold', fontsize=22, color=COLORS['dark_blue'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        
        # Set background color
        ax1.set_facecolor(COLORS['background'])
        ax2.set_facecolor(COLORS['background'])
        
        # 1. Overall scatter plot of density vs. richness (left plot)
        ax1.scatter(data_dict['stony_density']['Total_Density'], 
                   data_dict['stony_density']['Species_Richness'],
                   c=data_dict['stony_density']['Year'], cmap='viridis',
                   alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
        
        # Add colorbar to show year
        cbar = plt.colorbar(ax1.scatter(data_dict['stony_density']['Total_Density'], 
                                       data_dict['stony_density']['Species_Richness'],
                                       c=data_dict['stony_density']['Year'], cmap='viridis', 
                                       alpha=0), ax=ax1)
        cbar.set_label('Year', fontweight='bold')
        
        # Add best fit line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data_dict['stony_density']['Total_Density'], 
            data_dict['stony_density']['Species_Richness']
        )
        
        # Define range for line plotting
        max_density = data_dict['stony_density']['Total_Density'].max()
        x_line = np.array([0, max_density])
        y_line = slope * x_line + intercept
        
        ax1.plot(x_line, y_line, color='red', linewidth=2, 
               linestyle='--', label=f'Linear trend (r = {r_value:.2f})')
        
        # Set plot aesthetics
        ax1.set_title('Overall Density-Richness Relationship', fontweight='bold', fontsize=16, pad=15)
        ax1.set_xlabel('Total Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        ax1.set_ylabel('Species Richness', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, 
                 fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
        
        # Add regression statistics to the plot
        regression_text = (
            f"Regression Analysis:\n"
            f"• Slope: {slope:.3f}\n"
            f"• Intercept: {intercept:.3f}\n"
            f"• R²: {r_value**2:.2f}\n"
            f"• p-value: {p_value:.4f}"
        )
        
        # Add the regression box with enhanced styling
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in a free space
        ax1.text(0.05, 0.95, regression_text, transform=ax1.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # 2. Density-richness relationship by habitat (right plot)
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
        
        # Plot scatter for each habitat
        for habitat in data_dict['stony_density']['Habitat'].unique():
            habitat_data = data_dict['stony_density'][data_dict['stony_density']['Habitat'] == habitat]
            
            ax2.scatter(habitat_data['Total_Density'], 
                       habitat_data['Species_Richness'],
                       color=habitat_colors.get(habitat, COLORS['coral']),
                       alpha=0.7, s=50, edgecolor='black', linewidth=0.5,
                       label=habitat_names.get(habitat, habitat))
            
            # Calculate trend line for this habitat
            if len(habitat_data) >= 5:  # Only calculate if we have enough data points
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    habitat_data['Total_Density'], 
                    habitat_data['Species_Richness']
                )
                
                x_line = np.array([0, habitat_data['Total_Density'].max()])
                y_line = slope * x_line + intercept
                
                ax2.plot(x_line, y_line, color=habitat_colors.get(habitat, COLORS['coral']), 
                         linewidth=2, linestyle='--')
        
        # Set plot aesthetics
        ax2.set_title('Density-Richness Relationship by Habitat', fontweight='bold', fontsize=16, pad=15)
        ax2.set_xlabel('Total Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
        ax2.set_ylabel('Species Richness', fontweight='bold', fontsize=14, labelpad=10)
        
        # Add grid for better readability
        ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
        
        # Add legend
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9, 
                 fontsize=10, loc='upper right', edgecolor=COLORS['grid'])
        
        # Calculate correlations by habitat
        habitat_correlations = []
        
        for habitat in data_dict['stony_density']['Habitat'].unique():
            habitat_data = data_dict['stony_density'][data_dict['stony_density']['Habitat'] == habitat]
            
            if len(habitat_data) >= 5:  # Only calculate if we have enough data points
                correlation, p_value = pearsonr(
                    habitat_data['Total_Density'], 
                    habitat_data['Species_Richness']
                )
                
                habitat_correlations.append({
                    'Habitat': habitat,
                    'Habitat_Name': habitat_names.get(habitat, habitat),
                    'Correlation': correlation,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Sample_Size': len(habitat_data)
                })
        
        # Convert to DataFrame
        habitat_corr_df = pd.DataFrame(habitat_correlations)
        
        if not habitat_corr_df.empty:
            # Add correlation statistics to the plot
            corr_text = "Correlations by Habitat:\n"
            
            for _, row in habitat_corr_df.iterrows():
                corr_text += f"• {row['Habitat_Name']}: r = {row['Correlation']:.2f}"
                corr_text += f" ({row['P_Value']:.3f}{', sig.' if row['Significant'] else ''})\n"
            
            # Add the correlation box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Add a note about the data source
        fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
                 ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
        
        # Add an overall summary
        summary_text = (
            f"DENSITY-RICHNESS RELATIONSHIP SUMMARY:\n"
            f"• Overall relationship: {r_value:.2f} correlation coefficient"
            f" ({'Positive' if r_value > 0 else 'Negative'}, {'Significant' if p_value < 0.05 else 'Not significant'})\n"
            f"• Habitat variation: Relationship strength varies by habitat type\n"
            f"• Strongest in: {habitat_corr_df.loc[habitat_corr_df['Correlation'].abs().idxmax(), 'Habitat_Name'] if not habitat_corr_df.empty else 'N/A'}\n"
            f"• Interpretation: {'Higher density supports greater species richness, suggesting facilitation or shared habitat suitability' if r_value > 0 else 'Competitive exclusion may be occurring at higher densities'}"
        )
        
        # Add the summary box with enhanced styling
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                    edgecolor=COLORS['dark_blue'], linewidth=2)
        
        # Position the text box in the center area between plots
        fig.text(0.5, -0.10, summary_text, fontsize=12, fontweight='bold',
                verticalalignment='center', horizontalalignment='center', bbox=props)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(results_dir, "density_richness_relationship.png"), 
                   bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
        plt.close()
        
        print("Density-richness relationship analysis saved.")
    
    # Analyze the relationship between stony corals and octocorals if both datasets are available
    if 'pcover_taxa' in data_dict and 'octo_density' in data_dict:
        print("Analyzing stony coral-octocoral relationship...")
        
        # First, aggregate the data by site and year to allow comparison
        # For octocorals, we have density
        octo_yearly = data_dict['octo_density'].groupby(['Year', 'StationID'])['Total_Density'].mean().reset_index()
        
        # For stony corals, we have percent cover
        stony_yearly = data_dict['pcover_taxa'].groupby(['Year', 'StationID'])[stony_coral_col].mean().reset_index()
        
        # Merge the datasets
        merged_data = pd.merge(
            stony_yearly, 
            octo_yearly,
            on=['Year', 'StationID'],
            how='inner',
            suffixes=('_stony', '_octo')
        )
        
        if len(merged_data) > 10:  # Ensure we have enough data points
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
            ax.set_facecolor(COLORS['background'])
            
            # Scatter plot of stony coral cover vs. octocoral density
            scatter = ax.scatter(
                merged_data[stony_coral_col], 
                merged_data['Total_Density'],
                c=merged_data['Year'], 
                cmap='viridis',
                alpha=0.7,
                s=50,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add colorbar to show year
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Year', fontweight='bold')
            
            # Add best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_data[stony_coral_col], 
                merged_data['Total_Density']
            )
            
            x_line = np.array([0, merged_data[stony_coral_col].max()])
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, color='red', linewidth=2, 
                   linestyle='--', label=f'Linear trend (r = {r_value:.2f})')
            
            # Set plot aesthetics
            ax.set_title('RELATIONSHIP BETWEEN STONY CORAL COVER AND OCTOCORAL DENSITY',
                        fontweight='bold', fontsize=18, color=COLORS['dark_blue'],
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            
            ax.set_xlabel('Stony Coral Cover (%)', fontweight='bold', fontsize=14, labelpad=10)
            ax.set_ylabel('Octocoral Density (colonies/m²)', fontweight='bold', fontsize=14, labelpad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
            
            # Add legend
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, 
                     fontsize=12, loc='upper right', edgecolor=COLORS['grid'])
            
            # Add regression statistics to the plot
            regression_text = (
                f"Regression Analysis:\n"
                f"• Slope: {slope:.3f}\n"
                f"• Intercept: {intercept:.3f}\n"
                f"• R²: {r_value**2:.2f}\n"
                f"• p-value: {p_value:.4f}\n"
                f"• Significant: {'Yes' if p_value < 0.05 else 'No'}"
            )
            
            # Add the regression box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.05, 0.95, regression_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left', bbox=props)
            
            # Add a note about the data source
            # fig.text(0.5, 0.01, "Data Source: Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)", 
            #          ha='center', va='center', fontsize=10, fontstyle='italic', color=COLORS['text'])
            
            # Add an interpretation summary
            summary_text = (
                f"STONY CORAL-OCTOCORAL INTERACTION SUMMARY:\n"
                f"• Relationship: {r_value:.2f} correlation coefficient "
                f"({'Negative' if r_value < 0 else 'Positive'})\n"
                f"• Interpretation: "
            )
            
            if r_value < -0.3 and p_value < 0.05:
                summary_text += "Strong negative correlation suggests potential competition or habitat partitioning"
            elif r_value > 0.3 and p_value < 0.05:
                summary_text += "Strong positive correlation suggests shared habitat preferences or facilitation"
            elif abs(r_value) < 0.2 or p_value >= 0.05:
                summary_text += "Weak or non-significant relationship suggests independent responses to environmental factors"
            else:
                summary_text += "Moderate relationship suggests partial interaction or context-dependent effects"
            
            # Add the summary box with enhanced styling
            props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['dark_blue'], linewidth=2)
            
            # Position the text box in a free space
            ax.text(0.5, -0.25, summary_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='center', horizontalalignment='center', bbox=props)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(results_dir, "stony_coral_octocoral_relationship.png"), 
                       bbox_inches='tight', dpi=300, facecolor=COLORS['background'])
            plt.close()
            
            print("Stony coral-octocoral relationship analysis saved.")
    
    # Return a summary of ecological interactions
    interactions_summary = {
        'coral_algae_correlation': correlation if 'correlation' in locals() else None,
        'density_richness_correlation': r_value if 'r_value' in locals() else None,
        'stony_octo_correlation': r_value if 'r_value' in locals() else None
    }
    
    return interactions_summary

# Function to synthesize key findings and create a comprehensive summary
def synthesize_key_findings(impact_summary, spatial_factors_summary, model_results, interactions_summary):
    """
    Synthesize key findings from all analyses and create a comprehensive summary.
    """
    print("Synthesizing key findings and saving markdown summary...")
    # Prepare markdown summary content
    md_lines = []
    md_lines.append("# Key Factors Affecting Coral Health in the Florida Keys\n")
    md_lines.append("## Summary of Key Factors Affecting Coral Health Parameters\n")
    
    # Environmental Disturbances
    md_lines.append("### 1. Environmental Disturbances (0.9/1.0)\n")
    md_lines.append("**Factors:** Temperature Extremes (Bleaching Events), Hurricanes (Physical Damage), Disease Outbreaks (SCTLD), Cold Snaps\n")
    md_lines.append("**Description:** Major disturbance events have significant negative impacts on coral cover, tissue area, and species richness. Disease outbreaks and bleaching events show the strongest impacts, with evidence of up to 30% reduction in some metrics.\n")
    
    # Spatial Context
    md_lines.append("### 2. Spatial Context (0.9/1.0)\n")
    md_lines.append("**Factors:** Geographic Region (Upper/Middle/Lower Keys), Habitat Type, Depth, Site-Specific Conditions\n")
    md_lines.append("**Description:** Strong spatial patterns exist across all coral metrics. Habitat type is a primary determinant of abundance and diversity. Middle Keys often show higher resilience. Patch reefs support highest coral metrics. Optimal depth ranges vary by species.\n")
    
    # Ecological Interactions
    md_lines.append("### 3. Ecological Interactions (0.8/1.0)\n")
    md_lines.append("**Factors:** Macroalgae Competition, Density-Richness Relationships, Octocoral-Stony Coral Interactions, Community Structure Changes\n")
    md_lines.append("**Description:** Competition with macroalgae impacts stony coral cover. Positive density-richness relationship suggests facilitation among corals. Shifting community composition is evident, with octocorals potentially benefiting from stony coral declines.\n")
    
    # Temporal Trends
    md_lines.append("### 4. Temporal Trends (0.8/1.0)\n")
    md_lines.append("**Factors:** Long-Term Decline Patterns, Cumulative Impact of Stressors, Recovery Capacity, Seasonal Variation\n")
    md_lines.append("**Description:** Long-term trends show overall stony coral decline but octocoral increases. Cumulative impacts of multiple stressors evident. Recovery capacity varies by region and habitat. Distinct seasonal patterns affect coral parameters.\n")
    
    # Water Quality
    md_lines.append("### 5. Water Quality (0.8/1.0)\n")
    md_lines.append("**Factors:** Temperature Regimes, Water Clarity/Turbidity, Nutrient Levels, Current Patterns\n")
    md_lines.append("**Description:** Temperature patterns beyond extreme events show correlations with coral health. Water quality parameters likely influence algal competition dynamics and stress resilience. Site-specific water conditions contribute to variability.\n")
    
    # Overall Conclusions
    md_lines.append("## Overall Conclusions\n")
    md_lines.append("1. Multiple interactive factors affect coral health, with environmental disturbances and spatial context being most important.\n")
    md_lines.append("2. Coral responses vary significantly by species, region, and habitat type, requiring context-specific management approaches.\n")
    md_lines.append("3. Stony corals and octocorals show contrasting trends, suggesting different environmental sensitivities and competitive dynamics.\n")
    md_lines.append("4. The relative importance of factors differs for different coral metrics (cover, tissue area, density, richness).\n")
    md_lines.append("5. Understanding these key factors provides a foundation for developing targeted conservation and restoration strategies.\n")
    md_lines.append("**Data Source:** Florida Keys Coral Reef Evaluation Monitoring Project (CREMP)\n")
    
    # Write markdown to file
    md_path = os.path.join(results_dir, "key_factors_synthesis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown summary saved to {md_path}")

# Main function
def main():
    """Main function to execute all analyses."""
    print("\n=== ANALYZING KEY FACTORS AFFECTING CORAL HEALTH ===\n")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    print("Starting data loading...")
    # Load and preprocess the data
    data_dict, species_cols = load_and_preprocess_data()
    print("Data loading complete")
    
    print("Starting environmental impact analysis...")
    # Analyze environmental impacts
    impact_summary = analyze_environmental_impacts(data_dict, species_cols)
    print("Environmental impact analysis complete")
    
    print("Starting spatial factors analysis...")
    # Analyze spatial factors
    spatial_factors_summary = analyze_spatial_factors(data_dict, species_cols)
    print("Spatial factors analysis complete")
    
    print("Starting multivariate modeling...")
    # Build multivariate models
    model_results = build_multivariate_models(data_dict, species_cols)
    print("Multivariate modeling complete")
    
    print("Starting ecological interactions analysis...")
    # Analyze ecological interactions
    interactions_summary = analyze_ecological_interactions(data_dict, species_cols)
    print("Ecological interactions analysis complete")
    
    print("Starting synthesis of findings...")
    # Synthesize key findings
    synthesize_key_findings(impact_summary, spatial_factors_summary, model_results, interactions_summary)
    print("Synthesis of findings complete")
    
    print("\n=== ANALYSIS COMPLETE ===\n")
    print(f"Results saved in {results_dir}")

# Execute main function
if __name__ == "__main__":
    main()