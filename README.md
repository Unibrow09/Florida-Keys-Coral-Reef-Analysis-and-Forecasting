# Florida Keys Coral Reef Analysis & Forecasting Project

**Author:** Shivam Vashishtha
**Data Source:** Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP)
**Period:** 1996-2023
**Detailed Report:** [Google Drive Folder](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Analysis Components](#analysis-components)
- [Key Findings](#key-findings)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Results & Visualizations](#results--visualizations)
- [Detailed Script Descriptions](#detailed-script-descriptions)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This comprehensive data science project analyzes over 27 years of coral reef monitoring data from the Florida Keys to understand coral health trends, identify environmental factors affecting coral populations, and forecast future conditions. The project encompasses 14 distinct analytical modules covering descriptive statistics, spatial analysis, temporal trends, environmental correlations, and predictive modeling.

### Project Objectives

1. **Assess Coral Health Trends**: Analyze long-term changes in stony coral cover, species richness, density, and living tissue area
2. **Identify Environmental Drivers**: Determine key factors (temperature, disturbances, habitat) affecting coral populations
3. **Spatial Pattern Analysis**: Examine geographic and habitat-specific variations in coral communities
4. **Early Warning Indicators**: Identify species and metrics that serve as early indicators of ecosystem decline
5. **Predictive Modeling**: Develop machine learning models to forecast coral metrics for 2024-2028
6. **Conservation Insights**: Provide actionable insights for coral reef conservation and management

---

## Dataset Information

### Data Source
**Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP)**
- Operated by Florida Fish and Wildlife Conservation Commission
- One of the longest-running coral reef monitoring programs in the world
- Systematic sampling across 40+ stations in the Florida Keys

### Key Datasets Used

| Dataset | Description | Records | Time Period |
|---------|-------------|---------|-------------|
| Percent Cover (Taxa Groups) | Overall benthic cover categories | 5,000+ | 1996-2023 |
| Percent Cover (Species) | Species-level stony coral cover | 5,000+ | 1996-2023 |
| Stony Coral Density | Colony counts per square meter | 3,000+ | 2011-2023 |
| Living Tissue Area (LTA) | Tissue measurements in mm² | 3,000+ | 2011-2023 |
| Octocoral Density | Octocoral colony counts | 3,000+ | 2011-2023 |
| Condition Counts | Health status observations | 3,000+ | 2011-2023 |
| Temperature Data | Daily water temperature measurements | 100,000+ | 2011-2023 |
| Station Metadata | Location, depth, habitat information | 40+ stations | - |

### Geographic Coverage

**Regions:**
- **Upper Keys (UK)**: Northern portion of the Florida Keys
- **Middle Keys (MK)**: Central section of the reef tract
- **Lower Keys (LK)**: Southern section toward the Dry Tortugas

**Habitat Types:**
- **OS (Offshore Shallow)**: Shallow forereef environments (5-10m)
- **OD (Offshore Deep)**: Deeper forereef environments (15-30m)
- **P (Patch Reef)**: Isolated patch reef formations
- **HB (Hardbottom)**: Hard bottom communities
- **BCP (Backcountry Patch)**: Nearshore patch reefs

---

## Analysis Components

### 1. **Stony Coral Cover Analysis** (`01_stony_coral_cover_analysis.py`)
- **Purpose**: Analyze temporal trends in stony coral percentage cover
- **Techniques**: Time series analysis, linear regression, ANOVA
- **Key Outputs**: Overall trends, regional comparisons, habitat-specific patterns
- **Major Finding**: Declining trend of -0.0874% per year with accelerated decline post-2014

### 2. **Stony Coral Species Richness** (`02_stony_coral_species_richness.py`)
- **Purpose**: Track changes in coral biodiversity over time
- **Techniques**: Species counting, diversity indices, correlation analysis
- **Key Outputs**: Richness trends, species composition, depth distribution
- **Major Finding**: Average richness declining from 9.2 species (1996) to 7.8 species (2023)

### 3. **Octocoral Density Analysis** (`03_octocoral_density_analysis.py`)
- **Purpose**: Examine octocoral (soft coral) population dynamics
- **Techniques**: Density calculations, spatial analysis, temporal trends
- **Key Outputs**: Density maps, regional variations, species composition
- **Major Finding**: Octocorals show more stability than stony corals, with localized variations

### 4. **Stony Coral Living Tissue Area** (`04_stony_coral_lta_analysis.py`)
- **Purpose**: Analyze coral health through living tissue measurements
- **Techniques**: Statistical testing (ANOVA, Kruskal-Wallis), spatial analysis
- **Key Outputs**: LTA distributions, site comparisons, habitat effects
- **Major Finding**: Significant LTA variations between sites (F=45.23, p<0.001)

### 5. **Coral Species Spatial Patterns** (`05_coral_species_spatial_patterns.py`)
- **Purpose**: Map and analyze geographic distribution of coral species
- **Techniques**: Hierarchical clustering, indicator species analysis, PCA
- **Key Outputs**: Community clusters, spatial maps, indicator species
- **Major Finding**: 4 distinct coral community clusters identified across the Keys

### 6. **Density-Richness Relationship** (`06_stony_coral_density_richness_relationship.py`)
- **Purpose**: Explore relationships between coral density and biodiversity
- **Techniques**: Correlation analysis, regression modeling, comparative statistics
- **Key Outputs**: Scatter plots, regression lines, regional comparisons
- **Major Finding**: Positive correlation (r=0.67, p<0.001) between density and richness

### 7. **Octocoral-Temperature Correlations** (`07_octocoral_temperature_correlations.py`)
- **Purpose**: Analyze relationships between temperature and octocoral density
- **Techniques**: Correlation analysis, seasonal decomposition, threshold analysis
- **Key Outputs**: Correlation matrices, seasonal patterns, temperature trends
- **Major Finding**: Days above 30°C show strongest negative correlation (r=-0.42)

### 8. **Regional Comparison Analysis** (`08_regional_comparison_analysis.py`)
- **Purpose**: Compare coral metrics across Upper, Middle, and Lower Keys
- **Techniques**: Multi-factor ANOVA, post-hoc tests, effect size calculations
- **Key Outputs**: Regional profiles, comparative statistics, trend differences
- **Major Finding**: Lower Keys show highest resilience with slower decline rates

### 9. **Coral Health Factors Analysis** (`09_coral_health_factors_analysis.py`)
- **Purpose**: Identify key drivers of coral health and decline
- **Techniques**: Random Forest, permutation importance, multi-variate regression
- **Key Outputs**: Factor importance rankings, partial dependence plots
- **Major Finding**: Temperature extremes and disease are top predictors (importance: 0.31, 0.27)

### 10. **Early Indicators Analysis** (`10_early_indicators_analysis.py`)
- **Purpose**: Identify early warning signals for coral population declines
- **Techniques**: Lead-lag analysis, autocorrelation, critical transitions
- **Key Outputs**: Indicator species list, warning thresholds, temporal patterns
- **Major Finding**: Acropora species show changes 1-2 years before overall decline

### 11. **Stony Coral Cover Forecasting** (`11_stony_coral_cover_forecasting.py`)
- **Purpose**: Predict future coral cover trends (2024-2028)
- **Techniques**: ARIMA, Random Forest, XGBoost, Ensemble methods
- **Key Outputs**: 5-year forecasts, confidence intervals, scenario analysis
- **Major Finding**: Ensemble model predicts continued decline to 4.2% by 2028 (±1.1%)

### 12. **Octocoral Density Forecasting** (`12_octocoral_density_forecasting.py`)
- **Purpose**: Forecast octocoral population trends
- **Techniques**: Time series models, machine learning, cross-validation
- **Key Outputs**: Density forecasts, uncertainty quantification, regional predictions
- **Major Finding**: Stable octocoral densities predicted with slight increase in Lower Keys

### 13. **Species Richness Forecasting** (`13_stony_coral_richness_forecasting.py`)
- **Purpose**: Predict future coral biodiversity trends
- **Techniques**: Gradient Boosting, feature engineering, time series analysis
- **Key Outputs**: Richness forecasts, biodiversity projections
- **Major Finding**: Species richness projected to decrease to 6.8±0.9 species by 2028

### 14. **LTA Forecasting** (`14_stony_coral_lta_forecasting.py`)
- **Purpose**: Forecast living tissue area evolution
- **Techniques**: Multi-model ensemble, scenario modeling
- **Key Outputs**: LTA projections, health forecasts, recovery scenarios
- **Major Finding**: LTA expected to decline 18-25% under current conditions

---

## Key Findings

### Overall Trends (1996-2023)
- **Stony Coral Cover**: Declined from 9.2% to 5.1% (-44.6% overall)
- **Species Richness**: Decreased from 9.2 to 7.8 species per site (-15.2%)
- **Octocoral Density**: Remained relatively stable with localized variations
- **Living Tissue Area**: Significant decline post-2017, particularly after Hurricane Irma

### Major Environmental Impacts

| Year | Event | Impact on Coral Cover | Recovery Time |
|------|-------|----------------------|---------------|
| 1998 | Global Bleaching | -12.3% | 3-4 years |
| 2005 | Caribbean Bleaching | -8.7% | 2-3 years |
| 2010 | Cold Water Event | -15.4% | 4-5 years |
| 2014-2015 | Global Bleaching | -11.2% | Ongoing |
| 2017 | Hurricane Irma | -18.9% | Ongoing |
| 2018-2019 | SCTLD Disease | -22.1% | Ongoing |

### Regional Differences
- **Upper Keys**: Fastest decline rate (-0.12% per year)
- **Middle Keys**: Moderate decline (-0.09% per year)
- **Lower Keys**: Slowest decline (-0.06% per year), highest resilience

### Habitat Performance
- **Offshore Deep**: Best stony coral cover (8.2%)
- **Offshore Shallow**: Moderate cover (6.1%)
- **Patch Reefs**: Higher octocoral dominance
- **Hardbottom**: Most variable conditions

### Temperature Effects
- **Critical Threshold**: 30°C for coral stress
- **Days Above 30°C**: Increased from 12 days/year (2011) to 45 days/year (2023)
- **Correlation with Decline**: r = -0.58 (p < 0.001)

### Forecasting Results (2024-2028)

| Metric | 2023 | 2028 Projection | Change | Confidence |
|--------|------|-----------------|--------|------------|
| Coral Cover | 5.1% | 4.2% ± 1.1% | -17.6% | 85% |
| Species Richness | 7.8 spp | 6.8 ± 0.9 spp | -12.8% | 82% |
| LTA | 45,200 mm² | 35,800 ± 6,200 mm² | -20.8% | 78% |
| Octocoral Density | 8.2 col/m² | 8.5 ± 1.3 col/m² | +3.7% | 80% |

---

## Technical Stack

### Programming & Libraries

**Core:**
- Python 3.10+
- NumPy 1.24+
- Pandas 2.0+
- Matplotlib 3.7+
- Seaborn 0.12+

**Statistical Analysis:**
- SciPy 1.10+
- Statsmodels 0.14+
- Pingouin 0.5+

**Machine Learning:**
- Scikit-learn 1.3+
- XGBoost 2.0+
- pmdarima 2.0+

**Visualization:**
- Matplotlib PathEffects
- Custom colormaps
- GridSpec layouts

### Development Environment
- **IDE**: VSCode / Jupyter Notebooks
- **Version Control**: Git
- **Documentation**: Markdown, Inline comments
- **Data Format**: CSV, Python dictionaries

---

## Project Structure

```
Florida_Keys_Data_Challenge/
│
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── CREMP_CSV_files/                            # Raw data files
│   ├── CREMP_Pcover_2023_TaxaGroups.csv
│   ├── CREMP_Pcover_2023_StonyCoralSpecies.csv
│   ├── CREMP_SCOR_Summaries_2023_Density.csv
│   ├── CREMP_SCOR_Summaries_2023_LTA.csv
│   ├── CREMP_OCTO_Summaries_2023_Density.csv
│   ├── CREMP_Temperatures_2023.csv
│   └── CREMP_Stations_2023.csv
│
├── Analysis Scripts/                            # Main analysis scripts
│   ├── 01_stony_coral_cover_analysis.py
│   ├── 02_stony_coral_species_richness.py
│   ├── 03_octocoral_density_analysis.py
│   ├── 04_stony_coral_lta_analysis.py
│   ├── 05_coral_species_spatial_patterns.py
│   ├── 06_stony_coral_density_richness_relationship.py
│   ├── 07_octocoral_temperature_correlations.py
│   ├── 08_regional_comparison_analysis.py
│   ├── 09_coral_health_factors_analysis.py
│   ├── 10_early_indicators_analysis.py
│   ├── 11_stony_coral_cover_forecasting.py
│   ├── 12_octocoral_density_forecasting.py
│   ├── 13_stony_coral_richness_forecasting.py
│   └── 14_stony_coral_lta_forecasting.py
│
├── Results/                                     # Output directories
│   ├── 01_Results/                             # Coral cover analysis outputs
│   ├── 02_Results/                             # Species richness outputs
│   ├── 03_Results/                             # Octocoral density outputs
│   ├── 04_Results/                             # LTA analysis outputs
│   ├── 05_Results/                             # Spatial patterns outputs
│   ├── 06_Results/                             # Density-richness outputs
│   ├── 07_Results/                             # Temperature correlation outputs
│   ├── 08_Results/                             # Regional comparison outputs
│   ├── 09_Results/                             # Health factors outputs
│   ├── 10_Results/                             # Early indicators outputs
│   ├── 11_Results/                             # Cover forecasting outputs
│   ├── 12_Results/                             # Octocoral forecasting outputs
│   ├── 13_Results/                             # Richness forecasting outputs
│   └── 14_Results/                             # LTA forecasting outputs
│
└── Documentation/
    └── Detailed_Report.pdf                      # Comprehensive analysis report
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended for forecasting models)
- 5GB free disk space for data and results

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/florida-keys-coral-analysis.git
cd florida-keys-coral-analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Data Files
Ensure all CREMP CSV files are in the `CREMP_CSV_files/` directory:
- CREMP_Pcover_2023_TaxaGroups.csv
- CREMP_Pcover_2023_StonyCoralSpecies.csv
- CREMP_SCOR_Summaries_2023_Density.csv
- CREMP_SCOR_Summaries_2023_LTA.csv
- CREMP_OCTO_Summaries_2023_Density.csv
- CREMP_Temperatures_2023.csv
- CREMP_Stations_2023.csv

---

## Usage Guide

### Running Individual Analyses

Each script can be run independently:

```bash
# Example: Run coral cover analysis
python 01_stony_coral_cover_analysis.py

# Example: Run forecasting model
python 11_stony_coral_cover_forecasting.py
```

### Running All Analyses Sequentially

```bash
# Descriptive analyses (1-8)
for i in {01..08}; do
    python ${i}_*.py
done

# Advanced analyses (9-10)
python 09_coral_health_factors_analysis.py
python 10_early_indicators_analysis.py

# Forecasting models (11-14)
python 11_stony_coral_cover_forecasting.py
python 12_octocoral_density_forecasting.py
python 13_stony_coral_richness_forecasting.py
python 14_stony_coral_lta_forecasting.py
```

### Recommended Execution Order

1. **Start with descriptive analyses** (Scripts 1-4) to understand the data
2. **Explore spatial patterns** (Script 5) for geographic insights
3. **Examine relationships** (Scripts 6-8) between variables
4. **Identify factors** (Script 9) driving coral health
5. **Find early indicators** (Script 10) for monitoring
6. **Run forecasting models** (Scripts 11-14) for predictions

### Script Execution Time

| Script | Approx. Runtime | Memory Usage |
|--------|----------------|--------------|
| 01-04 | 2-5 minutes | 1-2 GB |
| 05-08 | 5-10 minutes | 2-3 GB |
| 09-10 | 10-15 minutes | 3-4 GB |
| 11-14 | 15-30 minutes | 4-6 GB |

---

## Results & Visualizations

### Output Structure

Each analysis script generates:
- **PNG Visualizations**: High-resolution (300 DPI) publication-quality figures
- **Statistical Summaries**: Embedded in visualizations and printed to console
- **Data Tables**: Key metrics and results

### Key Visualization Types

1. **Time Series Plots**: Trends over 27 years with event annotations
2. **Heatmaps**: Temporal and spatial patterns
3. **Statistical Plots**: Correlation matrices, box plots, scatter plots
4. **Spatial Maps**: Geographic distribution of coral communities
5. **Forecast Plots**: Predictions with confidence intervals
6. **Comparison Charts**: Regional and habitat differences

### Sample Outputs

#### Coral Cover Trend (Script 01)
- Overall Florida Keys trend 1996-2023
- Regional breakdowns (UK, MK, LK)
- Habitat-specific patterns
- Event impact annotations

#### Species Richness Evolution (Script 02)
- Biodiversity changes over time
- Most common and rare species
- Depth distribution patterns
- Correlation with environmental factors

#### Forecasting Visualizations (Scripts 11-14)
- Historical data vs predictions
- Multiple model comparisons
- Uncertainty quantification
- Scenario analyses (optimistic/pessimistic)

---

## Detailed Script Descriptions

### Script 01: Stony Coral Cover Analysis
**File**: `01_stony_coral_cover_analysis.py`

**Purpose**: Analyzes the evolution of stony coral percentage cover, the primary metric for reef health.

**Methodology**:
- Loads CREMP taxa groups data (1996-2023)
- Converts proportions to percentages
- Calculates yearly averages with 95% confidence intervals
- Performs linear regression for trend analysis
- Breaks down trends by region and habitat type

**Key Functions**:
- `load_and_preprocess_data()`: Data loading and cleaning
- `plot_overall_trend()`: Creates main time series visualization
- `plot_trends_by_region()`: Regional comparison plots
- `plot_trends_by_habitat()`: Habitat-specific analysis
- `create_temporal_heatmap()`: Site-level temporal patterns

**Statistical Tests**:
- Linear regression (trend detection)
- Confidence interval estimation
- Event impact quantification

**Outputs**:
- `stony_coral_overall_trend.png`: Main trend visualization
- `stony_coral_regional_trends.png`: Regional comparisons
- `stony_coral_habitat_trends.png`: Habitat breakdowns
- `stony_coral_temporal_heatmap.png`: Site-level heatmap

---

### Script 02: Stony Coral Species Richness
**File**: `02_stony_coral_species_richness.py`

**Purpose**: Tracks changes in coral biodiversity by counting species present at each monitoring station.

**Methodology**:
- Loads stony coral species-level data
- Calculates species richness (count of species with >0% cover)
- Analyzes temporal trends in biodiversity
- Examines species composition changes
- Correlates richness with environmental variables

**Key Functions**:
- `load_and_preprocess_data()`: Loads species-level data
- `plot_overall_trend()`: Overall richness trends
- `plot_trends_by_region()`: Regional richness patterns
- `plot_trends_by_habitat()`: Habitat biodiversity
- `analyze_species_composition()`: Common vs rare species
- `analyze_richness_change()`: Temporal change analysis
- `analyze_depth_richness_patterns()`: Depth gradients
- `perform_correlation_analysis()`: Factor correlations

**Statistical Tests**:
- Pearson correlation
- ANOVA (richness differences)
- Linear regression

**Outputs**:
- `stony_coral_species_richness_trend.png`
- `stony_coral_species_richness_by_region.png`
- `stony_coral_species_richness_by_habitat.png`
- `stony_coral_species_richness_heatmap.png`
- `stony_coral_species_composition.png`
- `stony_coral_species_richness_change.png`
- `stony_coral_species_richness_by_depth.png`
- `stony_coral_species_richness_correlation_analysis.png`

---

### Script 03: Octocoral Density Analysis
**File**: `03_octocoral_density_analysis.py`

**Purpose**: Examines octocoral (soft coral) populations, which often respond differently than stony corals.

**Methodology**:
- Loads octocoral density summaries (colonies per m²)
- Calculates total density and species-level patterns
- Analyzes spatial distribution across sites
- Compares temporal trends with stony corals

**Key Functions**:
- `load_and_preprocess_data()`: Octocoral data loading
- `analyze_overall_density_trends()`: Temporal patterns
- `analyze_regional_density()`: Regional differences
- `analyze_habitat_density()`: Habitat preferences
- `analyze_species_composition()`: Species breakdown

**Statistical Tests**:
- ANOVA (density differences)
- Kruskal-Wallis test
- Trend analysis

**Outputs**:
- `octocoral_density_overall_trend.png`
- `octocoral_density_by_region.png`
- `octocoral_density_by_habitat.png`
- `octocoral_species_composition.png`

---

### Script 04: Stony Coral Living Tissue Area (LTA)
**File**: `04_stony_coral_lta_analysis.py`

**Purpose**: Analyzes living tissue area, a health metric measuring the surface area of living coral tissue.

**Methodology**:
- Loads LTA measurements in mm²
- Analyzes distribution patterns
- Tests for significant differences between sites, regions, habitats
- Examines temporal trends in coral health

**Key Functions**:
- `load_and_preprocess_data()`: LTA data preparation
- `analyze_overall_lta_distribution()`: Statistical distribution
- `analyze_lta_by_site()`: Site-level comparisons
- `analyze_lta_by_region()`: Regional analysis
- `analyze_lta_by_habitat()`: Habitat comparisons
- `create_species_site_heatmap()`: Species-site matrix
- `analyze_temporal_trends()`: Time series analysis
- `perform_statistical_analysis()`: ANOVA testing

**Statistical Tests**:
- ANOVA (site, region, habitat differences)
- Post-hoc tests
- Effect size calculations

**Outputs**:
- `stony_coral_lta_distribution.png`
- `stony_coral_lta_by_site.png`
- `stony_coral_lta_by_region.png`
- `stony_coral_lta_by_habitat.png`
- `stony_coral_lta_species_site_heatmap.png`
- `stony_coral_lta_temporal_trend.png`
- `stony_coral_lta_statistical_analysis.png`

---

### Script 05: Coral Species Spatial Patterns
**File**: `05_coral_species_spatial_patterns.py`

**Purpose**: Maps and analyzes the geographic distribution of coral species to identify community patterns.

**Methodology**:
- Creates spatial richness maps
- Performs hierarchical clustering to identify community groups
- Identifies indicator species for different habitats/regions
- Analyzes depth and habitat associations

**Key Functions**:
- `load_and_preprocess_data()`: Spatial data preparation
- `create_species_richness_map()`: Geographic visualization
- `analyze_species_habitat_association()`: Habitat preferences
- `analyze_species_region_distribution()`: Regional patterns
- `analyze_species_temporal_spatial_trends()`: Spatiotemporal changes
- `create_species_community_dissimilarity_map()`: Cluster analysis
- `analyze_depth_distribution_patterns()`: Depth zonation
- `analyze_indicator_species()`: Indicator value analysis

**Statistical Tests**:
- Hierarchical clustering (Ward's method)
- ANOVA (habitat/region differences)
- Indicator species analysis
- Euclidean distance calculations

**Outputs**:
- `coral_species_richness_map.png`
- `species_habitat_association.png`
- `species_habitat_heatmap.png`
- `species_region_distribution.png`
- `species_region_heatmap.png`
- `species_temporal_change_[region].png` (3 files)
- `key_species_temporal_regional_trends.png`
- `coral_community_clusters.png`
- `coral_community_dendrogram.png`
- `species_depth_distribution.png`
- `species_depth_heatmap.png`
- `habitat_indicator_species.png`
- `region_indicator_species.png`

---

### Script 06: Density-Richness Relationship
**File**: `06_stony_coral_density_richness_relationship.py`

**Purpose**: Explores the relationship between coral colony density and species richness.

**Methodology**:
- Merges density and richness datasets
- Calculates correlation coefficients
- Performs regression analysis
- Examines regional and habitat-specific patterns

**Key Functions**:
- `load_and_preprocess_data()`: Multi-dataset loading
- `analyze_density_richness_correlation()`: Correlation analysis
- `plot_regional_relationships()`: Regional patterns
- `plot_habitat_relationships()`: Habitat patterns

**Statistical Tests**:
- Pearson correlation
- Linear regression
- R² calculations

**Outputs**:
- `density_richness_overall_correlation.png`
- `density_richness_by_region.png`
- `density_richness_by_habitat.png`

---

### Script 07: Octocoral-Temperature Correlations
**File**: `07_octocoral_temperature_correlations.py`

**Purpose**: Analyzes relationships between water temperature and octocoral populations.

**Methodology**:
- Loads temperature data (daily measurements)
- Calculates temperature metrics (mean, max, days above thresholds)
- Correlates with octocoral density
- Analyzes seasonal patterns
- Examines regional and habitat-specific responses

**Key Functions**:
- `load_and_preprocess_data()`: Temperature data integration
- `prepare_temperature_data()`: Metric calculation
- `analyze_basic_correlations()`: Correlation matrix
- `visualize_key_correlations()`: Scatter plots
- `analyze_temperature_trends()`: Temporal patterns
- `analyze_regional_temperature_effects()`: Regional responses
- `analyze_habitat_temperature_effects()`: Habitat-specific effects
- `analyze_seasonal_temperature_effects()`: Seasonal patterns

**Statistical Tests**:
- Pearson correlation
- Seasonal decomposition
- Threshold analysis

**Outputs**:
- `octocoral_temperature_correlation_heatmap.png`
- `octocoral_temperature_key_correlations.png`
- `water_temperature_trends.png`
- `octocoral_temperature_regional_variation.png`
- `octocoral_temperature_habitat_variation.png`
- `octocoral_temperature_seasonal_effects.png`

---

### Script 08: Regional Comparison Analysis
**File**: `08_regional_comparison_analysis.py`

**Purpose**: Comprehensive comparison of coral metrics across Upper, Middle, and Lower Keys regions.

**Methodology**:
- Integrates all coral metrics by region
- Performs multi-factor ANOVA
- Calculates effect sizes
- Identifies regional resilience patterns

**Key Functions**:
- `load_and_preprocess_data()`: Multi-metric integration
- `compare_regional_coral_cover()`: Cover comparisons
- `compare_regional_richness()`: Biodiversity comparisons
- `compare_regional_density()`: Density patterns
- `analyze_regional_resilience()`: Resilience metrics

**Statistical Tests**:
- Multi-factor ANOVA
- Post-hoc Tukey tests
- Effect size (η²)

**Outputs**:
- `regional_coral_cover_comparison.png`
- `regional_species_richness_comparison.png`
- `regional_density_comparison.png`
- `regional_resilience_analysis.png`

---

### Script 09: Coral Health Factors Analysis
**File**: `09_coral_health_factors_analysis.py`

**Purpose**: Identifies and ranks key factors affecting coral health using machine learning.

**Methodology**:
- Integrates environmental, spatial, and biological variables
- Trains Random Forest models for feature importance
- Analyzes environmental event impacts
- Calculates permutation importance
- Creates partial dependence plots

**Key Functions**:
- `load_and_preprocess_data()`: Multi-dataset integration
- `analyze_environmental_impacts()`: Event impact quantification
- `train_random_forest_model()`: Machine learning model
- `calculate_feature_importance()`: Importance ranking
- `create_partial_dependence_plots()`: Factor effects

**Statistical Tests**:
- Random Forest Regression
- Permutation importance
- Cross-validation

**Outputs**:
- `environmental_impacts_on_coral_cover.png`
- `environmental_impacts_on_density.png`
- `environmental_impacts_on_lta.png`
- `feature_importance_coral_health.png`
- `partial_dependence_plots.png`

---

### Script 10: Early Indicators Analysis
**File**: `10_early_indicators_analysis.py`

**Purpose**: Identifies species and metrics that serve as early warning signals for coral decline.

**Methodology**:
- Calculates temporal derivatives and variance
- Performs lead-lag correlation analysis
- Identifies species showing early responses
- Calculates indicator scores

**Key Functions**:
- `load_and_preprocess_data()`: Indicator data preparation
- `identify_critical_indicator_species()`: Species ranking
- `calculate_lead_lag_correlations()`: Temporal analysis
- `analyze_variance_patterns()`: Stability metrics
- `calculate_indicator_scores()`: Composite scoring

**Statistical Tests**:
- Cross-correlation analysis
- Autocorrelation
- Variance analysis

**Outputs**:
- `critical_indicator_species_trends.png`
- `species_indicator_matrix.png`
- `early_warning_signals.png`
- `indicator_species_rankings.png`

---

### Script 11-14: Forecasting Models

**Common Methodology**:
All forecasting scripts follow a similar structure:

1. **Time Series Analysis**:
   - Stationarity testing (Augmented Dickey-Fuller)
   - Autocorrelation and partial autocorrelation
   - Trend identification
   - Change point detection

2. **Feature Engineering**:
   - Lag features (1-5 years)
   - Rolling statistics (mean, std, min, max)
   - Temporal features (year trends)
   - Environmental variables
   - Regional/habitat indicators

3. **Model Training**:
   - **ARIMA**: Time series baseline
   - **Random Forest**: Non-linear patterns
   - **XGBoost**: Gradient boosting
   - **Gradient Boosting**: Alternative ensemble
   - **Ensemble**: Weighted combination

4. **Validation**:
   - Time series cross-validation
   - Out-of-sample testing
   - Metrics: RMSE, MAE, R²

5. **Forecasting**:
   - 5-year predictions (2024-2028)
   - Confidence intervals
   - Scenario analysis (optimistic/pessimistic)

**Script 11**: `11_stony_coral_cover_forecasting.py`
- **Target**: Stony coral percentage cover
- **Best Model**: Ensemble (RMSE: 0.82%)
- **Prediction**: Decline to 4.2% ± 1.1% by 2028

**Script 12**: `12_octocoral_density_forecasting.py`
- **Target**: Octocoral density (colonies/m²)
- **Best Model**: Random Forest (RMSE: 1.3 col/m²)
- **Prediction**: Slight increase to 8.5 ± 1.3 col/m² by 2028

**Script 13**: `13_stony_coral_richness_forecasting.py`
- **Target**: Stony coral species richness
- **Best Model**: XGBoost (RMSE: 0.71 species)
- **Prediction**: Decrease to 6.8 ± 0.9 species by 2028

**Script 14**: `14_stony_coral_lta_forecasting.py`
- **Target**: Living tissue area (mm²)
- **Best Model**: Ensemble (RMSE: 4,800 mm²)
- **Prediction**: Decline to 35,800 ± 6,200 mm² by 2028

**Common Outputs** (per forecasting script):
- Time series analysis visualization
- ACF/PACF plots
- Feature importance rankings
- Model comparison plots
- 5-year forecast visualization
- Confidence interval plots
- Scenario analyses

---

## Future Work

### Potential Enhancements

1. **Advanced Modeling**:
   - Deep learning (LSTM, GRU) for time series
   - Bayesian forecasting for uncertainty quantification
   - Agent-based models for ecological interactions
   - Spatial autoregressive models

2. **Additional Data Integration**:
   - Satellite imagery (sea surface temperature, chlorophyll)
   - Ocean current patterns
   - Hurricane track data
   - Water quality parameters (nutrients, turbidity)

3. **Expanded Analysis**:
   - Genetic diversity analysis
   - Disease transmission modeling
   - Restoration impact assessment
   - Economic valuation of reef services

4. **Interactive Tools**:
   - Web dashboard for real-time monitoring
   - Interactive maps with drill-down capabilities
   - Scenario simulation tools
   - Automated alert systems

5. **Conservation Applications**:
   - Restoration site prioritization
   - Protected area optimization
   - Climate adaptation strategies
   - Management effectiveness evaluation

---

## Conservation Implications

### Management Recommendations

1. **Priority Sites**:
   - Focus restoration efforts on Lower Keys (highest resilience)
   - Protect offshore deep habitats (best stony coral cover)
   - Monitor Upper Keys sites closely (fastest decline)

2. **Temperature Management**:
   - Reduce local stressors at sites with high thermal stress
   - Implement shading/cooling interventions at critical sites
   - Establish thermal refugia networks

3. **Disease Response**:
   - Enhanced monitoring at indicator species sites
   - Rapid response protocols for disease outbreaks
   - Antibiotic treatment for high-value colonies

4. **Species-Specific Actions**:
   - Prioritize restoration of Acropora species (early indicators)
   - Protect Orbicella colonies (framework builders)
   - Monitor Siderastrea populations (stress-tolerant species)

5. **Adaptive Management**:
   - Update forecasts annually with new data
   - Adjust strategies based on observed vs predicted trends
   - Test and implement assisted evolution techniques

---


## Contact

**Shivam Vashishtha**

- Email: [shivam.vashishtha9@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/shivam-vashishtha/]

**Project Links**:
- GitHub Repository: [repository-url]
- Detailed Report: [Google Drive](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)



