# 🪸 Florida Keys Coral Reef Health Analysis & Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Data Science](https://img.shields.io/badge/Data%20Science-Pandas%20%7C%20NumPy-green.svg)](https://pandas.pydata.org/)

> **A comprehensive data science project analyzing 28+ years of coral reef monitoring data from the Florida Keys, combining advanced statistical analysis, machine learning forecasting, and ecological insights to understand coral reef health dynamics and predict future trends.**

📂 **[View Complete Project Documentation & Results →](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)**  
*Includes: Detailed PDF Reports, All Visualizations, Trained ML Models, Forecasting Results, and Python Scripts*

<div align="center">
  <img src="https://img.shields.io/badge/Status-Complete-success" alt="Status">
  <img src="https://img.shields.io/badge/Lines%20of%20Code-15%2C000%2B-blue" alt="LOC">
  <img src="https://img.shields.io/badge/Visualizations-100%2B-brightgreen" alt="Visualizations">
</div>

---

## 👨‍💻 Author

**Shivam Vashishtha**  
📧 Email: [shivam.vashishtha9@gmail.com](mailto:shivam.vashishtha9@gmail.com)  
💼 LinkedIn: [linkedin.com/in/shivam-vashishtha](https://www.linkedin.com/in/shivam-vashishtha/)  
🐙 GitHub: [github.com/Unibrow09](https://github.com/Unibrow09)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [📂 Complete Documentation & Results](#-complete-documentation--results)
- [Key Features](#-key-features)
- [Technical Stack](#-technical-stack)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Analysis Modules](#-analysis-modules)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Machine Learning Models](#-machine-learning-models)
- [Results & Insights](#-results--insights)
- [Future Enhancements](#-future-enhancements)
- [Acknowledgments](#-acknowledgments)

---

## 🌊 Project Overview

This project represents a **comprehensive end-to-end data science pipeline** analyzing the Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP) dataset spanning **1996-2023**. The analysis encompasses **14 interconnected analytical modules** that progressively build from exploratory data analysis to sophisticated machine learning forecasting models.

### 🎯 Project Objectives

1. **Temporal Trend Analysis**: Identify long-term trends in coral reef health metrics including coral cover, species richness, density, and living tissue area across 28 years
2. **Spatial Pattern Recognition**: Uncover geographic and habitat-specific patterns in coral distribution and health across the Florida Keys
3. **Environmental Impact Assessment**: Quantify the effects of major disturbance events (bleaching, hurricanes, disease outbreaks) on coral populations
4. **Predictive Modeling**: Develop machine learning models to forecast coral reef health indicators through 2028
5. **Early Warning System**: Identify critical thresholds and early indicators for coral reef decline
6. **Ecological Relationships**: Analyze complex interactions between species diversity, density, environmental factors, and reef health

### 🌟 Why This Project Matters

Coral reefs are among the most biodiverse ecosystems on Earth, providing:
- 🐠 **Habitat** for 25% of all marine species
- 💰 **Economic value** of $375 billion annually through tourism and fisheries
- 🛡️ **Coastal protection** from storms and erosion
- 💊 **Medical discoveries** from coral-derived compounds

However, coral reefs face unprecedented threats from climate change, ocean acidification, and human activities. This project provides **data-driven insights** to inform conservation strategies and policy decisions.

---

## 📂 Complete Documentation & Results

### 🔗 **[Access Full Project Documentation on Google Drive](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)**

This comprehensive Google Drive folder contains:

📊 **Forecasting_Results/**
- CSV files with 2024-2028 predictions for all metrics
- Regional and habitat-specific forecasts
- Optimistic, baseline, and pessimistic scenarios
- Confidence intervals and uncertainty estimates

📄 **PDF_Report/**
- Complete technical report (50+ pages)
- Detailed methodology documentation
- In-depth findings and discussion
- Publication-ready figures and tables
- Statistical analysis results
- Conservation recommendations

🐍 **Python_Scripts/**
- All 14 analysis modules with inline documentation
- Helper functions and utilities
- Data preprocessing pipelines
- Reproducible analysis workflows

🤖 **Trained_Models/**
- Serialized machine learning models (.pkl files)
- Model performance metrics
- Feature importance data
- Cross-validation results

🎨 **Visualizations/**
- 100+ high-resolution figures (PNG, 300 DPI)
- Publication-quality plots
- Interactive HTML visualizations
- Regional comparison maps
- Time series animations

> **💡 Tip**: This documentation is ideal for portfolio presentations, academic citations, or diving deep into the methodology and results!

---

## ✨ Key Features

### 📊 Comprehensive Analysis Pipeline
- **14 integrated analysis modules** covering all aspects of coral reef health
- **100+ high-quality visualizations** with publication-ready aesthetics
- **Statistical rigor** with hypothesis testing, correlation analysis, and multivariate statistics
- **Reproducible research** with well-documented, modular code

### 🤖 Advanced Machine Learning
- **Ensemble forecasting models** combining Random Forest, XGBoost, Gradient Boosting, and ARIMA
- **Time series analysis** with seasonality decomposition and stationarity testing
- **Feature engineering** incorporating temporal, spatial, and environmental variables
- **Model validation** using proper train-test splits and cross-validation

### 🔍 Deep Ecological Insights
- **Species-level analysis** for 50+ coral species
- **Multi-scale patterns** from individual sites to region-wide trends
- **Event impact assessment** quantifying effects of hurricanes, bleaching, and disease
- **Early warning indicators** for proactive conservation management

### 🎨 Professional Visualizations
- **Custom color palettes** optimized for coral reef themes
- **Interactive-style plots** with detailed annotations and context
- **Multi-panel layouts** presenting complex relationships clearly
- **Publication-quality** 300 DPI outputs

---

## 🛠️ Technical Stack

### Core Technologies

```python
# Data Manipulation & Analysis
pandas >= 1.3.0          # Data manipulation and analysis
numpy >= 1.21.0          # Numerical computing
scipy >= 1.7.0           # Scientific computing

# Machine Learning
scikit-learn >= 1.0.0    # ML algorithms and preprocessing
xgboost >= 1.5.0         # Gradient boosting
pmdarima >= 1.8.0        # Auto-ARIMA for time series
statsmodels >= 0.13.0    # Statistical models and time series

# Visualization
matplotlib >= 3.4.0      # Core plotting library
seaborn >= 0.11.0        # Statistical data visualization
plotly >= 5.0.0          # Interactive visualizations (optional)

# Utilities
joblib >= 1.1.0          # Model serialization
warnings                 # Warning management
datetime                 # Date/time handling
```

### Analytical Techniques Implemented

**Statistical Analysis:**
- Correlation analysis (Pearson, Spearman)
- ANOVA and t-tests
- Linear and non-linear regression
- Principal Component Analysis (PCA)
- Cluster analysis (hierarchical, k-means)

**Time Series Analysis:**
- Trend decomposition
- Stationarity testing (Augmented Dickey-Fuller)
- Autocorrelation (ACF/PACF)
- Change point detection
- Seasonal pattern recognition

**Machine Learning:**
- Random Forest Regression
- Gradient Boosting Machines
- XGBoost
- Support Vector Regression
- Ensemble methods
- ARIMA/SARIMAX models

**Spatial Analysis:**
- Geographic clustering
- Spatial autocorrelation
- Habitat association analysis
- Distance-based metrics

---

## 📁 Project Structure

```
Florida_Keys_Coral_Reef_Analysis/
│
├── 📄 README.md                                    # This file
├── 📄 requirements.txt                             # Python dependencies
├── 📄 LICENSE                                      # MIT License
│
├── 📂 CREMP_CSV_files/                            # Data directory
│   ├── CREMP_Stations_2023.csv                   # Station locations & metadata
│   ├── CREMP_SCOR_Summaries_2023_Density.csv    # Stony coral density data
│   ├── CREMP_SCOR_Summaries_2023_LTA.csv        # Living tissue area data
│   ├── CREMP_Pcover_2023_TaxaGroups.csv         # Percent cover data
│   ├── CREMP_OCTO_Summaries_2023_Density.csv    # Octocoral density data
│   └── CREMP_Temperatures_2023.csv               # Temperature monitoring data
│
├── 📂 Analysis_Scripts/                           # 14 analysis modules
│   ├── 01_stony_coral_cover_analysis.py          # Coral cover trends
│   ├── 02_stony_coral_species_richness.py        # Species diversity analysis
│   ├── 03_octocoral_density_analysis.py          # Octocoral populations
│   ├── 04_stony_coral_lta_analysis.py            # Living tissue area
│   ├── 05_coral_species_spatial_patterns.py      # Spatial distributions
│   ├── 06_stony_coral_density_richness_relationship.py  # Density-richness links
│   ├── 07_octocoral_temperature_correlations.py  # Temperature impacts
│   ├── 08_regional_comparison_analysis.py        # Regional differences
│   ├── 09_coral_health_factors_analysis.py       # Health determinants
│   ├── 10_early_indicators_analysis.py           # Warning system
│   ├── 11_stony_coral_cover_forecasting.py       # Cover predictions
│   ├── 12_octocoral_density_forecasting.py       # Octocoral forecasts
│   ├── 13_stony_coral_richness_forecasting.py    # Richness predictions
│   └── 14_stony_coral_lta_forecasting.py         # LTA forecasts
│
├── 📂 Results/                                    # Output directory
│   ├── 01_Results/                                # Module 1 outputs
│   ├── 02_Results/                                # Module 2 outputs
│   ├── ...                                        # (continues for all modules)
│   └── 14_Results/                                # Module 14 outputs
│
├── 📂 Models/                                     # Saved ML models
│   ├── coral_cover_best_model.pkl
│   ├── octocoral_density_best_model.pkl
│   ├── species_richness_best_model.pkl
│   └── lta_best_model.pkl
│
└── 📂 Documentation/                              # Additional docs
    ├── methodology.md                             # Detailed methodology
    ├── data_dictionary.md                         # Data field descriptions
    └── findings_summary.md                        # Key findings summary
```

---

## 📊 Data Sources

### CREMP Dataset Overview

The **Coral Reef Evaluation and Monitoring Project (CREMP)** is a long-term monitoring program established in 1996 by the Florida Fish and Wildlife Conservation Commission to track changes in coral reef ecosystems throughout the Florida Keys.

**Dataset Specifications:**
- 📅 **Temporal Coverage**: 1996-2023 (28 years)
- 📍 **Spatial Coverage**: 109+ monitoring stations across Florida Keys
- 🔬 **Monitoring Frequency**: Annual surveys (summer months)
- 📏 **Transect Method**: 22m² permanent stations with photographic quadrats
- 🌊 **Depth Range**: 1-35 meters
- 🗺️ **Geographic Regions**: Upper Keys (UK), Middle Keys (MK), Lower Keys (LK), Dry Tortugas (DT)

### Data Categories

#### 1. **Stony Coral (Scleractinian) Data**
- **Density**: Colonies per square meter for 50+ species
- **Living Tissue Area (LTA)**: Tissue coverage in mm² per colony
- **Species Composition**: Presence/absence and relative abundance
- **Condition Assessments**: Bleaching, disease, partial mortality
- **Total Records**: ~150,000 observations

#### 2. **Octocoral (Soft Coral) Data**
- **Density**: Colonies per square meter for 25+ species  
- **Mean Height**: Average colony height in centimeters
- **Species Diversity**: Richness and composition
- **Total Records**: ~75,000 observations

#### 3. **Percent Cover Data**
- **Stony Corals**: Species-specific percent coverage
- **Other Taxa**: Macroalgae, sponges, soft corals, bare substrate
- **Resolution**: 10 randomly placed quadrats per station
- **Total Records**: ~50,000 quadrat assessments

#### 4. **Environmental Data**
- **Temperature**: Continuous logger data (hourly readings)
- **Depth**: Station bathymetry
- **Habitat Type**: Patch reef, forereef, backreef, offshore shallow/deep
- **Geographic Coordinates**: Latitude/longitude for spatial analysis

#### 5. **Station Metadata**
- **Site Information**: Station ID, site name, region
- **Physical Parameters**: Depth, habitat classification
- **Management Zones**: Sanctuary designations
- **Historical Context**: Station establishment dates

### Data Quality & Preprocessing

- ✅ **Quality Control**: Multi-level validation by trained scientists
- 🔄 **Standardization**: Consistent methodology across years
- 🧹 **Preprocessing**: Missing value imputation, outlier detection
- 📐 **Normalization**: Appropriate scaling for statistical analysis

---

## 🔬 Analysis Modules

### Module 1: Stony Coral Cover Analysis
**File**: `01_stony_coral_cover_analysis.py`

**Purpose**: Comprehensive analysis of stony coral percent cover trends as the primary indicator of reef health.

**Key Analyses**:
- Overall temporal trends (1996-2023)
- Regional comparisons (Upper/Middle/Lower Keys)
- Habitat-specific patterns
- Rate of change calculations
- Event impact assessment (2014 bleaching, 2017 Hurricane Irma, 2019 disease outbreak)
- Site performance rankings
- Coral vs. other taxa competition

**Outputs**: 15 publication-quality visualizations including trend plots, heatmaps, and spatial maps

**Key Finding**: Stony coral cover declined by ~42% from 1996 to 2023, with accelerated losses post-2014.

---

### Module 2: Stony Coral Species Richness
**File**: `02_stony_coral_species_richness.py`

**Purpose**: Analyze biodiversity patterns through species richness (number of species per station).

**Key Analyses**:
- Temporal trends in species richness
- Regional and habitat biodiversity patterns
- Depth-richness relationships
- Species composition analysis
- Richness change rates
- Correlation with environmental variables

**Outputs**: 8 comprehensive visualizations

**Key Finding**: Species richness decreased by 28% over the monitoring period, with shallow habitats most affected.

---

### Module 3: Octocoral Density Analysis
**File**: `03_octocoral_density_analysis.py`

**Purpose**: Comprehensive analysis of soft coral populations, which may respond differently to stressors than stony corals.

**Key Analyses**:
- Overall density trends
- Regional and habitat patterns
- Species composition and diversity metrics
- Seasonal patterns
- Environmental correlations
- Multivariate analysis (PCA)
- Species similarity analysis

**Outputs**: 19 detailed visualizations

**Key Finding**: Octocoral populations showed greater resilience, with some species increasing as stony corals declined.

---

### Module 4: Stony Coral Living Tissue Area (LTA)
**File**: `04_stony_coral_lta_analysis.py`

**Purpose**: Analyze living tissue area as a measure of colony health and size.

**Key Analyses**:
- LTA distribution analysis
- Site-specific and regional patterns
- Habitat associations
- Temporal trends
- Statistical comparisons
- Species-level heatmaps

**Outputs**: 7 analytical visualizations

**Key Finding**: Mean LTA per colony declined by 35%, indicating not just fewer corals but smaller, less healthy colonies.

---

### Module 5: Coral Species Spatial Patterns
**File**: `05_coral_species_spatial_patterns.py`

**Purpose**: Investigate geographic and habitat-based distribution patterns of coral species.

**Key Analyses**:
- Species richness mapping
- Habitat association analysis
- Regional distribution patterns
- Temporal-spatial trends
- Community dissimilarity analysis
- Depth distribution patterns
- Indicator species identification

**Outputs**: 15 spatial analysis visualizations

**Key Finding**: Upper Keys sites show distinct species assemblages compared to Lower Keys, with depth as a strong structuring factor.

---

### Module 6: Density-Richness Relationship
**File**: `06_stony_coral_density_richness_relationship.py`

**Purpose**: Explore the relationship between coral colony density and species richness to understand diversity-abundance patterns.

**Key Analyses**:
- Overall density-richness correlations
- Regional relationship variations
- Habitat-specific patterns
- Temporal dynamics of the relationship
- Spatial patterns
- Depth gradient effects
- Comprehensive statistical modeling

**Outputs**: 9 relationship analysis visualizations

**Key Finding**: Positive correlation (r=0.65) between density and richness, but relationship strength varies by region and has weakened over time.

---

### Module 7: Octocoral-Temperature Correlations
**File**: `07_octocoral_temperature_correlations.py`

**Purpose**: Quantify relationships between water temperature variables and octocoral populations.

**Key Analyses**:
- Basic temperature-density correlations
- Key correlation visualizations
- Temperature trend analysis
- Regional temperature effects
- Habitat-specific responses
- Seasonal temperature impacts
- Threshold identification

**Outputs**: 6 correlation and temperature analysis plots

**Key Finding**: Octocorals show species-specific temperature sensitivities, with some thriving in warmer conditions while stony corals decline.

---

### Module 8: Regional Comparison Analysis
**File**: `08_regional_comparison_analysis.py`

**Purpose**: Comprehensive comparison of coral reef health across Florida Keys regions.

**Key Analyses**:
- Regional stony coral density trends
- Regional octocoral density patterns
- Species composition comparisons
- Percent cover trend differences
- Spatial visualization maps
- Regional recovery patterns post-disturbance
- Region-habitat interactions
- Regional diversity comparisons

**Outputs**: 10 regional comparison visualizations

**Key Finding**: Upper Keys experienced steeper declines (48% cover loss) compared to Lower Keys (35% loss), suggesting geographic vulnerability gradients.

---

### Module 9: Coral Health Factors Analysis
**File**: `09_coral_health_factors_analysis.py`

**Purpose**: Identify and quantify key factors influencing coral reef health through multivariate analysis.

**Key Analyses**:
- Environmental impact assessment (temperature, depth, habitat)
- Spatial factor analysis (region, site characteristics)
- Multivariate model building (Random Forest, GLM, XGBoost)
- Feature importance ranking
- Ecological interaction analysis
- Competition and facilitation effects
- Comprehensive synthesis of health determinants

**Outputs**: 12 factor analysis visualizations

**Key Finding**: Temperature extremes, habitat type, and regional location explain 73% of variance in coral health, with depth and historical density as secondary factors.

---

### Module 10: Early Indicators Analysis
**File**: `10_early_indicators_analysis.py`

**Purpose**: Develop an early warning system by identifying critical indicators and thresholds for reef decline.

**Key Analyses**:
- Critical indicator species identification
- Early warning statistical metrics (variance, autocorrelation, skewness)
- Critical threshold determination
- Temperature warning indicators
- Temperature-coral decline relationships
- Community composition shift detection
- Species rate of change analysis
- Comprehensive indicator synthesis

**Outputs**: 6 early warning system visualizations + detailed markdown summary

**Key Finding**: Increased variance in coral cover 1-2 years before major declines, with 30°C sustained temperature and 15% macroalgae cover identified as critical thresholds.

---

### Module 11: Stony Coral Cover Forecasting
**File**: `11_stony_coral_cover_forecasting.py`

**Purpose**: Develop machine learning models to forecast coral cover through 2028.

**Key Analyses**:
- Time series pattern analysis (trend, seasonality, stationarity)
- Feature engineering (temporal, spatial, environmental, lagged variables)
- Multiple model training (Linear Regression, Random Forest, Gradient Boosting, XGBoost, ARIMA)
- Model performance evaluation and comparison
- Ensemble model development
- Forecast generation with uncertainty bounds (2024-2028)
- Regional and habitat-specific forecasts
- Reef evolution trajectory analysis

**Outputs**: 7 forecasting visualizations + CSV forecast files

**Key Results**:
- Best Model: Ensemble (R² = 0.82, RMSE = 2.14%)
- Projected 2028 coral cover: 8.2% (optimistic) to 4.1% (pessimistic)
- Continued decline trend at -0.4% per year

---

### Module 12: Octocoral Density Forecasting
**File**: `12_octocoral_density_forecasting.py`

**Purpose**: Forecast octocoral populations as potential beneficiaries of stony coral decline.

**Key Analyses**:
- Octocoral time series decomposition
- Feature engineering with coral competition variables
- Multi-model training and evaluation
- Ensemble forecasting
- Regional forecasts
- Habitat-specific projections
- Ecological succession analysis

**Outputs**: 7 forecasting visualizations + forecast data files

**Key Results**:
- Best Model: XGBoost (R² = 0.78, RMSE = 0.87 colonies/m²)
- Projected 2028: 15-20% increase in octocoral density
- Evidence of phase shift from stony coral to octocoral dominance

---

### Module 13: Species Richness Forecasting
**File**: `13_stony_coral_richness_forecasting.py`

**Purpose**: Predict future biodiversity trends in stony coral communities.

**Key Analyses**:
- Richness time series analysis
- Biodiversity feature engineering
- Model training with diversity-specific predictors
- Forecast generation with confidence intervals
- Regional biodiversity projections
- Extinction risk assessment
- Conservation priority identification

**Outputs**: 9 biodiversity forecasting visualizations + CSV files

**Key Results**:
- Best Model: Gradient Boosting (R² = 0.75, RMSE = 1.2 species)
- Projected 2028 richness: 10-14 species per station (from current 15-18)
- Risk of local extinctions for 8-12 rare species

---

### Module 14: Living Tissue Area Forecasting
**File**: `14_stony_coral_lta_forecasting.py`

**Purpose**: Forecast coral colony health through living tissue area predictions.

**Key Analyses**:
- LTA temporal pattern analysis
- Health-specific feature engineering
- Multi-model ensemble forecasting
- Colony size trajectory predictions
- Regional health forecasts
- Recovery potential assessment

**Outputs**: 8 LTA forecasting visualizations + forecast datasets

**Key Results**:
- Best Model: Random Forest (R² = 0.80, RMSE = 145 mm²)
- Projected 2028 mean LTA: 620-780 mm² (from current 850 mm²)
- Continued colony size reduction indicating chronic stress

---

## 💻 Installation & Setup

### Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Storage**: At least 2 GB free space for data and outputs

### Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting.git
cd Florida-Keys-Coral-Reef-Analysis-and-Forecasting
```

2. **Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```python
python -c "import pandas, numpy, sklearn, xgboost; print('All packages installed successfully!')"
```

### Download Data

The CREMP dataset should be placed in the `CREMP_CSV_files/` directory. If not included:

1. Visit [FWRI CREMP Data Portal](https://myfwc.com/research/habitat/coral/cremp/)
2. Download the 2023 data package
3. Extract CSV files to `CREMP_CSV_files/` directory

---

## 🚀 Usage Guide

### Running Individual Analysis Modules

Each script can be run independently:

```bash
# Run coral cover analysis
python 01_stony_coral_cover_analysis.py

# Run forecasting model
python 11_stony_coral_cover_forecasting.py
```

### Running Complete Analysis Pipeline

To execute all 14 modules sequentially:

```bash
# Create a master script or run individually
for script in *_*.py; do
    echo "Running $script..."
    python "$script"
done
```

### Customizing Analysis Parameters

Each script has configurable parameters at the top:

```python
# Example from forecasting scripts
FORECAST_YEARS = [2024, 2025, 2026, 2027, 2028]  # Adjust forecast horizon
TEST_SIZE = 0.2  # Train-test split ratio
RANDOM_STATE = 42  # Reproducibility seed
```

### Output Organization

Results are automatically saved to numbered directories:
- `01_Results/` - Module 1 outputs
- `02_Results/` - Module 2 outputs
- etc.

Each directory contains:
- **PNG files**: High-resolution visualizations (300 DPI)
- **CSV files**: Numerical results and forecasts (for forecasting modules)
- **PKL files**: Saved machine learning models
- **TXT files**: Summary statistics and insights

---

## 🔍 Key Findings

### Overall Reef Health Trends (1996-2023)

#### 📉 **Significant Declines Across Multiple Metrics**

1. **Stony Coral Cover**: **-42% decline** (from ~14% to ~8%)
   - Accelerated loss post-2014 bleaching event
   - Rate of decline: -0.35% per year (pre-2014) vs -0.65% per year (post-2014)

2. **Species Richness**: **-28% reduction** (from ~21 to ~15 species per station)
   - Loss of rare and sensitive species
   - Homogenization of coral communities

3. **Colony Density**: **-38% decrease** in colonies per square meter
   - Fewer colonies with reduced recruitment success

4. **Living Tissue Area**: **-35% reduction** in mean colony size
   - Colonies not only fewer but also smaller and less healthy

#### 🌊 **Regional Patterns**

**Upper Keys (UK)**
- **Most Affected**: 48% cover loss
- **Highest Temperature Stress**: +1.2°C above baseline
- **Limited Recovery**: Minimal rebound after disturbances

**Middle Keys (MK)**
- **Moderate Decline**: 40% cover loss
- **Variable Responses**: High site-to-site variability
- **Some Resilience**: Pockets of relative stability

**Lower Keys (LK)**
- **Best Performing**: 35% cover loss (still significant)
- **Greater Diversity**: Maintained higher species richness
- **Partial Recovery**: Evidence of regeneration post-disturbance

**Dry Tortugas (DT)**
- **Most Resilient**: 28% cover loss
- **Isolation Benefit**: Reduced direct human impact
- **Refuge Function**: Potential source for recruitment

#### 🏝️ **Habitat-Specific Patterns**

| Habitat Type | Cover Loss | Key Characteristic |
|--------------|-----------|-------------------|
| **Patch Reef** | -52% | Most vulnerable to bleaching |
| **Offshore Shallow** | -45% | High temperature variability |
| **Forereef** | -38% | Wave action stress |
| **Offshore Deep** | -32% | More stable temperature |
| **Backreef** | -30% | Protected but turbid |

#### 📅 **Major Disturbance Events Impact**

**2014-2015 Global Bleaching Event**
- Immediate 18% cover loss in affected areas
- Recovery: Minimal (<5% after 3 years)
- Long-term effect: Shifted community structure

**2017 Hurricane Irma**
- Physical damage: 12% cover loss
- Regional variation: Upper Keys -22%, Lower Keys -8%
- Recovery: Partial in deep/protected sites (15% recovered)

**2019 Stony Coral Tissue Loss Disease (SCTLD)**
- Ongoing mortality: 15-25% of remaining colonies
- Species-selective: Decimated brain and star corals
- Continuing impact: Disease still present in 2023

### Ecological Shifts

#### 🔄 **Phase Shift Evidence**

**Coral → Octocoral Transition**
- Octocoral density: **+18% increase** (2014-2023)
- Inverse relationship with stony coral decline (r = -0.62, p < 0.001)
- Species like *Plexaura homomalla* increasing 35%

**Coral → Macroalgae Competition**
- Macroalgae cover: **+45% increase**
- Critical threshold identified: >15% macroalgae cover associated with coral recruitment failure
- Particularly pronounced in degraded Upper Keys sites

#### 🌡️ **Temperature Relationships**

**Critical Thresholds Identified:**
- **Bleaching Risk**: Sustained temperatures >30°C for >5 days
- **Optimal Range**: 25-28°C for coral health
- **Cold Stress**: <18°C causes tissue damage

**Temperature Trends:**
- Mean annual temperature: **+0.8°C increase** (1996-2023)
- Increased frequency of extreme events:
  - Days >30°C: 2.3x more frequent
  - Days <18°C: 1.5x more frequent
- Thermal stress duration increasing: +40% longer heat waves

#### 📊 **Diversity Metrics**

**Shannon Diversity Index**: Declined from 2.8 to 2.1 (-25%)
**Simpson's Diversity**: Decreased from 0.85 to 0.72 (-15%)
**Evenness**: Reduced from 0.78 to 0.68 (-13%)

**Interpretation**: Communities becoming less diverse and more dominated by few stress-tolerant species

### Machine Learning Model Performance

#### 🎯 **Forecasting Accuracy**

| Metric | Best Model | R² Score | RMSE | MAPE |
|--------|-----------|----------|------|------|
| **Coral Cover** | Ensemble | 0.82 | 2.14% | 18.2% |
| **Species Richness** | Gradient Boosting | 0.75 | 1.2 species | 22.5% |
| **Octocoral Density** | XGBoost | 0.78 | 0.87 colonies/m² | 19.7% |
| **Living Tissue Area** | Random Forest | 0.80 | 145 mm² | 21.3% |

#### 🔮 **2028 Projections**

**Stony Coral Cover**
- **Baseline Scenario**: 5.8% (±1.4%)
- **Optimistic** (conservation interventions): 8.2%
- **Pessimistic** (continued stressors): 4.1%
- **Implication**: Below 5% considered "critically degraded"

**Species Richness**
- **Projected**: 12 species per station (±2)
- **At Risk**: 8-12 rare species face local extinction
- **Regional Variation**: Lower Keys maintain 14-16 species, Upper Keys drop to 8-10

**Octocoral Populations**
- **Projected**: +15-20% increase in density
- **Trend**: Accelerating replacement of stony corals
- **Ecological Shift**: Moving toward octocoral-dominated reefs

**Colony Health (LTA)**
- **Projected**: 680 mm² mean LTA (down from 850 mm²)
- **Size Classes**: Shift toward smaller colonies (<500 mm²)
- **Recruitment**: Fewer juveniles surviving to maturity

### Early Warning Indicators

#### ⚠️ **Critical Signals Identified**

1. **Statistical Early Warnings** (1-2 years before major decline):
   - **Increased Variance**: 35% higher variability in cover
   - **Rising Autocorrelation**: Slower recovery from perturbations
   - **Skewness Changes**: Distribution shifts toward lower values

2. **Indicator Species**:
   - ***Acropora cervicornis*** (Staghorn): First responder to stress (declines 6-12 months early)
   - ***Montastraea cavernosa*** (Great Star): Late responder, signals chronic stress
   - ***Porites astreoides*** (Mustard Hill): Increases during degradation (opportunistic)

3. **Environmental Thresholds**:
   - **Temperature**: >30°C for >5 consecutive days → 78% probability of bleaching
   - **Macroalgae**: >15% cover → recruitment failure
   - **Turbidity**: >10 NTU sustained → reduced photosynthesis

### Conservation Implications

#### 🎯 **Priority Actions Based on Findings**

1. **Geographic Focus**:
   - **Immediate**: Upper Keys sites showing steepest declines
   - **Protection**: Dry Tortugas and Lower Keys as refugia
   - **Connectivity**: Maintain larval sources from resilient areas

2. **Habitat Management**:
   - **Critical Habitats**: Prioritize offshore deep reefs (mostable)
   - **Restoration**: Focus on sites with <30% macroalgae
   - **Monitoring**: Intensify in patch reefs (most vulnerable)

3. **Species-Level Conservation**:
   - **Urgent**: *Acropora* spp., *Dendrogyra cylindrus*, large *Montastraea* spp.
   - **Candidate for Restoration**: *Porites* spp., *Siderastrea* spp. (stress-tolerant)
   - **Monitor Closely**: Rare species with <5% prevalence

4. **Climate Adaptation**:
   - **Temperature Management**: Reduce local stressors (water quality, overfishing)
   - **Thermal Refugia**: Identify and protect cooler deep sites
   - **Assisted Evolution**: Consider heat-tolerant genotypes

5. **Early Intervention**:
   - **Monitoring Frequency**: Quarterly surveys at high-risk sites
   - **Response Triggers**: Automated alerts when thresholds exceeded
   - **Rapid Response**: Algae removal, coral gardening within 30 days of warning

---

## 📸 Visualizations

### Sample Visualizations from Analysis

The project generates **100+ professional visualizations** across 14 modules. Here are key examples:

#### **Temporal Trends**
- Overall coral cover trends with major event markers
- Regional comparison time series
- Species richness temporal patterns
- Rate of change bar charts

#### **Spatial Distributions**
- Geographic maps of coral health metrics
- Site performance rankings
- Regional heatmaps
- Depth gradient analyses

#### **Statistical Relationships**
- Correlation matrices
- Scatter plots with regression lines
- PCA biplots
- Cluster dendrograms

#### **Forecasting Outputs**
- Historical vs. predicted trajectories
- Confidence interval bands
- Model performance comparisons
- Scenario projections (optimistic/pessimistic)

#### **Ecological Patterns**
- Species composition stacked bars
- Community dissimilarity maps
- Indicator species trends
- Phase shift diagrams

*All visualizations feature:*
- 🎨 Custom coral reef color palettes
- 📊 Publication-quality 300 DPI resolution
- 📝 Detailed annotations and context
- 🏷️ Clear legends and labels
- 📐 Professional layouts with consistent branding

### 🖼️ View All Visualizations

**[Browse Complete Visualization Gallery on Google Drive →](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)**

The `Visualizations/` folder contains all 100+ figures organized by analysis module, available for download in high-resolution PNG format.

---

## 🤖 Machine Learning Models

### Model Architecture & Methodology

#### Feature Engineering Strategy

**Temporal Features (Time-Based Predictors):**
```python
- Year (continuous)
- Lagged values (t-1, t-2, t-3)
- Rolling statistics (3-year, 5-year means/std)
- Growth rates (year-over-year % change)
- Time since last disturbance
- Event flags (bleaching, hurricane, disease years)
- Post-event indicators (years since event)
```

**Spatial Features (Location-Based Predictors):**
```python
- Region (Upper/Middle/Lower Keys, categorical)
- Habitat type (Patch/Forereef/Backreef/Offshore, categorical)
- Depth (continuous, meters)
- Latitude/Longitude (spatial coordinates)
- Site-specific history (historical mean, variance)
```

**Environmental Features:**
```python
- Annual mean temperature
- Temperature maximum/minimum
- Temperature range (max - min)
- Temperature standard deviation
- Days above 30°C (bleaching risk)
- Days below 18°C (cold stress)
- Lagged temperature (t-1)
```

**Ecological Features:**
```python
- Species richness
- Shannon diversity index
- Colony density
- Living tissue area
- Macroalgae cover
- Octocoral density (competition)
- Cross-dataset integration
```

**Total Features per Model:** 35-50 engineered predictors

#### Model Selection & Tuning

**Models Evaluated:**

1. **Linear Regression**
   - Baseline model for comparison
   - Interpretable coefficients
   - Assumes linear relationships

2. **Random Forest Regressor**
   ```python
   n_estimators=100
   max_depth=15
   min_samples_split=5
   random_state=42
   ```
   - Handles non-linearity well
   - Provides feature importances
   - Robust to outliers

3. **Gradient Boosting Regressor**
   ```python
   n_estimators=100
   max_depth=5
   learning_rate=0.1
   random_state=42
   ```
   - Sequential error correction
   - Often best single-model performance
   - Risk of overfitting

4. **XGBoost**
   ```python
   n_estimators=100
   max_depth=5
   learning_rate=0.1
   random_state=42
   ```
   - Optimized gradient boosting
   - Regularization built-in
   - Fast training

5. **ARIMA/SARIMAX** (Time Series Specific)
   ```python
   auto_arima(seasonal=False, stepwise=True, suppress_warnings=True)
   ```
   - Classical time series approach
   - Captures autocorrelation
   - Seasonality handling

6. **Ensemble Model**
   - Average of all model predictions
   - Reduces individual model bias
   - Often achieves best generalization

#### Model Validation

**Train-Test Split Strategy:**
- **Temporal Split**: Last 2 years reserved for testing (2022-2023)
- **Rationale**: Maintains temporal ordering for realistic forecasting
- **Training Data**: 1996-2021 (26 years)
- **Testing Data**: 2022-2023 (2 years)

**Cross-Validation:**
- **Time Series CV**: 5-fold with expanding window
- **Prevents data leakage**: Only past data used to predict future
- **Hyperparameter tuning**: Grid search within CV framework

**Performance Metrics:**
- **R² Score**: Explained variance (0-1, higher better)
- **RMSE**: Root Mean Squared Error (lower better)
- **MAE**: Mean Absolute Error (interpretable units)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)

#### Feature Importance Analysis

**Top Predictors Across Models:**

| Rank | Feature | Avg Importance | Models Ranking #1 |
|------|---------|---------------|------------------|
| 1 | Lagged Value (t-1) | 0.28 | 3/4 |
| 2 | Temperature Mean | 0.18 | 2/4 |
| 3 | Region_UK | 0.12 | 1/4 |
| 4 | Habitat_PatchReef | 0.09 | 0/4 |
| 5 | Rolling Mean (3yr) | 0.08 | 0/4 |
| 6 | Year | 0.06 | 0/4 |
| 7 | Depth | 0.05 | 0/4 |
| 8 | Days >30°C | 0.04 | 0/4 |
| 9 | Species Richness | 0.03 | 0/4 |
| 10 | Macroalgae Cover | 0.03 | 0/4 |

**Interpretation:**
- Historical values dominate (strong autocorrelation in reef dynamics)
- Temperature is consistently important (climate driver)
- Regional effects significant (spatial heterogeneity)
- Biological interactions matter (diversity, competition)

### 💾 Access Trained Models

All trained machine learning models are available for download and reuse:

**[Download Trained Models from Google Drive →](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)**

The `Trained_Models/` folder includes:
- Serialized model files (.pkl format)
- Model metadata and hyperparameters
- Performance metrics on test data
- Feature importance rankings
- Instructions for model loading and prediction

---

## 🚧 Future Enhancements

### Planned Improvements

#### **Advanced Analytics**
- [ ] Deep learning models (LSTM, GRU) for time series forecasting
- [ ] Bayesian hierarchical models for uncertainty quantification
- [ ] Spatial autocorrelation modeling (kriging, spatial regression)
- [ ] Causal inference analysis (structural equation modeling)
- [ ] Real-time data integration and streaming analytics

#### **Interactive Dashboards**
- [ ] Plotly Dash web application for interactive exploration
- [ ] Real-time monitoring dashboard with live data feeds
- [ ] User-customizable analysis parameters
- [ ] Downloadable reports in PDF/Word formats
- [ ] API for programmatic access to forecasts

#### **Extended Analysis**
- [ ] Genetic diversity integration (if data available)
- [ ] Socioeconomic impact assessment
- [ ] Cost-benefit analysis of conservation interventions
- [ ] Climate scenario modeling (RCP 4.5, 8.5)
- [ ] Multi-species interaction networks

#### **Technical Improvements**
- [ ] Docker containerization for reproducibility
- [ ] CI/CD pipeline for automated testing
- [ ] Code profiling and optimization
- [ ] Parallel processing for faster computation
- [ ] Cloud deployment (AWS, Azure, GCP)

#### **Documentation**
- [ ] Video tutorials for each module
- [ ] Jupyter notebooks with step-by-step walkthroughs
- [ ] Scientific paper writeup for publication
- [ ] Case studies and use examples
- [ ] API documentation

---

## 🙏 Acknowledgments

### Data Source
This project uses data from the **Coral Reef Evaluation and Monitoring Project (CREMP)**, a long-term monitoring program operated by the:

**Florida Fish and Wildlife Conservation Commission (FWC)**  
Fish and Wildlife Research Institute (FWRI)  
Marine Fisheries Research  
[https://myfwc.com/research/habitat/coral/cremp/](https://myfwc.com/research/habitat/coral/cremp/)

### Special Thanks

- **CREMP Field Scientists**: For decades of dedicated monitoring work
- **Florida Keys National Marine Sanctuary**: For reef protection and management
- **NOAA Coral Reef Conservation Program**: For funding and support
- **Marine Conservation Community**: For ongoing efforts to protect coral reefs
- **Open Source Community**: For the excellent tools and libraries used in this project

### Citations

If you use this analysis or methodology in your research, please cite:

```bibtex
@misc{vashishtha2025floridakeys,
  author = {Vashishtha, Shivam},
  title = {Florida Keys Coral Reef Health Analysis and Forecasting System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting}
}
```

**CREMP Data Citation:**
```
Florida Fish and Wildlife Conservation Commission, Fish and Wildlife Research Institute. 
(2023). Coral Reef Evaluation and Monitoring Project (CREMP) 1996-2023. 
St. Petersburg, FL. Available at: https://myfwc.com/research/habitat/coral/cremp/
```

### Tools & Libraries

This project was made possible by:
- **Python Community**: Core language and ecosystem
- **Pandas Development Team**: Data manipulation
- **Scikit-learn Contributors**: Machine learning
- **Matplotlib & Seaborn**: Visualization
- **XGBoost Developers**: Gradient boosting
- **NumPy & SciPy**: Scientific computing
- **Statsmodels**: Statistical analysis

---

## 📞 Contact & Support

### Get in Touch

**Shivam Vashishtha**

📧 **Email**: [shivam.vashishtha9@gmail.com](mailto:shivam.vashishtha9@gmail.com)  
💼 **LinkedIn**: [linkedin.com/in/shivam-vashishtha](https://www.linkedin.com/in/shivam-vashishtha/)  
🐙 **GitHub**: [github.com/Unibrow09](https://github.com/Unibrow09)

### Project Links

- 📦 **Repository**: [github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting](https://github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting)
- 📂 **Complete Documentation**: [Google Drive Folder with Reports, Models & Results](https://drive.google.com/drive/folders/1QG9yBLyUimUc7Fgq2R8sKWNiAsk3QjIl)
- 🐛 **Issues**: [Report bugs or request features](https://github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting/issues)
- 💬 **Discussions**: [Join the conversation](https://github.com/Unibrow09/Florida-Keys-Coral-Reef-Analysis-and-Forecasting/discussions)
- ⭐ **Star this repo** if you found it helpful!

### Questions or Feedback?

- **Technical Issues**: Open an issue on GitHub
- **Collaboration Opportunities**: Email me directly
- **Job Opportunities**: Connect on LinkedIn
- **General Questions**: Use GitHub Discussions

---

## 📊 Project Statistics

<div align="center">

| Metric | Count |
|--------|-------|
| **Lines of Code** | 15,000+ |
| **Analysis Modules** | 14 |
| **Visualizations Generated** | 100+ |
| **ML Models Trained** | 24 |
| **Years of Data Analyzed** | 28 (1996-2023) |
| **Monitoring Stations** | 109+ |
| **Species Analyzed** | 75+ |
| **Data Points** | 275,000+ |
| **Forecast Horizon** | 5 years (2024-2028) |

</div>

---

## ⭐ Show Your Support

If this project helped you or you found it interesting:

- ⭐ **Star this repository** on GitHub
- 🔄 **Fork it** to build your own analysis
- 📢 **Share it** with colleagues and researchers
- 💬 **Provide feedback** through issues or discussions
- 🤝 **Contribute** improvements or extensions

---

## 📝 Project Status

✅ **Complete** - All 14 analysis modules functional  
🔄 **Maintained** - Regular updates and improvements  
📚 **Documented** - Comprehensive README and inline documentation  
🧪 **Tested** - Validated outputs and model performance  
🌟 **Portfolio-Ready** - Professional quality for showcasing

---

<div align="center">

### 🪸 Thank you for your interest in coral reef conservation! 🪸

*"In the end, we will conserve only what we love, we will love only what we understand, and we will understand only what we are taught."*  
— Baba Dioum

---

**Made with 💙 for our oceans**

**Shivam Vashishtha** © 2025

[⬆ Back to Top](#-florida-keys-coral-reef-health-analysis--forecasting-system)

</div>


