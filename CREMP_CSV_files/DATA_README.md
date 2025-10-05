# CREMP Data Files

## Required Data Files

To run the analysis scripts in this project, you need to place the following CSV files in this directory:

1. **PER_COVER.csv** - Coral percentage cover measurements
2. **STATIONS.csv** - Station location and metadata
3. **SPECIES.csv** - Stony coral species presence/absence data
4. **OCTO_DENS.csv** - Octocoral density measurements
5. **LTA.csv** - Living tissue area measurements
6. **DIST.csv** - Disturbance events data
7. **TEMP_DAILY.csv** - Daily temperature measurements
8. **SPEC_DENS.csv** - Species-specific density data

## Data Source

The data for this project comes from the **Florida Keys Coral Reef Evaluation and Monitoring Project (CREMP)**.

### How to Obtain the Data:

1. Visit the Florida Fish and Wildlife Conservation Commission (FWC) website
2. Navigate to the CREMP data portal: [https://ocean.floridamarine.org/CREMP/](https://ocean.floridamarine.org/CREMP/)
3. Request access to the historical monitoring data (1996-2023)
4. Download the CSV files listed above
5. Place them in this directory (`CREMP_CSV_files/`)

### Alternative:

If you have access to the original data files from the Florida Keys Data Challenge, place them in this directory.

### Data Format Notes:

- All CSV files should be UTF-8 encoded
- Date columns should be in format compatible with pandas datetime parsing
- No modifications to original file names are needed - the scripts are designed to work with the standard CREMP data exports

## Contact for Data Access:

For questions about data access, contact the Florida Fish and Wildlife Conservation Commission or refer to the CREMP program documentation.

---

**Note:** Due to data licensing and size constraints, the raw data files are not included in this repository. Users must obtain them independently from the official CREMP sources.