# CREMP Data Directory

This directory should contain the Coral Reef Evaluation and Monitoring Project (CREMP) dataset files.

## Required Data Files

The following CSV files are required to run the analysis scripts:

### Station Metadata
- `CREMP_Stations_2023.csv` - Station locations, coordinates, depth, habitat types, and regional classifications

### Stony Coral Data
- `CREMP_SCOR_Summaries_2023_Density.csv` - Stony coral colony density data by species
- `CREMP_SCOR_Summaries_2023_LTA.csv` - Living tissue area measurements
- `CREMP_SCOR_Summaries_2023_Counts.csv` - Colony counts by species
- `CREMP_SCOR_Summaries_2023_ConditionCounts.csv` - Coral condition assessments
- `CREMP_SCOR_RawData_2023.csv` - Raw stony coral observation data

### Percent Cover Data
- `CREMP_Pcover_2023_TaxaGroups.csv` - Percent cover by taxonomic groups
- `CREMP_Pcover_2023_StonyCoralSpecies.csv` - Species-level percent cover data

### Octocoral Data
- `CREMP_OCTO_Summaries_2023_Density.csv` - Octocoral density by species
- `CREMP_OCTO_Summaries_2023_MeanHeight.csv` - Mean colony heights
- `CREMP_OCTO_RawData_2023.csv` - Raw octocoral observation data

### Environmental Data
- `CREMP_Temperatures_2023.csv` - Temperature logger data (hourly readings)

## Data Source

All data files can be downloaded from the Florida Fish and Wildlife Conservation Commission:

**Official CREMP Data Portal:**  
🔗 [https://myfwc.com/research/habitat/coral/cremp/](https://myfwc.com/research/habitat/coral/cremp/)

### How to Download

1. Visit the CREMP data portal link above
2. Navigate to the "Data" section
3. Download the 2023 dataset package (latest available)
4. Extract all CSV files to this directory

## Data Size

⚠️ **Note**: The complete dataset is approximately **500 MB - 1 GB** depending on the files included.

- Most CSV files: 1-50 MB each
- Temperature data: ~200-400 MB (hourly data for 28 years)
- Total extracted: ~800 MB

## Data Structure

Each CSV file contains:
- **Temporal data**: Year, Date columns
- **Spatial data**: StationID, SiteID, Subregion, Habitat
- **Species data**: Individual columns for each species
- **Metadata**: Various descriptive and categorical fields

## Data Usage Guidelines

Please cite the data source in any publications or reports:

```
Florida Fish and Wildlife Conservation Commission, Fish and Wildlife Research Institute. 
(2023). Coral Reef Evaluation and Monitoring Project (CREMP) 1996-2023. 
St. Petersburg, FL. Available at: https://myfwc.com/research/habitat/coral/cremp/
```

## File Format Notes

- **Delimiter**: Comma-separated (CSV)
- **Encoding**: UTF-8
- **Missing values**: Represented as empty cells or "NA"
- **Date format**: YYYY-MM-DD or MM/DD/YYYY (varies by file)
- **Numeric precision**: Varies by measurement type

## Troubleshooting

### File Not Found Errors

If you encounter "file not found" errors when running scripts:

1. Ensure all required CSV files are in this directory
2. Check that file names match exactly (case-sensitive on some systems)
3. Verify files are not corrupted (try re-downloading)

### Memory Issues

The temperature dataset is very large. If you encounter memory issues:

1. The scripts handle this by loading temperature data separately
2. Consider using a machine with at least 8 GB RAM
3. Close other applications while running analysis

### Data Version Compatibility

These scripts were developed using the **2023 dataset version**. If using a different year:

1. Update file names in the scripts accordingly
2. Check for any schema changes in the data structure
3. Test each script individually to identify issues

## Privacy & Ethics

This is publicly available environmental monitoring data with no personal information. All data collection was conducted under appropriate permits and follows ethical scientific standards.

## Questions?

For data-related questions, contact the CREMP team:
- 📧 Email: [FWRICoralProgram@MyFWC.com](mailto:FWRICoralProgram@MyFWC.com)
- 📞 Phone: (727) 896-8626

For technical issues with the analysis scripts, open an issue on GitHub.

---

**Status**: ⚠️ Data files not included in repository (too large)  
**Action Required**: Download data files before running analysis scripts

