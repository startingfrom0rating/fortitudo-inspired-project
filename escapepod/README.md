# Escapepod - Portfolio Optimization Input Data

This folder contains **only the input data** from the portfolio optimization simulation - no outputs, analysis reports, or generated visualizations.

## Contents Overview

This escapepod contains all the source data used for the portfolio optimization analysis, including ETF data, bond data, historic returns, historic prices, and related input datasets.

## File Inventory

### ETF Data Files
- **`etfs_prices.csv`** (5.7MB) - Historical daily price data for all ETFs
- **`etfs_returns.csv`** (5.2MB) - Calculated returns data for ETFs  
- **`etfs_returns.backup.csv`** (5.3MB) - Backup copy of ETF returns
- **`etfs_returns_backfill.csv`** (534KB) - Backfilled ETF returns data
- **`etfs_identifiers.csv`** (819 bytes) - ETF ticker identifiers in CSV format
- **`etfs_identifiers.txt`** (25KB) - Detailed ETF information including names, asset classes, sectors
- **`etfs_invalid.csv`** (444 bytes) - List of invalid/excluded ETF tickers

### Bond/Treasury Data Files  
- **`treasuries_returns_monthly.csv`** (140KB) - Monthly returns for treasury bonds
- **`treasuries_identifiers.csv`** (6 bytes) - Treasury bond identifiers (CSV)
- **`treasuries_identifiers.txt`** (5.7KB) - Detailed treasury bond information
- **`fred_treasuries_timeseries.csv`** (535KB) - FRED economic data for treasuries
- **`fred_mapping_treasuries.csv`** (4.4KB) - Mapping between FRED codes and treasury bonds

### Historic Returns & Prices
- **`R_monthly.csv`** (390KB) - **Primary monthly returns matrix** used for optimization (111 months × 120 instruments)
- **`p_monthly.csv`** (3.6KB) - Probability weights for scenarios  
- **`v_monthly.csv`** (4.9KB) - Additional variance/volatility data

### Metadata & Configuration
- **`meta_monthly_columns.txt`** (4.1KB) - Column metadata for monthly data files

### Official Approval Lists
- **`25-26-WGHIC-Approved-ETF-List.pdf`** (144KB) - Wharton competition approved ETF list
- **`25-26-WGHIC-Approved-Treasury-Bonds-List.pdf`** (76KB) - Wharton competition approved treasury bonds

## Data Timeline
- **Period Covered**: June 2016 to August 2025
- **Frequency**: Monthly data (111 observations)
- **Universe**: 100 ETFs + 20 Treasury bonds = 120 total instruments

## Data Usage
This data was used as input for mean-variance portfolio optimization with the following constraints:
- Long-only (no short selling)
- Maximum 4% position size per instrument  
- Minimum 50% equity allocation
- Maximum 12% annual volatility
- Full investment (weights sum to 100%)

## Key Input Files for Analysis
1. **`R_monthly.csv`** - Main returns matrix for optimization
2. **`etfs_prices.csv`** - Historical ETF price data
3. **`etfs_identifiers.txt`** - ETF classification information
4. **`treasuries_returns_monthly.csv`** - Bond returns data
5. **PDF files** - Official approval lists for investment universe

## Total Data Size
Approximately 17.7MB of input data across 19 files.

---

**Note**: This folder contains NO outputs, analysis reports, visualizations, or generated results from the optimization process. Only raw input data that was fed into the portfolio optimization algorithm.

*Generated on: September 29, 2025*
*Source: Portfolio Optimization Simulation Data*