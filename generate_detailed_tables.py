"""
Generate detailed portfolio tables showing all holdings for the comprehensive report
"""
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.generate_plots_and_report import load_inputs, classify_indices, optimize
from fortitudo.tech.functions import simulation_moments

def generate_comprehensive_holdings_tables():
    """Generate comprehensive tables showing all portfolio holdings"""
    
    print("Generating comprehensive portfolio holdings tables...")
    
    # Load data and run optimization
    Rdf, keep_cols, etf_cols = load_inputs()
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=0.12)
    
    # Create comprehensive portfolio holdings dataframe
    w_series = pd.Series(w, index=keep_cols).sort_values(ascending=False)
    
    # Filter to only positions with weight > 1e-6 (effectively > 0)
    holdings = w_series[w_series > 1e-6].copy()
    
    # Create detailed holdings table
    holdings_data = []
    for rank, (ticker, weight) in enumerate(holdings.items(), 1):
        # Determine asset type
        if ticker in etf_cols:
            if equity_idx and keep_cols.index(ticker) in equity_idx:
                asset_type = "Equity ETF"
            elif bond_idx and keep_cols.index(ticker) in bond_idx:
                asset_type = "Bond ETF"
            else:
                asset_type = "ETF"
        else:
            asset_type = "Treasury Bond"
        
        # Determine sector/category for ETFs
        sector_mapping = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare', 'XLI': 'Industrials',
            'XLE': 'Energy', 'XLU': 'Utilities', 'XLB': 'Materials', 'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary', 'XLC': 'Communication Services',
            'QQQ': 'Technology (NASDAQ)', 'SMH': 'Semiconductors', 'VGT': 'Technology',
            'VHT': 'Healthcare', 'VIG': 'Dividend Growth', 'VOO': 'S&P 500', 'VTI': 'Total Market',
            'AGG': 'Aggregate Bonds', 'BND': 'Total Bond Market', 'TLT': 'Long Treasury',
            'IEF': 'Intermediate Treasury', 'SHY': 'Short Treasury', 'TIP': 'TIPS',
            'HYG': 'High Yield Corporate', 'LQD': 'Investment Grade Corporate',
            'USFR': 'Ultra Short Treasury', 'JPST': 'Ultra Short Income',
            'SHV': 'Short Treasury', 'VTIP': 'Short-Term TIPS', 'VGSH': 'Short Government',
            'BSV': 'Short-Term Corporate', 'JEPI': 'Equity Premium Income', 'IGSB': 'Short IG Corporate',
            'VCSH': 'Short Corporate'
        }
        
        if ticker in sector_mapping:
            sector = sector_mapping[ticker]
        elif ticker.startswith('TENOR_DGS') or ticker.startswith('IRLTLT01'):
            sector = 'Government Bond'
        else:
            sector = 'Other'
        
        holdings_data.append({
            'Rank': rank,
            'Ticker': ticker,
            'Weight': weight,
            'Weight_Pct': f"{weight:.4%}",
            'Asset_Type': asset_type,
            'Sector_Category': sector
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    # Generate markdown tables
    markdown_content = []
    
    # All Holdings Table
    markdown_content.append("## Complete Portfolio Holdings Table\n")
    markdown_content.append("### All 120 Instruments with Positive Allocations\n")
    markdown_content.append("| Rank | Ticker | Weight | Asset Type | Sector/Category |")
    markdown_content.append("|------|--------|--------|------------|----------------|")
    
    for _, row in holdings_df.iterrows():
        markdown_content.append(f"| {row['Rank']:3d} | {row['Ticker']:20s} | {row['Weight_Pct']:8s} | {row['Asset_Type']:15s} | {row['Sector_Category']} |")
    
    # Top 50 Holdings
    markdown_content.append("\n\n## Top 50 Holdings Detailed Analysis\n")
    markdown_content.append("| Rank | Ticker | Weight | Asset Type | Sector/Category |")
    markdown_content.append("|------|--------|--------|------------|----------------|")
    
    for _, row in holdings_df.head(50).iterrows():
        markdown_content.append(f"| {row['Rank']:3d} | {row['Ticker']:20s} | {row['Weight_Pct']:8s} | {row['Asset_Type']:15s} | {row['Sector_Category']} |")
    
    # Equity Holdings Table
    equity_holdings = holdings_df[holdings_df['Asset_Type'].str.contains('Equity')]
    markdown_content.append(f"\n\n## All Equity ETF Holdings ({len(equity_holdings)} positions)\n")
    markdown_content.append("| Rank | Ticker | Weight | Sector/Category |")
    markdown_content.append("|------|--------|--------|----------------|")
    
    for _, row in equity_holdings.iterrows():
        markdown_content.append(f"| {row['Rank']:3d} | {row['Ticker']:20s} | {row['Weight_Pct']:8s} | {row['Sector_Category']} |")
    
    # Bond Holdings Table
    bond_holdings = holdings_df[holdings_df['Asset_Type'].str.contains('Bond') | 
                               holdings_df['Asset_Type'].str.contains('Treasury')]
    markdown_content.append(f"\n\n## All Bond Holdings ({len(bond_holdings)} positions)\n")
    markdown_content.append("| Rank | Ticker | Weight | Sector/Category |")
    markdown_content.append("|------|--------|--------|----------------|")
    
    for _, row in bond_holdings.iterrows():
        markdown_content.append(f"| {row['Rank']:3d} | {row['Ticker']:50s} | {row['Weight_Pct']:8s} | {row['Sector_Category']} |")
    
    # Summary Statistics
    markdown_content.append(f"\n\n## Portfolio Statistics Summary\n")
    markdown_content.append("| Metric | Value |")
    markdown_content.append("|--------|-------|")
    markdown_content.append(f"| Total Positions | {len(holdings)} |")
    markdown_content.append(f"| Equity Positions | {len(equity_holdings)} |")
    markdown_content.append(f"| Bond Positions | {len(bond_holdings)} |")
    markdown_content.append(f"| Maximum Weight | {holdings.max():.4%} |")
    markdown_content.append(f"| Minimum Weight | {holdings.min():.6%} |")
    markdown_content.append(f"| Median Weight | {holdings.median():.4%} |")
    markdown_content.append(f"| Equity Total Weight | {equity_holdings['Weight'].sum():.2%} |")
    markdown_content.append(f"| Bond Total Weight | {bond_holdings['Weight'].sum():.2%} |")
    
    # Sector allocation breakdown
    sector_allocation = holdings_df.groupby('Sector_Category')['Weight'].sum().sort_values(ascending=False)
    markdown_content.append(f"\n\n## Sector Allocation Breakdown\n")
    markdown_content.append("| Sector/Category | Total Weight | Number of Holdings |")
    markdown_content.append("|----------------|-------------|-------------------|")
    
    for sector, weight in sector_allocation.items():
        count = holdings_df[holdings_df['Sector_Category'] == sector].shape[0]
        markdown_content.append(f"| {sector:30s} | {weight:8.2%} | {count:3d} |")
    
    # Save to file
    output_file = "DETAILED_PORTFOLIO_HOLDINGS_TABLES.md"
    with open(output_file, 'w') as f:
        f.write("# Detailed Portfolio Holdings Tables\n\n")
        f.write("This document provides comprehensive tables showing all portfolio holdings from the optimization analysis.\n\n")
        f.write("Generated from portfolio optimization analysis of 111 monthly scenarios across 120 instruments (June 2016 - August 2025).\n\n")
        f.write("---\n\n")
        f.write('\n'.join(markdown_content))
    
    print(f"✅ Generated detailed holdings tables in: {output_file}")
    print(f"📊 Total positions: {len(holdings)}")
    print(f"🏢 Equity positions: {len(equity_holdings)}")
    print(f"🏛️ Bond positions: {len(bond_holdings)}")
    print(f"📈 Top holding: {holdings.index[0]} ({holdings.iloc[0]:.4%})")
    
    return holdings_df, output_file

def generate_position_size_analysis():
    """Generate additional analysis of position size distribution"""
    
    # Load data and run optimization
    Rdf, keep_cols, etf_cols = load_inputs()
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=0.12)
    
    w_series = pd.Series(w, index=keep_cols).sort_values(ascending=False)
    holdings = w_series[w_series > 0]
    
    # Analyze position size distribution
    print("\n" + "="*60)
    print("POSITION SIZE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create buckets
    buckets = [
        (0.035, 0.040, "3.5-4.0%"),
        (0.030, 0.035, "3.0-3.5%"),
        (0.025, 0.030, "2.5-3.0%"),
        (0.020, 0.025, "2.0-2.5%"),
        (0.015, 0.020, "1.5-2.0%"),
        (0.010, 0.015, "1.0-1.5%"),
        (0.005, 0.010, "0.5-1.0%"),
        (0.001, 0.005, "0.1-0.5%"),
        (0.0001, 0.001, "0.01-0.1%"),
        (0.0, 0.0001, "<0.01%")
    ]
    
    for min_w, max_w, label in buckets:
        count = ((holdings >= min_w) & (holdings < max_w)).sum()
        total_weight = holdings[(holdings >= min_w) & (holdings < max_w)].sum()
        if count > 0:
            print(f"{label:>10s}: {count:3d} positions ({total_weight:6.2%} total weight)")
    
    print(f"\nAt 4% cap: {(holdings >= 0.04).sum()} positions")
    print(f"Above 3%:  {(holdings >= 0.03).sum()} positions")
    print(f"Above 2%:  {(holdings >= 0.02).sum()} positions")
    print(f"Above 1%:  {(holdings >= 0.01).sum()} positions")

if __name__ == "__main__":
    holdings_df, output_file = generate_comprehensive_holdings_tables()
    generate_position_size_analysis()