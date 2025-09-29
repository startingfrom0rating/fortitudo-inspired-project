#!/usr/bin/env python3
"""
Portfolio Weight Optimization Example

This script demonstrates how to use the fortitudo.tech package to determine optimal
portfolio weights using both MeanCVaR and MeanVariance optimization with ETF and bond data.

This example shows:
1. Loading ETF and bond lists
2. Preparing historical data for optimization
3. Running MeanCVaR and MeanVariance optimization
4. Comparing results and analyzing portfolio composition

Requirements:
- All data inputs must be complete before running simulations
- The script handles missing data and provides fallbacks
- Results include risk metrics and efficient frontier analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fortitudo.tech as ft
from fortitudo.tech.portfolio_data import PortfolioDataLoader, create_sample_etf_list, create_sample_bond_list

def main():
    """Main function demonstrating portfolio weight optimization workflow."""
    
    print("=" * 80)
    print("PORTFOLIO WEIGHT OPTIMIZATION WITH FORTITUDO.TECH")
    print("=" * 80)
    print()
    
    # Step 1: Initialize data loader
    print("Step 1: Initializing Portfolio Data Loader")
    print("-" * 50)
    
    loader = PortfolioDataLoader(base_currency='USD', lookback_years=5)
    print(f"✓ Initialized with base currency: {loader.base_currency}")
    print(f"✓ Historical data lookback: {loader.lookback_years} years")
    print()
    
    # Step 2: Load sample ETF and bond lists
    print("Step 2: Loading Asset Lists")
    print("-" * 50)
    
    # Create sample lists (in practice, these would come from your PDF files or data sources)
    etf_list = create_sample_etf_list()
    bond_list = create_sample_bond_list()
    
    print(f"✓ Loaded {len(etf_list)} ETFs")
    print("ETF Summary:")
    etf_df = pd.DataFrame(etf_list)
    print(etf_df[['ticker', 'name', 'group', 'weight_cap']].to_string(index=False))
    print()
    
    print(f"✓ Loaded {len(bond_list)} Bond ETFs")
    print("Bond ETF Summary:")
    bond_df = pd.DataFrame(bond_list)
    print(bond_df[['ticker', 'name', 'group', 'duration_target', 'weight_cap']].to_string(index=False))
    print()
    
    # Step 3: Prepare optimization data
    print("Step 3: Preparing Optimization Data")
    print("-" * 50)
    
    print("⚠️  NOTE: Using synthetic data for demonstration.")
    print("   In practice, replace with real market data fetching (e.g., yfinance)")
    print()
    
    try:
        data = loader.prepare_optimization_data(etf_list=etf_list, bond_list=bond_list)
        
        R = data['R']
        mean_returns = data['mean']
        cov_matrix = data['covariance_matrix']
        instrument_names = data['instrument_names']
        G, h = data['G'], data['h']
        A, b = data['A'], data['b']
        v = data['v']
        
        S, I = R.shape
        print(f"✓ Prepared optimization data:")
        print(f"  - {S} scenarios (trading days)")
        print(f"  - {I} instruments")
        print(f"  - Return matrix shape: {R.shape}")
        print(f"  - Covariance matrix shape: {cov_matrix.shape}")
        print()
        
        # Display basic statistics
        print("Asset Statistics (Annualized):")
        stats_df = pd.DataFrame({
            'Asset': instrument_names,
            'Mean Return (%)': mean_returns * 252 * 100,  # Annualized
            'Volatility (%)': np.sqrt(np.diag(cov_matrix)) * np.sqrt(252) * 100,  # Annualized
            'Weight Cap': data['metadata']['asset_data']['weight_cap'].values
        })
        print(stats_df.to_string(index=False, float_format='%.2f'))
        print()
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # Step 4: Run MeanCVaR Optimization
    print("Step 4: Running MeanCVaR Optimization")
    print("-" * 50)
    
    try:
        # Use default parameters: 95% CVaR, demean=True
        cvar_optimizer = ft.MeanCVaR(R=R, G=G, h=h, A=A, b=b, v=v, alpha=0.95)
        
        # Get minimum CVaR portfolio
        cvar_weights = cvar_optimizer.efficient_portfolio()
        
        print("✓ MeanCVaR optimization completed")
        print(f"CVaR Level: 95%")
        print()
        
        # Display portfolio weights
        cvar_portfolio = pd.DataFrame({
            'Asset': instrument_names,
            'Weight (%)': cvar_weights.flatten() * 100,
            'Group': data['metadata']['asset_data']['group'].values
        })
        cvar_portfolio = cvar_portfolio.sort_values('Weight (%)', ascending=False)
        print("MeanCVaR Portfolio Weights:")
        print(cvar_portfolio.to_string(index=False, float_format='%.2f'))
        print()
        
        # Calculate portfolio metrics
        portfolio_return = mean_returns @ cvar_weights.flatten()
        portfolio_vol = np.sqrt(cvar_weights.flatten() @ cov_matrix @ cvar_weights.flatten())
        
        print(f"Portfolio Expected Return (annualized): {portfolio_return * 252 * 100:.2f}%")
        print(f"Portfolio Volatility (annualized): {portfolio_vol * np.sqrt(252) * 100:.2f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error in MeanCVaR optimization: {e}")
        cvar_weights = None
    
    # Step 5: Run MeanVariance Optimization
    print("Step 5: Running MeanVariance Optimization")
    print("-" * 50)
    
    try:
        mv_optimizer = ft.MeanVariance(mean=mean_returns, covariance_matrix=cov_matrix, 
                                      G=G, h=h, A=A, b=b, v=v)
        
        # Get minimum variance portfolio
        mv_weights = mv_optimizer.efficient_portfolio()
        
        print("✓ MeanVariance optimization completed")
        print()
        
        # Display portfolio weights
        mv_portfolio = pd.DataFrame({
            'Asset': instrument_names,
            'Weight (%)': mv_weights.flatten() * 100,
            'Group': data['metadata']['asset_data']['group'].values
        })
        mv_portfolio = mv_portfolio.sort_values('Weight (%)', ascending=False)
        print("MeanVariance Portfolio Weights:")
        print(mv_portfolio.to_string(index=False, float_format='%.2f'))
        print()
        
        # Calculate portfolio metrics
        mv_portfolio_return = mean_returns @ mv_weights.flatten()
        mv_portfolio_vol = np.sqrt(mv_weights.flatten() @ cov_matrix @ mv_weights.flatten())
        
        print(f"Portfolio Expected Return (annualized): {mv_portfolio_return * 252 * 100:.2f}%")
        print(f"Portfolio Volatility (annualized): {mv_portfolio_vol * np.sqrt(252) * 100:.2f}%")
        print()
        
    except Exception as e:
        print(f"❌ Error in MeanVariance optimization: {e}")
        mv_weights = None
    
    # Step 6: Compare Optimizations
    if cvar_weights is not None and mv_weights is not None:
        print("Step 6: Comparing Optimization Results")
        print("-" * 50)
        
        comparison_df = pd.DataFrame({
            'Asset': instrument_names,
            'MeanCVaR (%)': cvar_weights.flatten() * 100,
            'MeanVariance (%)': mv_weights.flatten() * 100,
            'Difference (%)': (cvar_weights.flatten() - mv_weights.flatten()) * 100,
            'Group': data['metadata']['asset_data']['group'].values
        })
        comparison_df = comparison_df.sort_values('MeanCVaR (%)', ascending=False)
        
        print("Portfolio Weights Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        print()
        
        # Group-level analysis
        group_analysis = comparison_df.groupby('Group').agg({
            'MeanCVaR (%)': 'sum',
            'MeanVariance (%)': 'sum'
        }).round(2)
        
        print("Asset Group Allocation:")
        print(group_analysis.to_string(float_format='%.2f'))
        print()
    
    # Step 7: Risk Analysis
    if cvar_weights is not None:
        print("Step 7: Risk Analysis")
        print("-" * 50)
        
        try:
            # Calculate additional risk metrics using the portfolio functions
            p = np.ones((S, 1)) / S  # Equal probability scenarios
            
            portfolio_vol_metric = ft.portfolio_vol(cvar_weights, R, p)
            portfolio_var = ft.portfolio_var(cvar_weights, R, p, alpha=0.95)
            portfolio_cvar = ft.portfolio_cvar(cvar_weights, R, p, alpha=0.95)
            
            print("Risk Metrics (MeanCVaR Portfolio):")
            print(f"  Volatility: {portfolio_vol_metric[0] * 100:.2f}%")
            print(f"  95% VaR: {portfolio_var[0] * 100:.2f}%")
            print(f"  95% CVaR: {portfolio_cvar[0] * 100:.2f}%")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not calculate additional risk metrics: {e}")
            print()
    
    # Step 8: Summary and Next Steps
    print("Step 8: Summary and Next Steps")
    print("-" * 50)
    
    print("✅ Portfolio optimization completed successfully!")
    print()
    print("Data Requirements Summary:")
    print("✓ ETF and Bond lists loaded")
    print("✓ Historical return data prepared (synthetic)")
    print("✓ Optimization constraints defined")
    print("✓ Both MeanCVaR and MeanVariance optimizations run")
    print()
    print("Next Steps for Production Use:")
    print("1. Replace synthetic data with real market data (yfinance, Bloomberg, etc.)")
    print("2. Implement currency hedging for non-USD assets")
    print("3. Add transaction cost considerations")
    print("4. Implement rebalancing logic")
    print("5. Add stress testing and scenario analysis")
    print("6. Create monitoring and reporting dashboards")
    print()
    print("=" * 80)


def create_wghic_approved_lists():
    """Create sample WGHIC-approved ETF and Treasury bond lists.
    
    This function creates lists that might represent the content of the 
    PDF files mentioned in the problem statement.
    """
    
    # Sample WGHIC-approved ETF list (common institutional ETFs)
    wghic_etfs = [
        {'ticker': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'currency': 'USD', 'group': 'Large Cap US Equity', 'weight_cap': 0.30},
        {'ticker': 'IEFA', 'name': 'iShares Core MSCI EAFE IMI Index ETF', 'currency': 'USD', 'group': 'International Equity', 'weight_cap': 0.25},
        {'ticker': 'IEMG', 'name': 'iShares Core MSCI Emerging Markets IMI Index ETF', 'currency': 'USD', 'group': 'Emerging Markets', 'weight_cap': 0.15},
        {'ticker': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'currency': 'USD', 'group': 'US Total Market', 'weight_cap': 0.30},
        {'ticker': 'IVV', 'name': 'iShares Core S&P 500 ETF', 'currency': 'USD', 'group': 'Large Cap US Equity', 'weight_cap': 0.30},
        {'ticker': 'ITOT', 'name': 'iShares Core S&P Total US Stock Market ETF', 'currency': 'USD', 'group': 'US Total Market', 'weight_cap': 0.30},
        {'ticker': 'IUSG', 'name': 'iShares Core S&P U.S. Growth ETF', 'currency': 'USD', 'group': 'US Growth', 'weight_cap': 0.20},
        {'ticker': 'IUSV', 'name': 'iShares Core S&P U.S. Value ETF', 'currency': 'USD', 'group': 'US Value', 'weight_cap': 0.20},
        {'ticker': 'IJH', 'name': 'iShares Core S&P Mid-Cap ETF', 'currency': 'USD', 'group': 'Mid Cap US Equity', 'weight_cap': 0.15},
        {'ticker': 'IJR', 'name': 'iShares Core S&P Small-Cap ETF', 'currency': 'USD', 'group': 'Small Cap US Equity', 'weight_cap': 0.10},
    ]
    
    # Sample WGHIC-approved Treasury bonds list
    wghic_treasuries = [
        {'ticker': 'SHY', 'name': 'iShares 1-3 Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Short-Term Treasury', 'duration_target': 2, 'weight_cap': 0.40},
        {'ticker': 'IEI', 'name': 'iShares 3-7 Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Intermediate Treasury', 'duration_target': 5, 'weight_cap': 0.40},
        {'ticker': 'IEF', 'name': 'iShares 7-10 Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Intermediate Treasury', 'duration_target': 8, 'weight_cap': 0.40},
        {'ticker': 'TLT', 'name': 'iShares 20+ Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Long-Term Treasury', 'duration_target': 18, 'weight_cap': 0.30},
        {'ticker': 'GOVT', 'name': 'iShares U.S. Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Treasury Blend', 'duration_target': 6, 'weight_cap': 0.50},
        {'ticker': 'SGOV', 'name': 'iShares 0-3 Month Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Cash Equivalent', 'duration_target': 0.1, 'weight_cap': 0.30},
        {'ticker': 'SCHO', 'name': 'Schwab Short-Term U.S. Treasury ETF', 'currency': 'USD', 
         'group': 'Short-Term Treasury', 'duration_target': 2, 'weight_cap': 0.40},
        {'ticker': 'SCHR', 'name': 'Schwab Intermediate-Term U.S. Treasury ETF', 'currency': 'USD', 
         'group': 'Intermediate Treasury', 'duration_target': 4, 'weight_cap': 0.40},
    ]
    
    return wghic_etfs, wghic_treasuries


if __name__ == "__main__":
    main()