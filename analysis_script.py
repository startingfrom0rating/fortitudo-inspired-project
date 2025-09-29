"""
Analyze the portfolio optimization results to understand the mathematical framework
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.generate_plots_and_report import load_inputs, classify_indices, optimize
from fortitudo.tech.functions import portfolio_vol, portfolio_var, portfolio_cvar, simulation_moments

def main():
    print("=== PORTFOLIO OPTIMIZATION ANALYSIS ===\n")
    
    # Load data
    Rdf, keep_cols, etf_cols = load_inputs()
    print(f"Data dimensions: {Rdf.shape[0]} scenarios x {len(keep_cols)} instruments")
    print(f"ETFs: {len(etf_cols)}, Treasury bonds: {len(keep_cols) - len(etf_cols)}")
    print(f"Date range: {Rdf.index[0]} to {Rdf.index[-1]}\n")
    
    # Run optimization
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=0.12)
    
    # Analyze results
    w_series = pd.Series(w, index=keep_cols).sort_values(ascending=False)
    
    print("=== PORTFOLIO COMPOSITION ===")
    
    # Calculate asset class allocations
    equity_weight = w[equity_idx].sum() if equity_idx else 0.0
    bond_weight = w[bond_idx].sum() if bond_idx else 0.0
    
    print(f"Equity allocation: {equity_weight:.1%}")
    print(f"Bond allocation: {bond_weight:.1%}")
    print(f"Total allocation: {w.sum():.1%}\n")
    
    # Show top holdings
    longs = w_series[w_series > 0]
    print("=== TOP 15 HOLDINGS ===")
    for i, (ticker, weight) in enumerate(longs.head(15).items()):
        asset_type = "Equity" if ticker in etf_cols and equity_idx and keep_cols.index(ticker) in equity_idx else "Bond"
        print(f"{i+1:2d}. {ticker:20s} {weight:6.2%} ({asset_type})")
    
    print(f"\nTotal positions with weight > 0: {len(longs)}")
    print(f"Max position size: {longs.max():.2%}")
    print(f"Min position size: {longs.min():.4%}\n")
    
    # Portfolio risk metrics
    R = Rdf[keep_cols].values
    p = np.ones((R.shape[0], 1)) / R.shape[0]
    
    w_col = w.reshape(-1, 1)
    port_vol = portfolio_vol(w_col, R, p)
    port_var_95 = portfolio_var(w_col, R, p, alpha=0.95)
    port_cvar_95 = portfolio_cvar(w_col, R, p, alpha=0.95)
    
    print("=== PORTFOLIO RISK METRICS (Monthly) ===")
    print(f"Expected return: {means[best_i]:.3%}")
    print(f"Volatility: {vols[best_i]:.3%}")
    print(f"Sharpe ratio: {sharpe[best_i]:.3f}")
    print(f"95% VaR: {port_var_95:.3%}")
    print(f"95% CVaR: {port_cvar_95:.3%}\n")
    
    print("=== PORTFOLIO RISK METRICS (Annualized) ===")
    print(f"Expected return: {(1+means[best_i])**12-1:.2%}")
    print(f"Volatility: {vols[best_i]*np.sqrt(12):.2%}")
    print(f"Sharpe ratio: {sharpe[best_i]*np.sqrt(12):.3f}")
    print(f"95% VaR: {port_var_95*np.sqrt(12):.2%}")
    print(f"95% CVaR: {port_cvar_95*np.sqrt(12):.2%}\n")
    
    # Show frontier characteristics
    print("=== EFFICIENT FRONTIER ANALYSIS ===")
    print(f"Number of frontier points: {len(means)}")
    print(f"Minimum volatility: {vols.min():.3%} (monthly)")
    print(f"Maximum return: {means.max():.3%} (monthly)")
    print(f"Best Sharpe portfolio volatility: {vols[best_i]:.3%} (monthly)")
    print(f"Risk cap applied: 12% annualized = {0.12/np.sqrt(12):.3%} monthly\n")
    
    # Show asset class breakdown of ETFs vs Bonds
    equity_holdings = [(ticker, w[keep_cols.index(ticker)]) for ticker in etf_cols 
                      if keep_cols.index(ticker) in equity_idx and w[keep_cols.index(ticker)] > 0]
    bond_holdings = [(ticker, w[keep_cols.index(ticker)]) for ticker in keep_cols 
                    if keep_cols.index(ticker) in bond_idx and w[keep_cols.index(ticker)] > 0]
    
    print("=== EQUITY ETF HOLDINGS ===")
    equity_holdings.sort(key=lambda x: x[1], reverse=True)
    for ticker, weight in equity_holdings[:10]:
        print(f"{ticker:20s} {weight:6.2%}")
    print(f"Total equity positions: {len(equity_holdings)}")
    
    print("\n=== BOND HOLDINGS ===")
    bond_holdings.sort(key=lambda x: x[1], reverse=True) 
    for ticker, weight in bond_holdings[:10]:
        print(f"{ticker:50s} {weight:6.2%}")
    print(f"Total bond positions: {len(bond_holdings)}")
    
    # Mathematical constraints summary
    print("\n=== MATHEMATICAL CONSTRAINTS ===")
    print("1. Long-only: w_i >= 0 for all i")
    print("2. Fully invested: sum(w_i) = 1")
    print("3. Position size limit: w_i <= 4% for all i")
    print("4. Equity minimum: sum(w_equity) >= 50%")
    print("5. Risk cap: annual volatility <= 12%")
    
    # Check constraint satisfaction
    print("\n=== CONSTRAINT VERIFICATION ===")
    print(f"All weights >= 0: {(w >= 0).all()}")
    print(f"Sum of weights = 1: {abs(w.sum() - 1.0) < 1e-6}")
    print(f"Max weight <= 4%: {w.max() <= 0.04 + 1e-6}")
    print(f"Equity weight >= 50%: {equity_weight >= 0.5 - 1e-6}")
    print(f"Annual vol <= 12%: {vols[best_i]*np.sqrt(12) <= 0.12 + 1e-6}")

if __name__ == "__main__":
    main()