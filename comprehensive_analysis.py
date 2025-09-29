"""
Generate additional visualizations and analysis for the comprehensive report
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.generate_plots_and_report import load_inputs, classify_indices, optimize
from fortitudo.tech.functions import simulation_moments, covariance_matrix

def create_comprehensive_analysis():
    """Create additional analysis and visualizations for the final report"""
    
    # Load data and run optimization
    Rdf, keep_cols, etf_cols = load_inputs()
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=0.12)
    
    # Create plots directory
    os.makedirs("data/plots", exist_ok=True)
    
    # 1. Asset Allocation Pie Chart
    plt.figure(figsize=(10, 8))
    
    w_series = pd.Series(w, index=keep_cols)
    equity_weight = w[equity_idx].sum() if equity_idx else 0.0
    bond_weight = w[bond_idx].sum() if bond_idx else 0.0
    
    # Top holdings breakdown
    top_holdings = w_series.sort_values(ascending=False).head(10)
    other_weight = 1.0 - top_holdings.sum()
    
    # Pie chart data
    pie_data = list(top_holdings.values) + [other_weight]
    pie_labels = list(top_holdings.index) + [f'Others ({len(w_series[w_series > 0]) - 10} positions)']
    
    plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Allocation - Top 10 Holdings + Others', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('data/plots/portfolio_allocation_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Asset Class Allocation
    plt.figure(figsize=(8, 6))
    asset_classes = ['Equity ETFs', 'Bonds']
    allocations = [equity_weight, bond_weight]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = plt.bar(asset_classes, allocations, color=colors, alpha=0.8)
    plt.ylabel('Allocation (%)', fontsize=12)
    plt.title('Asset Class Allocation', fontsize=14, fontweight='bold')
    plt.ylim(0, 0.6)
    
    # Add percentage labels on bars
    for bar, alloc in zip(bars, allocations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{alloc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line at 50%
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Min Equity Requirement (50%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/asset_class_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Risk-Return Scatter Plot
    plt.figure(figsize=(10, 7))
    
    # Plot efficient frontier
    plt.plot(vols, means, 'bo-', label='Efficient Frontier', linewidth=2, markersize=6)
    
    # Highlight optimal portfolio
    plt.scatter(vols[best_i], means[best_i], color='red', s=200, marker='*', 
               label=f'Optimal Portfolio (Sharpe: {sharpe[best_i]:.3f})', zorder=5)
    
    # Add risk cap line
    risk_cap_monthly = 0.12 / np.sqrt(12)
    plt.axvline(x=risk_cap_monthly, color='orange', linestyle='--', alpha=0.7, 
               label=f'Risk Cap ({risk_cap_monthly:.2%} monthly)')
    
    plt.xlabel('Volatility (Monthly)', fontsize=12)
    plt.ylabel('Expected Return (Monthly)', fontsize=12)
    plt.title('Efficient Frontier and Optimal Portfolio', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/plots/efficient_frontier_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sector Allocation Analysis
    plt.figure(figsize=(12, 8))
    
    # Identify sector ETFs
    sector_mapping = {
        'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare', 'XLI': 'Industrials',
        'XLE': 'Energy', 'XLU': 'Utilities', 'XLB': 'Materials', 'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary', 'XLC': 'Communication Services',
        'QQQ': 'Technology (NASDAQ)', 'SMH': 'Semiconductors', 'VGT': 'Technology',
        'VHT': 'Healthcare', 'VIG': 'Dividend Growth', 'VOO': 'S&P 500', 'VTI': 'Total Market'
    }
    
    # Calculate sector allocations
    sector_weights = {}
    for ticker in etf_cols:
        if ticker in keep_cols and w[keep_cols.index(ticker)] > 0:
            sector = sector_mapping.get(ticker, 'Other Equity')
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += w[keep_cols.index(ticker)]
    
    # Add bond allocation
    sector_weights['Bonds'] = bond_weight
    
    # Create horizontal bar chart
    sectors = list(sector_weights.keys())
    weights = list(sector_weights.values())
    
    plt.barh(sectors, weights, alpha=0.8)
    plt.xlabel('Allocation', fontsize=12)
    plt.title('Portfolio Allocation by Sector/Asset Class', fontsize=14, fontweight='bold')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    plt.tight_layout()
    plt.savefig('data/plots/sector_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Risk Metrics Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio returns histogram
    R = Rdf[keep_cols].values
    portfolio_returns = R @ w
    ax1.hist(portfolio_returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(portfolio_returns.mean(), color='red', linestyle='--', label=f'Mean: {portfolio_returns.mean():.3f}')
    ax1.set_xlabel('Monthly Returns')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Portfolio Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    ax2.plot(Rdf.index, cumulative_returns, linewidth=2, color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Value')
    ax2.set_title('Portfolio Growth ($1 Initial Investment)')
    ax2.grid(True, alpha=0.3)
    
    # Risk metrics comparison
    metrics_names = ['Return\n(Ann.)', 'Volatility\n(Ann.)', 'Sharpe\n(Ann.)', 'Max\nDrawdown']
    portfolio_metrics = [
        (1 + means[best_i])**12 - 1,
        vols[best_i] * np.sqrt(12),
        sharpe[best_i] * np.sqrt(12),
        -np.min(portfolio_returns)  # Simplified max drawdown
    ]
    
    bars = ax3.bar(metrics_names, portfolio_metrics, alpha=0.8, color=['green', 'orange', 'blue', 'red'])
    ax3.set_title('Key Portfolio Metrics')
    ax3.set_ylabel('Value')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, portfolio_metrics)):
        if 'Return' in metrics_names[i] or 'Volatility' in metrics_names[i] or 'Max' in metrics_names[i]:
            label = f'{value:.2%}'
        else:
            label = f'{value:.2f}'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(portfolio_metrics)*0.01,
                label, ha='center', va='bottom', fontweight='bold')
    
    # Position size distribution
    position_sizes = w[w > 0]
    ax4.hist(position_sizes, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(0.04, color='red', linestyle='--', label='4% Cap')
    ax4.set_xlabel('Position Size')
    ax4.set_ylabel('Number of Positions')
    ax4.set_title('Position Size Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/plots/risk_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Generated comprehensive analysis visualizations:")
    print("   • Portfolio allocation pie chart")
    print("   • Asset class allocation bar chart") 
    print("   • Detailed efficient frontier plot")
    print("   • Sector allocation analysis")
    print("   • Risk metrics dashboard")
    print(f"\n💾 All visualizations saved to: {os.path.abspath('data/plots/')}")

def generate_summary_statistics():
    """Generate summary statistics table for the report"""
    
    Rdf, keep_cols, etf_cols = load_inputs()
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=0.12)
    
    # Get moments for individual assets
    R = Rdf[keep_cols].values
    moments = simulation_moments(R)
    
    # Portfolio statistics
    portfolio_return = means[best_i]
    portfolio_vol = vols[best_i]
    portfolio_sharpe = sharpe[best_i]
    
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nDATA CHARACTERISTICS:")
    print(f"  • Time Period: {Rdf.index[0].strftime('%Y-%m-%d')} to {Rdf.index[-1].strftime('%Y-%m-%d')}")
    print(f"  • Monthly Observations: {len(Rdf)}")
    print(f"  • Total Instruments: {len(keep_cols)}")
    print(f"  • ETFs: {len(etf_cols)}")
    print(f"  • Treasury Bonds: {len(keep_cols) - len(etf_cols)}")
    
    print(f"\nOPTIMAL PORTFOLIO METRICS:")
    print(f"  • Expected Return (Monthly): {portfolio_return:.3%}")
    print(f"  • Expected Return (Annual): {(1+portfolio_return)**12-1:.2%}")
    print(f"  • Volatility (Monthly): {portfolio_vol:.3%}")
    print(f"  • Volatility (Annual): {portfolio_vol*np.sqrt(12):.2%}")
    print(f"  • Sharpe Ratio (Monthly): {portfolio_sharpe:.3f}")
    print(f"  • Sharpe Ratio (Annual): {portfolio_sharpe*np.sqrt(12):.3f}")
    
    equity_weight = w[equity_idx].sum() if equity_idx else 0.0
    bond_weight = w[bond_idx].sum() if bond_idx else 0.0
    
    print(f"\nASSET ALLOCATION:")
    print(f"  • Equity ETFs: {equity_weight:.1%}")
    print(f"  • Bonds: {bond_weight:.1%}")
    print(f"  • Number of Positions: {len(w[w > 0])}")
    print(f"  • Maximum Position Size: {w.max():.2%}")
    print(f"  • Minimum Position Size: {w[w > 0].min():.4%}")
    
    print(f"\nCONSTRAINT VERIFICATION:")
    print(f"  • Long-only satisfied: {(w >= 0).all()}")
    print(f"  • Sum to 1.0: {abs(w.sum() - 1.0) < 1e-6}")
    print(f"  • Max 4% position: {w.max() <= 0.04 + 1e-6}")
    print(f"  • Min 50% equity: {equity_weight >= 0.5 - 1e-6}")
    print(f"  • Risk cap (12% ann): {portfolio_vol*np.sqrt(12) <= 0.12 + 1e-6}")
    
    # Top holdings
    w_series = pd.Series(w, index=keep_cols).sort_values(ascending=False)
    print(f"\nTOP 10 HOLDINGS:")
    for i, (ticker, weight) in enumerate(w_series.head(10).items()):
        asset_type = "Equity" if ticker in etf_cols else "Bond"
        print(f"  {i+1:2d}. {ticker:8s} {weight:6.2%} ({asset_type})")

if __name__ == "__main__":
    create_comprehensive_analysis()
    generate_summary_statistics()