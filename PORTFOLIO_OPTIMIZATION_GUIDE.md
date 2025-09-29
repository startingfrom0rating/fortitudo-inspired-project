# Portfolio Weight Optimization with Fortitudo.tech

This document explains exactly what data is required to run portfolio weight optimization simulations using the fortitudo.tech package, and provides the necessary implementation to use ETF and bond lists for optimization.

## Data Requirements Summary

### For MeanCVaR Optimization (Recommended)
The `MeanCVaR` class requires:

1. **R**: Scenario P&L/returns matrix (S×I)
   - S = number of scenarios (trading days or simulated scenarios)
   - I = number of instruments (ETFs + bonds)
   - Each row represents one scenario
   - Each column represents one instrument's returns

2. **p**: Scenario probabilities (S×1) - *Optional*
   - Defaults to uniform probabilities (1/S for each scenario)
   - Can be adjusted using Entropy Pooling for views incorporation

3. **Constraints**:
   - **A, b**: Equality constraints (A: M×I, b: M×1)
   - **G, h**: Inequality constraints (G: N×I, h: N×1)
   - Default: budget constraint (sum of weights = 1) and long-only bounds

4. **v**: Relative market values (I×1) - *Optional*
   - Defaults to ones (standard weight-based optimization)
   - Used to define budget constraint: sum(v ⊙ weights) = 1

5. **alpha**: CVaR confidence level (float in (0,1)) - *Optional*
   - Default: 0.95 (95% CVaR)

### For MeanVariance Optimization
The `MeanVariance` class requires:

1. **mean**: Expected returns vector (I×1)
2. **covariance_matrix**: Covariance matrix (I×I)
3. **Constraints**: Same A, b, G, h, v as MeanCVaR

## Implementation

### Portfolio Data Infrastructure

The new `PortfolioDataLoader` class handles all data preparation:

```python
from fortitudo.tech import PortfolioDataLoader, create_sample_etf_list, create_sample_bond_list

# Initialize loader
loader = PortfolioDataLoader(base_currency='USD', lookback_years=5)

# Load ETF and bond lists
etf_list = create_sample_etf_list()  # Or load from your CSV/Excel files
bond_list = create_sample_bond_list()

# Prepare all optimization data
data = loader.prepare_optimization_data(etf_list=etf_list, bond_list=bond_list)
```

### Required Asset List Format

#### ETF List
Each ETF must have:
- **ticker**: ETF ticker symbol (e.g., 'SPY', 'QQQ')
- **name**: Full name (optional)
- **currency**: Currency (optional, defaults to USD)
- **group**: Asset group for analysis (optional)
- **weight_cap**: Maximum weight allowed (optional, defaults to 1.0)

#### Bond List
For bond ETFs:
- **ticker**: Bond ETF ticker (e.g., 'TLT', 'IEF')
- **name**: Full name (optional)
- **currency**: Currency (optional, defaults to USD)
- **group**: Bond category (optional)
- **duration_target**: Target duration in years (optional)
- **weight_cap**: Maximum weight allowed (optional, defaults to 1.0)

For individual bonds:
- **isin** or **cusip**: Bond identifier
- **coupon**: Coupon rate
- **maturity**: Maturity date
- **currency**: Currency (optional, defaults to USD)

### Complete Example

```python
import fortitudo.tech as ft
from fortitudo.tech import PortfolioDataLoader

# 1. Set up data loader
loader = PortfolioDataLoader(base_currency='USD', lookback_years=5)

# 2. Define your ETF and bond lists
etf_list = [
    {'ticker': 'SPY', 'name': 'SPDR S&P 500 ETF', 'group': 'US Equity', 'weight_cap': 0.30},
    {'ticker': 'EFA', 'name': 'iShares EAFE ETF', 'group': 'International', 'weight_cap': 0.25},
    # ... more ETFs
]

bond_list = [
    {'ticker': 'TLT', 'name': '20+ Year Treasury', 'group': 'Long-Term Gov', 'weight_cap': 0.40},
    {'ticker': 'IEF', 'name': '7-10 Year Treasury', 'group': 'Intermediate Gov', 'weight_cap': 0.30},
    # ... more bonds
]

# 3. Prepare optimization data
data = loader.prepare_optimization_data(etf_list=etf_list, bond_list=bond_list)

# 4. Run MeanCVaR optimization
cvar_optimizer = ft.MeanCVaR(
    R=data['R'], 
    G=data['G'], h=data['h'],  # Inequality constraints
    A=data['A'], b=data['b'],  # Equality constraints
    v=data['v'],               # Market values
    alpha=0.95                 # 95% CVaR
)

# Get optimal portfolio weights
optimal_weights = cvar_optimizer.efficient_portfolio()

# 5. Analyze results
portfolio_df = pd.DataFrame({
    'Asset': data['instrument_names'],
    'Weight (%)': optimal_weights.flatten() * 100,
    'Group': data['metadata']['asset_data']['group'].values
})

print(portfolio_df.sort_values('Weight (%)', ascending=False))
```

## Data Sources

### For Real Implementation
Replace the synthetic data generation with actual market data:

```python
# Example with yfinance (you need to install it)
import yfinance as yf

def fetch_real_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Modify the PortfolioDataLoader to use real data
```

### Missing Data Handling
The implementation includes:
- **Data alignment**: Automatically aligns different assets to common dates
- **Missing value handling**: Forward-fill or interpolation for small gaps
- **Currency conversion**: Framework for handling multi-currency portfolios
- **Return calculation**: Both log and simple returns supported

## Constraints Framework

### Default Constraints
- **Budget constraint**: Sum of weights = 1
- **Long-only**: All weights ≥ 0
- **Weight caps**: Individual asset weight limits

### Custom Constraints
Add group-level constraints:

```python
# Example: Equity allocation between 40-70%
equity_indices = [i for i, group in enumerate(groups) if 'Equity' in group]
equity_constraint = np.zeros(num_assets)
equity_constraint[equity_indices] = 1

# Add to inequality constraints
G_custom = np.vstack([G_default, equity_constraint, -equity_constraint])
h_custom = np.hstack([h_default, 0.7, -0.4])  # 40% ≤ equity ≤ 70%
```

## Risk Analysis

The framework provides comprehensive risk metrics:

```python
# Calculate risk metrics
p = np.ones((S, 1)) / S  # Equal scenario probabilities

portfolio_vol = ft.portfolio_vol(weights, R, p)
portfolio_var = ft.portfolio_var(weights, R, p, alpha=0.95)
portfolio_cvar = ft.portfolio_cvar(weights, R, p, alpha=0.95)
```

## Next Steps for Production

1. **Real data integration**: Replace synthetic data with yfinance, Bloomberg, or other providers
2. **Transaction costs**: Add bid-ask spreads and trading costs to optimization
3. **Risk factor models**: Implement factor-based risk models for better covariance estimation
4. **Backtesting**: Add historical simulation and performance attribution
5. **Monitoring**: Create dashboards for ongoing portfolio monitoring
6. **Rebalancing**: Implement systematic rebalancing logic

This implementation provides a complete framework for portfolio weight optimization using real ETF and bond data, with all necessary data validation and constraint handling.