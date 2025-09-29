# Mathematical Appendix: Portfolio Optimization Framework

## A. Optimization Problem Formulation

### A.1 Quadratic Programming Problem

The mean-variance optimization problem is formulated as:

```
minimize: f(w) = (1/2) * w^T * Σ * w
subject to: g_i(w) = G_i * w - h_i ≤ 0, i = 1, ..., m (inequality constraints)
           e_j(w) = A_j * w - b_j = 0, j = 1, ..., p (equality constraints)
```

Where:
- **w** ∈ ℝ^n: Portfolio weight vector (n = 120 instruments)
- **Σ** ∈ ℝ^(n×n): Covariance matrix of returns (positive semi-definite)
- **G** ∈ ℝ^(m×n): Inequality constraint matrix
- **h** ∈ ℝ^m: Inequality constraint bounds
- **A** ∈ ℝ^(p×n): Equality constraint matrix  
- **b** ∈ ℝ^p: Equality constraint bounds

### A.2 Constraint Matrix Construction

#### Long-Only Constraints (w ≥ 0)
```
G_1 = -I_{120×120}
h_1 = 0_{120×1}
```

#### Position Size Limits (w_i ≤ 0.04)
```
G_2 = I_{120×120}
h_2 = 0.04 * 1_{120×1}
```

#### Equity Minimum Constraint (Σw_equity ≥ 0.50)
```
G_3 = -[0, ..., 0, 1, 1, ..., 1, 0, ..., 0]_{1×120}  (1s for equity positions)
h_3 = -0.50
```

#### Budget Constraint (Σw = 1)
```
A = 1_{1×120}^T
b = 1
```

### A.3 Regularized Covariance Matrix

To ensure numerical stability and positive definiteness:

```
Σ_regularized = Σ_sample + λI
```

Where:
- Σ_sample = sample covariance matrix with Bessel correction
- λ = 1e-4 (regularization parameter)
- I = identity matrix

## B. Efficient Frontier Construction

### B.1 Parametric Approach

For target return r_target, solve:

```
minimize: w^T * Σ * w
subject to: μ^T * w = r_target
           Gw ≤ h
           Aw = b
```

This generates the efficient frontier as a parametric curve in (σ, μ) space.

### B.2 Return Target Grid

```python
r_min = min feasible return
r_max = max feasible return  
r_targets = r_min + k * Δr, k = 0, 1, ..., K
where Δr = 0.002 (monthly)
```

### B.3 Sharpe Ratio Maximization

The optimal portfolio maximizes:

```
SR(w) = (μ^T * w - r_f) / √(w^T * Σ * w)
```

Subject to the volatility constraint:
```
√(w^T * Σ * w) ≤ σ_max = 0.12/√12 ≈ 0.03464 (monthly)
```

## C. Risk Metrics Calculation

### C.1 Portfolio Volatility

```
σ_p = √(w^T * Σ * w)
```

### C.2 Value at Risk (VaR)

For confidence level α = 0.95:

```
VaR_α = -F^(-1)(1-α)
```

Where F^(-1) is the inverse CDF of portfolio returns.

### C.3 Conditional Value at Risk (CVaR)

```
CVaR_α = E[L | L ≥ VaR_α]
```

Where L represents portfolio losses.

### C.4 Scenario-Based Implementation

Given S = 111 scenarios with equal probability p_s = 1/S:

```python
portfolio_returns = R @ w  # (S × 1) vector
sorted_returns = sort(portfolio_returns)
var_index = floor((1-α) * S)
VaR = -sorted_returns[var_index]
CVaR = -mean(sorted_returns[1:var_index])
```

## D. Numerical Implementation

### D.1 CVXOPT Solver Interface

The problem is passed to CVXOPT as:

```python
solution = qp(P=P, q=q, G=G, h=h, A=A, b=b)
```

Where:
- P = 1000 * Σ (scaled for numerical precision)
- q = 0_n (no linear term in objective)
- G, h = inequality constraints
- A, b = equality constraints

### D.2 Convergence Criteria

The QP solver uses:
- **Tolerance**: 1e-8 for primal and dual feasibility
- **Maximum iterations**: Default CVXOPT limits
- **Scaling**: Objective function scaled by 1000 for stability

### D.3 Constraint Verification

Post-optimization checks:
```python
assert np.all(w >= -1e-8)  # Long-only
assert abs(np.sum(w) - 1.0) < 1e-6  # Budget
assert np.max(w) <= 0.04 + 1e-6  # Position limits
assert equity_allocation >= 0.50 - 1e-6  # Equity minimum
```

## E. Asset Classification Algorithm

### E.1 Bond ETF Identification

```python
bond_etfs = {
    'AGG', 'BND', 'BNDX', 'BSV', 'CORP', 'EMB', 'EMLC', 'GOVT', 
    'HYG', 'HYLS', 'IEF', 'IGSB', 'JNK', 'LOAN', 'LQD', 'MBB', 
    'MUB', 'MUNI', 'SHV', 'SHY', 'TIP', 'TIPS', 'TLT', 'TR', 
    'TRST', 'USFR', 'VCIT', 'VCSH', 'VGSH', 'VTIP', 'JEPI', 'JPST'
}
```

### E.2 Treasury Bond Identification

```python
def is_treasury(ticker):
    return ticker.startswith('DGS') or 'IRLTLT01' in ticker
```

### E.3 Classification Logic

```python
for i, ticker in enumerate(instruments):
    if is_treasury(ticker):
        bond_indices.append(i)
    elif ticker in bond_etfs:
        bond_indices.append(i)
    else:
        equity_indices.append(i)
```

## F. Performance Metrics

### F.1 Annualization

Monthly to annual conversion:
```
μ_annual = (1 + μ_monthly)^12 - 1
σ_annual = σ_monthly * √12
SR_annual = SR_monthly * √12
```

### F.2 Maximum Drawdown

```python
cumulative_returns = np.cumprod(1 + returns)
running_max = np.maximum.accumulate(cumulative_returns)
drawdowns = (cumulative_returns - running_max) / running_max
max_drawdown = np.min(drawdowns)
```

### F.3 Information Ratio

```
IR = (r_p - r_b) / σ(r_p - r_b)
```

Where r_b is the benchmark return.

## G. Computational Complexity

### G.1 Time Complexity

- **QP solver**: O(n³) for dense problems
- **Frontier construction**: O(K * n³) for K target returns
- **Constraint evaluation**: O(mn) for m constraints

### G.2 Space Complexity

- **Covariance matrix**: O(n²) = O(14,400) for n=120
- **Constraint matrices**: O(mn) = O(241 * 120) = O(28,920)
- **Return data**: O(Sn) = O(111 * 120) = O(13,320)

### G.3 Numerical Precision

- **Float precision**: 64-bit double precision
- **Constraint tolerance**: 1e-6 to 1e-8
- **Regularization**: 1e-4 * I for covariance matrix

This mathematical framework ensures robust, efficient portfolio optimization with comprehensive risk management and constraint satisfaction.