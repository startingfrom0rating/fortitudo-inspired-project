# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2025 Fortitudo Technologies.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings


class PortfolioDataLoader:
    """Class for loading and preparing portfolio optimization data from ETF and bond lists.
    
    This class handles the data requirements for both MeanCVaR and MeanVariance optimization
    by providing methods to load asset lists, fetch historical data, and prepare the required
    matrices and vectors for optimization.
    """

    def __init__(self, base_currency: str = 'USD', lookback_years: int = 5):
        """Initialize the portfolio data loader.
        
        Args:
            base_currency: Base currency for optimization (default: USD)
            lookback_years: Number of years of historical data to use (default: 5)
        """
        self.base_currency = base_currency
        self.lookback_years = lookback_years
        self.etf_data = None
        self.bond_data = None
        self.return_data = None
        
    def load_etf_list(self, etf_list: Union[List[Dict], pd.DataFrame, str]) -> pd.DataFrame:
        """Load ETF list from various sources.
        
        Args:
            etf_list: ETF list as list of dicts, DataFrame, or file path
            
        Returns:
            DataFrame with ETF information including required columns
            
        Required columns: ticker, name (optional), currency (optional), group (optional)
        """
        if isinstance(etf_list, str):
            # Load from file
            if etf_list.endswith('.csv'):
                df = pd.read_csv(etf_list)
            elif etf_list.endswith('.xlsx'):
                df = pd.read_excel(etf_list)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")
        elif isinstance(etf_list, list):
            df = pd.DataFrame(etf_list)
        elif isinstance(etf_list, pd.DataFrame):
            df = etf_list.copy()
        else:
            raise ValueError("etf_list must be a list of dicts, DataFrame, or file path")
            
        # Validate required columns
        if 'ticker' not in df.columns:
            raise ValueError("ETF list must contain 'ticker' column")
            
        # Add default values for optional columns
        if 'name' not in df.columns:
            df['name'] = df['ticker']
        if 'currency' not in df.columns:
            df['currency'] = self.base_currency
        if 'group' not in df.columns:
            df['group'] = 'ETF'
        if 'weight_cap' not in df.columns:
            df['weight_cap'] = 1.0  # No weight cap by default
            
        self.etf_data = df
        return df
        
    def load_bond_list(self, bond_list: Union[List[Dict], pd.DataFrame, str]) -> pd.DataFrame:
        """Load bond list from various sources.
        
        Args:
            bond_list: Bond list as list of dicts, DataFrame, or file path
            
        Returns:
            DataFrame with bond information including required columns
            
        For bond ETFs: ticker, name (optional), currency (optional), duration_target (optional)
        For individual bonds: isin/cusip, coupon, maturity, currency (optional)
        """
        if isinstance(bond_list, str):
            # Load from file
            if bond_list.endswith('.csv'):
                df = pd.read_csv(bond_list)
            elif bond_list.endswith('.xlsx'):
                df = pd.read_excel(bond_list)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")
        elif isinstance(bond_list, list):
            df = pd.DataFrame(bond_list)
        elif isinstance(bond_list, pd.DataFrame):
            df = bond_list.copy()
        else:
            raise ValueError("bond_list must be a list of dicts, DataFrame, or file path")
            
        # Determine if these are bond ETFs or individual bonds
        if 'ticker' in df.columns:
            # Bond ETFs
            if 'name' not in df.columns:
                df['name'] = df['ticker']
            if 'currency' not in df.columns:
                df['currency'] = self.base_currency
            if 'group' not in df.columns:
                df['group'] = 'Bond ETF'
            if 'duration_target' not in df.columns:
                df['duration_target'] = None
        elif 'isin' in df.columns or 'cusip' in df.columns:
            # Individual bonds
            if 'currency' not in df.columns:
                df['currency'] = self.base_currency
            if 'group' not in df.columns:
                df['group'] = 'Bond'
        else:
            raise ValueError("Bond list must contain either 'ticker' (for bond ETFs) or 'isin'/'cusip' (for individual bonds)")
            
        if 'weight_cap' not in df.columns:
            df['weight_cap'] = 1.0  # No weight cap by default
            
        self.bond_data = df
        return df
        
    def fetch_historical_data(self, tickers: List[str], start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical price data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data (default: lookback_years ago)
            end_date: End date for data (default: today)
            
        Returns:
            DataFrame with adjusted close prices
            
        Note: This is a placeholder method. In practice, you would use
        yfinance, pandas-datareader, or similar library to fetch data.
        """
        warnings.warn("fetch_historical_data is a placeholder. You need to implement "
                     "actual data fetching using yfinance or similar library.", 
                     UserWarning)
        
        # Placeholder implementation - returns synthetic data
        if start_date is None:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=self.lookback_years)
        if end_date is None:
            end_date = pd.Timestamp.now()
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate synthetic price data (for demonstration)
        np.random.seed(42)  # For reproducible results
        data = {}
        for ticker in tickers:
            initial_price = 100
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # ~12% annual return, 32% vol
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            data[ticker] = prices
            
        return pd.DataFrame(data, index=date_range)
        
    def calculate_returns(self, price_data: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """Calculate returns from price data.
        
        Args:
            price_data: DataFrame with price data
            method: 'log' for log returns or 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if method == 'log':
            return np.log(price_data / price_data.shift(1)).dropna()
        elif method == 'simple':
            return (price_data / price_data.shift(1) - 1).dropna()
        else:
            raise ValueError("method must be 'log' or 'simple'")
            
    def prepare_optimization_data(self, etf_list: Union[List[Dict], pd.DataFrame, str] = None,
                                bond_list: Union[List[Dict], pd.DataFrame, str] = None,
                                return_method: str = 'log') -> Dict:
        """Prepare all data required for portfolio optimization.
        
        Args:
            etf_list: ETF list data
            bond_list: Bond list data  
            return_method: Method for calculating returns ('log' or 'simple')
            
        Returns:
            Dictionary containing all optimization inputs:
            - R: Return scenarios matrix (S×I)
            - mean: Expected returns vector (I×1)
            - covariance_matrix: Covariance matrix (I×I)
            - instrument_names: List of instrument names
            - constraints: Default constraint matrices
            - metadata: Additional information about the assets
        """
        # Load asset lists
        all_assets = []
        all_tickers = []
        
        if etf_list is not None:
            etf_df = self.load_etf_list(etf_list)
            all_assets.append(etf_df)
            all_tickers.extend(etf_df['ticker'].tolist())
            
        if bond_list is not None:
            bond_df = self.load_bond_list(bond_list)
            if 'ticker' in bond_df.columns:
                all_assets.append(bond_df)
                all_tickers.extend(bond_df['ticker'].tolist())
            else:
                warnings.warn("Individual bonds not supported yet. Use bond ETFs with tickers.", 
                             UserWarning)
                
        if not all_assets:
            raise ValueError("Must provide at least one of etf_list or bond_list")
            
        # Combine asset data
        combined_assets = pd.concat(all_assets, ignore_index=True)
        
        # Fetch historical data
        price_data = self.fetch_historical_data(all_tickers)
        
        # Calculate returns
        returns = self.calculate_returns(price_data, method=return_method)
        self.return_data = returns
        
        # Prepare optimization inputs
        R = returns.values  # S×I matrix
        mean = returns.mean().values  # I×1 vector
        covariance_matrix = returns.cov().values  # I×I matrix
        instrument_names = returns.columns.tolist()
        
        # Create default constraints (long-only, budget constraint)
        I = len(instrument_names)
        
        # Inequality constraints: long-only bounds and weight caps
        G_list = []
        h_list = []
        
        # Long-only: w_i >= 0
        G_list.append(-np.eye(I))
        h_list.append(np.zeros(I))
        
        # Weight caps: w_i <= cap_i
        weight_caps = combined_assets['weight_cap'].values
        G_list.append(np.eye(I))
        h_list.append(weight_caps)
        
        G = np.vstack(G_list)
        h = np.concatenate(h_list)
        
        # Equality constraint: budget constraint (sum of weights = 1)
        A = np.ones((1, I))
        b = np.array([1.0])
        
        # Relative market values (for budget constraint)
        v = np.ones(I)
        
        return {
            'R': R,
            'mean': mean,
            'covariance_matrix': covariance_matrix,
            'instrument_names': instrument_names,
            'G': G,
            'h': h,
            'A': A,
            'b': b,
            'v': v,
            'metadata': {
                'asset_data': combined_assets,
                'price_data': price_data,
                'return_data': returns,
                'base_currency': self.base_currency,
                'lookback_years': self.lookback_years
            }
        }


def create_sample_etf_list() -> List[Dict]:
    """Create a sample ETF list for demonstration purposes.
    
    Returns:
        List of ETF dictionaries with common ETFs across asset classes
    """
    return [
        # US Equity ETFs
        {'ticker': 'SPY', 'name': 'SPDR S&P 500 ETF', 'currency': 'USD', 'group': 'US Equity', 'weight_cap': 0.25},
        {'ticker': 'QQQ', 'name': 'Invesco QQQ ETF', 'currency': 'USD', 'group': 'US Tech', 'weight_cap': 0.15},
        {'ticker': 'IWM', 'name': 'iShares Russell 2000 ETF', 'currency': 'USD', 'group': 'US Small Cap', 'weight_cap': 0.10},
        
        # International Equity ETFs  
        {'ticker': 'EFA', 'name': 'iShares MSCI EAFE ETF', 'currency': 'USD', 'group': 'Developed Markets', 'weight_cap': 0.20},
        {'ticker': 'EEM', 'name': 'iShares MSCI Emerging Markets ETF', 'currency': 'USD', 'group': 'Emerging Markets', 'weight_cap': 0.15},
        
        # Sector ETFs
        {'ticker': 'XLF', 'name': 'Financial Select Sector SPDR Fund', 'currency': 'USD', 'group': 'Financials', 'weight_cap': 0.10},
        {'ticker': 'XLE', 'name': 'Energy Select Sector SPDR Fund', 'currency': 'USD', 'group': 'Energy', 'weight_cap': 0.10},
        
        # Alternative Assets
        {'ticker': 'GLD', 'name': 'SPDR Gold Shares', 'currency': 'USD', 'group': 'Commodities', 'weight_cap': 0.05},
        {'ticker': 'VNQ', 'name': 'Vanguard Real Estate ETF', 'currency': 'USD', 'group': 'REITs', 'weight_cap': 0.10},
    ]


def create_sample_bond_list() -> List[Dict]:
    """Create a sample bond ETF list for demonstration purposes.
    
    Returns:
        List of bond ETF dictionaries across different duration and credit profiles
    """
    return [
        # Government Bonds
        {'ticker': 'TLT', 'name': 'iShares 20+ Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Long-Term Gov', 'duration_target': 18, 'weight_cap': 0.30},
        {'ticker': 'IEF', 'name': 'iShares 7-10 Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Intermediate Gov', 'duration_target': 8, 'weight_cap': 0.30},
        {'ticker': 'SHY', 'name': 'iShares 1-3 Year Treasury Bond ETF', 'currency': 'USD', 
         'group': 'Short-Term Gov', 'duration_target': 2, 'weight_cap': 0.30},
         
        # Corporate Bonds
        {'ticker': 'LQD', 'name': 'iShares iBoxx Investment Grade Corporate Bond ETF', 'currency': 'USD', 
         'group': 'Investment Grade Corp', 'duration_target': 8, 'weight_cap': 0.25},
        {'ticker': 'HYG', 'name': 'iShares iBoxx High Yield Corporate Bond ETF', 'currency': 'USD', 
         'group': 'High Yield Corp', 'duration_target': 4, 'weight_cap': 0.15},
         
        # International Bonds
        {'ticker': 'BNDX', 'name': 'Vanguard Total International Bond ETF', 'currency': 'USD', 
         'group': 'International Bonds', 'duration_target': 8, 'weight_cap': 0.20},
        {'ticker': 'EMB', 'name': 'iShares J.P. Morgan USD Emerging Markets Bond ETF', 'currency': 'USD', 
         'group': 'Emerging Market Bonds', 'duration_target': 7, 'weight_cap': 0.10},
    ]