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
import pytest
from fortitudo.tech.portfolio_data import PortfolioDataLoader, create_sample_etf_list, create_sample_bond_list


class TestPortfolioDataLoader:
    """Test suite for the PortfolioDataLoader class."""
    
    def test_initialization(self):
        """Test loader initialization with default and custom parameters."""
        # Default initialization
        loader = PortfolioDataLoader()
        assert loader.base_currency == 'USD'
        assert loader.lookback_years == 5
        
        # Custom initialization
        loader_custom = PortfolioDataLoader(base_currency='EUR', lookback_years=3)
        assert loader_custom.base_currency == 'EUR'
        assert loader_custom.lookback_years == 3
    
    def test_load_etf_list_from_dict_list(self):
        """Test loading ETF list from list of dictionaries."""
        loader = PortfolioDataLoader()
        etf_list = [
            {'ticker': 'SPY', 'name': 'SPDR S&P 500'},
            {'ticker': 'QQQ', 'name': 'Invesco QQQ'}
        ]
        
        df = loader.load_etf_list(etf_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'ticker' in df.columns
        assert 'name' in df.columns
        assert 'currency' in df.columns
        assert 'group' in df.columns
        assert 'weight_cap' in df.columns
        
        # Check default values
        assert all(df['currency'] == 'USD')
        assert all(df['group'] == 'ETF')
        assert all(df['weight_cap'] == 1.0)
    
    def test_load_etf_list_from_dataframe(self):
        """Test loading ETF list from DataFrame."""
        loader = PortfolioDataLoader()
        etf_df = pd.DataFrame({
            'ticker': ['SPY', 'QQQ'],
            'name': ['SPDR S&P 500', 'Invesco QQQ'],
            'currency': ['USD', 'USD']
        })
        
        df = loader.load_etf_list(etf_df)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(df['currency'] == 'USD')
    
    def test_load_etf_list_missing_ticker(self):
        """Test that missing ticker column raises error."""
        loader = PortfolioDataLoader()
        etf_list = [{'name': 'Test ETF'}]
        
        with pytest.raises(ValueError, match="ETF list must contain 'ticker' column"):
            loader.load_etf_list(etf_list)
    
    def test_load_bond_list_etf_format(self):
        """Test loading bond ETF list."""
        loader = PortfolioDataLoader()
        bond_list = [
            {'ticker': 'TLT', 'name': '20+ Year Treasury', 'duration_target': 18},
            {'ticker': 'IEF', 'name': '7-10 Year Treasury', 'duration_target': 8}
        ]
        
        df = loader.load_bond_list(bond_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'ticker' in df.columns
        assert 'duration_target' in df.columns
        assert all(df['group'] == 'Bond ETF')
    
    def test_load_bond_list_individual_bonds(self):
        """Test loading individual bond list."""
        loader = PortfolioDataLoader()
        bond_list = [
            {'isin': 'US912828XG93', 'coupon': 2.75, 'maturity': '2025-01-15'},
            {'cusip': '912828XH77', 'coupon': 3.0, 'maturity': '2027-05-15'}
        ]
        
        df = loader.load_bond_list(bond_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'isin' in df.columns or 'cusip' in df.columns
        assert all(df['group'] == 'Bond')
    
    def test_load_bond_list_invalid_format(self):
        """Test that invalid bond list format raises error."""
        loader = PortfolioDataLoader()
        bond_list = [{'name': 'Invalid Bond'}]  # No ticker, isin, or cusip
        
        with pytest.raises(ValueError, match="Bond list must contain either 'ticker'"):
            loader.load_bond_list(bond_list)
    
    def test_calculate_returns_log(self):
        """Test log returns calculation."""
        loader = PortfolioDataLoader()
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        prices = pd.DataFrame({
            'ASSET1': [100, 101, 102, 101, 103],
            'ASSET2': [50, 51, 50.5, 52, 51]
        }, index=dates)
        
        returns = loader.calculate_returns(prices, method='log')
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == 4  # One less than prices due to first difference
        assert returns.columns.tolist() == ['ASSET1', 'ASSET2']
        
        # Check first return for ASSET1: log(101/100)
        expected_first_return = np.log(101/100)
        assert abs(returns.iloc[0, 0] - expected_first_return) < 1e-10
    
    def test_calculate_returns_simple(self):
        """Test simple returns calculation."""
        loader = PortfolioDataLoader()
        
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        prices = pd.DataFrame({
            'ASSET1': [100, 110, 105]
        }, index=dates)
        
        returns = loader.calculate_returns(prices, method='simple')
        
        assert len(returns) == 2
        assert abs(returns.iloc[0, 0] - 0.1) < 1e-10  # (110-100)/100 = 0.1
        assert abs(returns.iloc[1, 0] - (-0.045454545)) < 1e-8  # (105-110)/110
    
    def test_calculate_returns_invalid_method(self):
        """Test that invalid return method raises error."""
        loader = PortfolioDataLoader()
        prices = pd.DataFrame({'ASSET1': [100, 101]})
        
        with pytest.raises(ValueError, match="method must be 'log' or 'simple'"):
            loader.calculate_returns(prices, method='invalid')
    
    def test_prepare_optimization_data(self):
        """Test full optimization data preparation."""
        loader = PortfolioDataLoader()
        
        # Use sample lists
        etf_list = [
            {'ticker': 'SPY', 'name': 'SPDR S&P 500', 'weight_cap': 0.5},
            {'ticker': 'QQQ', 'name': 'Invesco QQQ', 'weight_cap': 0.3}
        ]
        bond_list = [
            {'ticker': 'TLT', 'name': '20+ Year Treasury', 'weight_cap': 0.4}
        ]
        
        data = loader.prepare_optimization_data(etf_list=etf_list, bond_list=bond_list)
        
        # Check all required keys are present
        required_keys = ['R', 'mean', 'covariance_matrix', 'instrument_names', 
                        'G', 'h', 'A', 'b', 'v', 'metadata']
        for key in required_keys:
            assert key in data
        
        # Check dimensions
        S, I = data['R'].shape
        assert I == 3  # 2 ETFs + 1 bond ETF
        assert len(data['mean']) == I
        assert data['covariance_matrix'].shape == (I, I)
        assert len(data['instrument_names']) == I
        
        # Check constraints
        assert data['A'].shape == (1, I)  # Budget constraint
        assert data['b'].shape == (1,)
        assert np.allclose(data['A'], 1.0)  # Sum of weights = 1
        assert np.allclose(data['b'], 1.0)
        
        # Check inequality constraints (long-only + weight caps)
        assert data['G'].shape[1] == I
        assert len(data['h']) == data['G'].shape[0]
        
        # Check relative market values
        assert len(data['v']) == I
        assert np.allclose(data['v'], 1.0)
        
        # Check metadata
        assert 'asset_data' in data['metadata']
        assert 'price_data' in data['metadata']
        assert 'return_data' in data['metadata']
    
    def test_prepare_optimization_data_no_assets(self):
        """Test that no assets raises error."""
        loader = PortfolioDataLoader()
        
        with pytest.raises(ValueError, match="Must provide at least one of etf_list or bond_list"):
            loader.prepare_optimization_data()
    
    def test_fetch_historical_data_synthetic(self):
        """Test synthetic data generation."""
        loader = PortfolioDataLoader()
        
        tickers = ['SPY', 'QQQ', 'TLT']
        data = loader.fetch_historical_data(tickers)
        
        assert isinstance(data, pd.DataFrame)
        assert data.columns.tolist() == tickers
        assert len(data) > 0
        assert all(data > 0)  # Prices should be positive


class TestSampleDataCreation:
    """Test sample data creation functions."""
    
    def test_create_sample_etf_list(self):
        """Test sample ETF list creation."""
        etf_list = create_sample_etf_list()
        
        assert isinstance(etf_list, list)
        assert len(etf_list) > 0
        
        # Check required fields
        for etf in etf_list:
            assert 'ticker' in etf
            assert 'name' in etf
            assert 'currency' in etf
            assert 'group' in etf
            assert 'weight_cap' in etf
    
    def test_create_sample_bond_list(self):
        """Test sample bond list creation."""
        bond_list = create_sample_bond_list()
        
        assert isinstance(bond_list, list)
        assert len(bond_list) > 0
        
        # Check required fields
        for bond in bond_list:
            assert 'ticker' in bond
            assert 'name' in bond
            assert 'currency' in bond
            assert 'group' in bond
            assert 'duration_target' in bond
            assert 'weight_cap' in bond
    
    def test_sample_lists_integration(self):
        """Test that sample lists work with the data loader."""
        loader = PortfolioDataLoader()
        
        etf_list = create_sample_etf_list()
        bond_list = create_sample_bond_list()
        
        # Should not raise any errors
        etf_df = loader.load_etf_list(etf_list)
        bond_df = loader.load_bond_list(bond_list)
        
        assert len(etf_df) == len(etf_list)
        assert len(bond_df) == len(bond_list)
        
        # Test full optimization data preparation
        data = loader.prepare_optimization_data(etf_list=etf_list, bond_list=bond_list)
        
        expected_instruments = len(etf_list) + len(bond_list)
        assert len(data['instrument_names']) == expected_instruments