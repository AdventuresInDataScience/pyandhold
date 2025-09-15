"""Tests for data module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_optimizer.data import DataDownloader, DataPreprocessor, StockUniverse


class TestDataDownloader:
    """Test DataDownloader class."""
    
    def test_download_single_ticker(self):
        """Test downloading single ticker."""
        downloader = DataDownloader()
        data = downloader.download_data(
            ['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data.columns) == 1
        assert 'AAPL' in data.columns
        assert len(data) > 200  # Should have ~252 trading days
    
    def test_download_multiple_tickers(self):
        """Test downloading multiple tickers."""
        downloader = DataDownloader()
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        data = downloader.download_data(
            tickers,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data.columns) == 3
        assert all(ticker in data.columns for ticker in tickers)
    
    def test_download_returns(self):
        """Test downloading returns."""
        downloader = DataDownloader()
        returns = downloader.download_returns(
            ['SPY'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert isinstance(returns, pd.DataFrame)
        assert returns.isna().sum().sum() == 0  # No NaN after pct_change().dropna()
        assert all(returns.abs().max() < 1)  # Daily returns should be < 100%


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            index=dates,
            columns=['A', 'B', 'C']
        )
        # Add some outliers
        data.iloc[10, 0] = 0.5
        data.iloc[20, 1] = -0.5
        return data
    
    def test_winsorize(self, sample_data):
        """Test winsorization."""
        winsorized = DataPreprocessor.winsorize(
            sample_data,
            limits=(0.05, 0.05)
        )
        
        assert winsorized.shape == sample_data.shape
        assert winsorized.max().max() <= sample_data.max().max()
        assert winsorized.min().min() >= sample_data.min().min()
    
    def test_align_data(self, sample_data):
        """Test data alignment."""
        # Add NaN values
        sample_data.iloc[0:5, 0] = np.nan
        sample_data.iloc[-5:, 1] = np.nan
        
        aligned = DataPreprocessor.align_data(sample_data)
        
        assert aligned.isna().sum().sum() == 0
        assert len(aligned) < len(sample_data)
    
    def test_remove_low_variance(self, sample_data):
        """Test removing low variance assets."""
        # Set one column to near-constant
        sample_data['D'] = 0.0001
        
        filtered = DataPreprocessor.remove_low_variance_assets(
            sample_data,
            threshold=0.0001
        )
        
        assert 'D' not in filtered.columns
        assert len(filtered.columns) == 3
    
    def test_normalize_zscore(self, sample_data):
        """Test z-score normalization."""
        normalized = DataPreprocessor.normalize_data(
            sample_data,
            method='zscore'
        )
        
        assert np.allclose(normalized.mean(), 0, atol=1e-10)
        assert np.allclose(normalized.std(), 1, atol=1e-10)


class TestStockUniverse:
    """Test StockUniverse class."""
    
    def test_get_sp500_tickers(self):
        """Test getting S&P 500 tickers."""
        tickers = StockUniverse.get_sp500_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert 'AAPL' in tickers
    
    def test_get_sector_etfs(self):
        """Test getting sector ETFs."""
        etfs = StockUniverse.get_sector_etfs()
        
        assert isinstance(etfs, dict)
        assert 'technology' in etfs
        assert etfs['technology'] == 'XLK'
    
    def test_get_asset_class_etfs(self):
        """Test getting asset class ETFs."""
        etfs = StockUniverse.get_asset_class_etfs()
        
        assert isinstance(etfs, dict)
        assert 'bonds' in etfs
        assert 'gold' in etfs