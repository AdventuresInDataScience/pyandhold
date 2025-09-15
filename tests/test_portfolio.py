"""Tests for portfolio module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_optimizer.portfolio.portfolio import Portfolio
from portfolio_optimizer.data.downloader import DataDownloader


class TestPortfolio:
    """Test Portfolio class."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        weights = {
            'AAPL': 0.3,
            'MSFT': 0.3,
            'GOOGL': 0.2,
            'AMZN': 0.2
        }
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        return Portfolio(
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000
        )
    
    def test_portfolio_initialization(self, sample_portfolio):
        """Test portfolio initialization."""
        assert len(sample_portfolio.tickers) == 4
        assert sample_portfolio.initial_capital == 100000
        assert sum(sample_portfolio.weights.values()) == 1.0
    
    def test_fetch_data(self, sample_portfolio):
        """Test data fetching."""
        sample_portfolio.fetch_data()
        
        assert sample_portfolio.prices is not None
        assert sample_portfolio.returns is not None
        assert len(sample_portfolio.prices.columns) == 4
    
    def test_calculate_metrics(self, sample_portfolio):
        """Test metrics calculation."""
        metrics = sample_portfolio.calculate_metrics()
        
        assert 'sharpe_ratio' in metrics
        assert 'volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'cagr' in metrics
    
    def test_portfolio_returns(self, sample_portfolio):
        """Test portfolio returns calculation."""
        returns = sample_portfolio.calculate_portfolio_returns()
        
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
    
    def test_portfolio_value(self, sample_portfolio):
        """Test portfolio value calculation."""
        value = sample_portfolio.calculate_portfolio_value()
        
        assert isinstance(value, pd.Series)
        assert value.iloc[0] == pytest.approx(100000, rel=0.01)