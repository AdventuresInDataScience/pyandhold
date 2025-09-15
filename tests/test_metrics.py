"""Tests for metrics module."""

import pytest
import pandas as pd
import numpy as np
from portfolio_optimizer.metrics import ReturnMetrics, RiskMetrics, PerformanceMetrics


class TestReturnMetrics:
    """Test ReturnMetrics class."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2023-01-01', periods=252)
        prices = pd.DataFrame(
            100 * (1 + np.random.randn(252, 2) * 0.01).cumprod(),
            index=dates,
            columns=['Asset1', 'Asset2']
        )
        return prices
    
    def test_calculate_returns(self, sample_prices):
        """Test return calculation."""
        returns = ReturnMetrics.calculate_returns(sample_prices)
        
        assert len(returns) == len(sample_prices) - 1
        assert returns.isna().sum().sum() == 0
    
    def test_cumulative_returns(self, sample_prices):
        """Test cumulative returns."""
        returns = ReturnMetrics.calculate_returns(sample_prices)
        cum_returns = ReturnMetrics.cumulative_returns(returns)
        
        assert len(cum_returns) == len(returns)
        assert cum_returns.iloc[-1].values[0] == pytest.approx(
            (sample_prices.iloc[-1] / sample_prices.iloc[0] - 1).values[0],
            rel=0.01
        )
    
    def test_annualized_return(self, sample_prices):
        """Test annualized return calculation."""
        returns = ReturnMetrics.calculate_returns(sample_prices)
        ann_return = ReturnMetrics.annualized_return(returns)
        
        assert isinstance(ann_return, pd.Series)
        assert len(ann_return) == 2
        assert all(ann_return > -1)  # Returns should be > -100%
    
    def test_cagr(self, sample_prices):
        """Test CAGR calculation."""
        cagr = ReturnMetrics.cagr(sample_prices)
        
        assert isinstance(cagr, pd.Series)
        assert len(cagr) == 2
        assert all(cagr > -1)


class TestRiskMetrics:
    """Test RiskMetrics class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        dates = pd.date_range('2023-01-01', periods=252)
        returns = pd.DataFrame(
            np.random.randn(252, 2) * 0.01,
            index=dates,
            columns=['Asset1', 'Asset2']
        )
        return returns
    
    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        vol = RiskMetrics.volatility(sample_returns)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == 2
        assert all(vol > 0)
        assert all(vol < 1)  # Annual vol should be < 100%
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        dates = pd.date_range('2023-01-01', periods=100)
        prices = pd.DataFrame(
            [100, 110, 105, 95, 100, 90, 95, 100, 105, 110],
            index=dates[:10],
            columns=['Asset']
        )
        
        max_dd = RiskMetrics.max_drawdown(prices)
        
        assert max_dd['Asset'] == pytest.approx(-0.1818, rel=0.01)  # (90-110)/110
    
    def test_value_at_risk(self, sample_returns):
        """Test VaR calculation."""
        var = RiskMetrics.value_at_risk(sample_returns, 0.95)
        
        assert isinstance(var, pd.Series)
        assert len(var) == 2
        assert all(var < 0)  # VaR should be negative for losses
    
    def test_correlation_matrix(self, sample_returns):
        """Test correlation matrix."""
        corr = RiskMetrics.correlation_matrix(sample_returns)
        
        assert corr.shape == (2, 2)
        assert all(np.diag(corr) == 1)  # Diagonal should be 1
        assert corr.iloc[0, 1] == corr.iloc[1, 0]  # Symmetric


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range('2023-01-01', periods=252)
        returns = pd.DataFrame(
            np.random.randn(252, 2) * 0.01 + 0.0003,
            index=dates,
            columns=['Asset1', 'Asset2']
        )
        benchmark = pd.Series(
            np.random.randn(252) * 0.008 + 0.0002,
            index=dates
        )
        prices = pd.DataFrame(
            100 * (1 + returns).cumprod(),
            index=dates,
            columns=['Asset1', 'Asset2']
        )
        return returns, benchmark, prices
    
    def test_sharpe_ratio(self, sample_data):
        """Test Sharpe ratio calculation."""
        returns, _, _ = sample_data
        sharpe = PerformanceMetrics.sharpe_ratio(returns)
        
        assert isinstance(sharpe, pd.Series)
        assert len(sharpe) == 2
        assert all(sharpe > -5) and all(sharpe < 5)  # Reasonable range
    
    def test_sortino_ratio(self, sample_data):
        """Test Sortino ratio calculation."""
        returns, _, _ = sample_data
        sortino = PerformanceMetrics.sortino_ratio(returns)
        
        assert isinstance(sortino, pd.Series)
        assert len(sortino) == 2
    
    def test_calmar_ratio(self, sample_data):
        """Test Calmar ratio calculation."""
        _, _, prices = sample_data
        calmar = PerformanceMetrics.calmar_ratio(prices)
        
        assert isinstance(calmar, pd.Series)
        assert len(calmar) == 2
    
    def test_information_ratio(self, sample_data):
        """Test Information ratio calculation."""
        returns, benchmark, _ = sample_data
        ir = PerformanceMetrics.information_ratio(returns, benchmark)
        
        assert isinstance(ir, pd.Series)
        assert len(ir) == 2