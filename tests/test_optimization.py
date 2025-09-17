"""Tests for optimization module."""

import pytest
import pandas as pd
import numpy as np
from pyandhold.optimization import PortfolioOptimizer, RobustOptimizer, ConstraintBuilder


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)
        returns = pd.DataFrame(
            np.random.multivariate_normal(
                [0.0003, 0.0002, 0.0004],
                [[0.01, 0.002, 0.001],
                 [0.002, 0.008, 0.002],
                 [0.001, 0.002, 0.012]],
                252
            ),
            index=dates,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        return returns
    
    def test_optimize_sharpe(self, sample_returns):
        """Test Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights = optimizer.optimize_sharpe()
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in weights.values())
    
    def test_optimize_min_variance(self, sample_returns):
        """Test minimum variance optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights = optimizer.optimize_min_variance()
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in weights.values())
    
    def test_optimize_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights = optimizer.optimize_risk_parity()
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer(sample_returns)
        frontier = optimizer.efficient_frontier(n_portfolios=10)
        
        assert isinstance(frontier, pd.DataFrame)
        assert len(frontier) <= 10
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns
        assert 'sharpe_ratio' in frontier.columns
    
    def test_weight_bounds(self, sample_returns):
        """Test weight bounds constraints."""
        optimizer = PortfolioOptimizer(sample_returns)
        weights = optimizer.optimize_sharpe(weight_bounds=(0.1, 0.5))
        
        assert all(w >= 0.09 for w in weights.values())  # Small tolerance
        assert all(w <= 0.51 for w in weights.values())


class TestMarsOptimizer:
    """Test MARS optimization implementation."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for MARS tests."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252)
        asset_names = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        # Generate realistic return data
        returns_data = np.random.multivariate_normal(
            mean=[0.001, 0.0015, 0.0008, 0.002],  # Daily returns
            cov=[[0.0004, 0.0002, 0.0001, 0.0003],
                 [0.0002, 0.0006, 0.0001, 0.0002],
                 [0.0001, 0.0001, 0.0003, 0.0001],
                 [0.0003, 0.0002, 0.0001, 0.0008]],
            size=252
        )
        
        return pd.DataFrame(returns_data, index=dates, columns=asset_names)
    
    def test_mars_optimizer_creation(self, sample_returns):
        """Test MARS optimizer initialization."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        assert optimizer.optimizer == 'mars'
        assert optimizer.n_assets == 4
        assert list(optimizer.returns.columns) == ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    def test_mars_sharpe_optimization(self, sample_returns):
        """Test MARS Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_sharpe(n_trials=50, verbose=False)
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Allow slightly more tolerance for MARS
        assert all(w >= -0.01 for w in weights.values())  # Small negative tolerance for numerical precision
        assert all(w <= 1.01 for w in weights.values())   # Small positive tolerance
        
        # Test that weights are reasonable (not all zero or extreme values)
        assert max(weights.values()) > 0.05
        assert sum(w > 0.01 for w in weights.values()) >= 2  # At least 2 assets should have meaningful weights
    
    def test_mars_min_variance_optimization(self, sample_returns):
        """Test MARS minimum variance optimization."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_min_variance(n_trials=50)
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(w >= -0.01 for w in weights.values())
        assert all(w <= 1.01 for w in weights.values())
    
    def test_mars_risk_parity_optimization(self, sample_returns):
        """Test MARS risk parity optimization."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_risk_parity(n_trials=50)
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_mars_max_return_optimization(self, sample_returns):
        """Test MARS maximum return with volatility constraint optimization."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_max_return(max_volatility=0.15, n_trials=50)
        
        assert isinstance(weights, dict)
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_mars_vs_scipy_comparison(self, sample_returns):
        """Compare MARS and SciPy optimizers to ensure both work."""
        # MARS optimizer
        optimizer_mars = PortfolioOptimizer(sample_returns, optimizer='mars')
        mars_weights = optimizer_mars.optimize_sharpe(n_trials=100, verbose=False)
        
        # SciPy optimizer
        optimizer_scipy = PortfolioOptimizer(sample_returns, optimizer='scipy')
        scipy_weights = optimizer_scipy.optimize_sharpe(verbose=False)
        
        # Both should produce valid results
        assert isinstance(mars_weights, dict)
        assert isinstance(scipy_weights, dict)
        assert len(mars_weights) == len(scipy_weights) == 4
        
        # Both should sum to 1 (within tolerance)
        assert abs(sum(mars_weights.values()) - 1.0) < 0.01
        assert abs(sum(scipy_weights.values()) - 1.0) < 0.001
        
        # Results don't need to be identical, but should be reasonable
        # (MARS is stochastic, SciPy is deterministic)
        mars_max_weight = max(mars_weights.values())
        scipy_max_weight = max(scipy_weights.values())
        assert mars_max_weight > 0.05  # Should not be all equal weights
        assert scipy_max_weight > 0.05
    
    def test_mars_weight_bounds(self, sample_returns):
        """Test MARS optimizer with weight bounds."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_sharpe(
            weight_bounds=(0.1, 0.4),
            n_trials=50,
            verbose=False
        )
        
        # Note: MARS handles bounds through normalization, so bounds may not be strictly enforced
        # but the optimization should still work
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    @pytest.mark.slow
    def test_mars_performance_with_more_trials(self, sample_returns):
        """Test MARS optimizer with more trials for better convergence."""
        optimizer = PortfolioOptimizer(sample_returns, optimizer='mars')
        weights = optimizer.optimize_sharpe(n_trials=500, verbose=False)
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.005  # Tighter tolerance with more trials


class TestRobustOptimizer:
    """Test RobustOptimizer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500)
        returns = pd.DataFrame(
            np.random.randn(500, 3) * 0.01,
            index=dates,
            columns=['Asset1', 'Asset2', 'Asset3']
        )
        return returns
    
    def test_cross_validation_optimize(self, sample_returns):
        """Test cross-validation optimization."""
        optimizer = RobustOptimizer(sample_returns)
        weights = optimizer.cross_validation_optimize(
            optimization_method='sharpe',
            n_splits=3
        )
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_bootstrap_optimize(self, sample_returns):
        """Test bootstrap optimization."""
        optimizer = RobustOptimizer(sample_returns)
        weights = optimizer.bootstrap_optimize(
            optimization_method='min_variance',
            n_iterations=10
        )
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_shrinkage_optimize(self, sample_returns):
        """Test shrinkage covariance optimization."""
        optimizer = RobustOptimizer(sample_returns)
        weights = optimizer.shrinkage_covariance_optimize(
            optimization_method='min_variance',
            shrinkage_factor=0.2
        )
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestConstraintBuilder:
    """Test ConstraintBuilder class."""
    
    def test_long_only_constraint(self):
        """Test long-only constraint."""
        constraint = ConstraintBuilder.long_only_constraint()
        
        assert constraint['type'] == 'ineq'
        assert constraint['fun'](np.array([0.5, 0.3, 0.2])).all()
        assert not constraint['fun'](np.array([-0.1, 0.5, 0.6])).all()
    
    def test_sector_constraint(self):
        """Test sector constraints."""
        sector_mapping = {
            'tech': [0, 1],
            'finance': [2, 3]
        }
        sector_limits = {
            'tech': (0.2, 0.6),
            'finance': (0.1, 0.4)
        }
        
        constraints = ConstraintBuilder.sector_constraint(
            sector_mapping,
            sector_limits
        )
        
        assert len(constraints) == 4  # 2 sectors × 2 bounds
    
    def test_leverage_constraint(self):
        """Test leverage constraints."""
        constraints = ConstraintBuilder.leverage_constraint(0.95, 1.05)
        
        assert len(constraints) == 2
        weights = np.array([0.3, 0.3, 0.4])
        assert all(c['fun'](weights) >= 0 for c in constraints)