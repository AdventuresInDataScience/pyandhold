"""Tests for optimization module."""

import pytest
import pandas as pd
import numpy as np
from portfolio_optimizer.optimization import PortfolioOptimizer, RobustOptimizer, ConstraintBuilder


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