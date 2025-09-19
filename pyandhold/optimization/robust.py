"""Robust optimization methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold
from .optimizers import PortfolioOptimizer


class RobustOptimizer:
    """Robust portfolio optimization methods."""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize robust optimizer.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def cross_validation_optimize(
        self,
        optimization_method: str = 'sharpe',
        n_splits: int = 5,
        **optimization_kwargs
    ) -> Dict[str, float]:
        """
        Cross-validation based optimization.
        
        Args:
            optimization_method: Method to use ('sharpe', 'min_variance', etc.)
            n_splits: Number of CV splits
            **optimization_kwargs: Arguments for optimization method
            
        Returns:
            Dictionary of robust weights
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        weights_list = []
        
        for train_idx, _ in kf.split(self.returns):
            train_returns = self.returns.iloc[train_idx]
            
            # Create optimizer for this fold
            optimizer = PortfolioOptimizer(
                train_returns,
                self.risk_free_rate,
                self.periods_per_year
            )
            
            # Optimize based on method
            if optimization_method == 'sharpe':
                weights = optimizer.optimize_sharpe(**optimization_kwargs)
            elif optimization_method == 'min_variance':
                weights = optimizer.optimize_min_variance(**optimization_kwargs)
            elif optimization_method == 'risk_parity':
                weights = optimizer.optimize_risk_parity(**optimization_kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            weights_list.append(weights)
        
        # Average weights across folds
        avg_weights = {}
        for ticker in self.returns.columns:
            avg_weights[ticker] = np.mean([w[ticker] for w in weights_list])
        
        # Normalize to sum to 1
        total = sum(avg_weights.values())
        return {k: v/total for k, v in avg_weights.items()}
    
    def bootstrap_optimize(
        self,
        optimization_method: str = 'sharpe',
        n_iterations: int = 100,
        sample_size: Optional[int] = None,
        **optimization_kwargs
    ) -> Dict[str, float]:
        """
        Bootstrap resampling optimization.
        
        Args:
            optimization_method: Method to use
            n_iterations: Number of bootstrap iterations
            sample_size: Size of each bootstrap sample
            **optimization_kwargs: Arguments for optimization method
            
        Returns:
            Dictionary of robust weights
        """
        if sample_size is None:
            sample_size = len(self.returns)
        
        weights_list = []
        
        for _ in range(n_iterations):
            # Bootstrap sample
            sample_idx = np.random.choice(
                len(self.returns),
                size=sample_size,
                replace=True
            )
            sample_returns = self.returns.iloc[sample_idx]
            
            # Optimize on sample
            optimizer = PortfolioOptimizer(
                sample_returns,
                self.risk_free_rate,
                self.periods_per_year
            )
            
            if optimization_method == 'sharpe':
                weights = optimizer.optimize_sharpe(**optimization_kwargs)
            elif optimization_method == 'min_variance':
                weights = optimizer.optimize_min_variance(**optimization_kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            weights_list.append(weights)
        
        # Average weights
        avg_weights = {}
        for ticker in self.returns.columns:
            avg_weights[ticker] = np.mean([w[ticker] for w in weights_list])
        
        # Normalize
        total = sum(avg_weights.values())
        return {k: v/total for k, v in avg_weights.items()}
    
    def shrinkage_covariance_optimize(
        self,
        optimization_method: str = 'min_variance',
        shrinkage_factor: float = 0.1,
        **optimization_kwargs
    ) -> Dict[str, float]:
        """
        Optimization with shrinkage estimator for covariance.
        
        Args:
            optimization_method: Method to use
            shrinkage_factor: Shrinkage factor (0 to 1)
            **optimization_kwargs: Arguments for optimization method
            
        Returns:
            Dictionary of weights
        """
        # Calculate sample covariance
        sample_cov = self.returns.cov()
        
        # Shrinkage target - diagonal matrix with average variance
        avg_variance = np.mean(np.diag(sample_cov))
        shrinkage_target = np.eye(len(sample_cov)) * avg_variance
        
        # Shrunk covariance
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * shrinkage_target
        
        # Create modified returns with shrunk covariance
        # This is a workaround - ideally we'd modify the optimizer directly
        optimizer = PortfolioOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year
        )
        
        # Override covariance matrix
        optimizer.cov_matrix = shrunk_cov
        
        if optimization_method == 'sharpe':
            return optimizer.optimize_sharpe(**optimization_kwargs)
        elif optimization_method == 'min_variance':
            return optimizer.optimize_min_variance(**optimization_kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def black_litterman_optimization(
        self,
        market_caps: pd.Series,
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        tau: float = 0.1,
        risk_aversion: float = 3.0,
        **optimization_kwargs
    ) -> Dict[str, float]:
        """
        Black-Litterman optimization with views.
        
        Args:
            market_caps: Market capitalizations for assets
            views: Dictionary of views {asset: expected_return}
            view_confidences: Dictionary of view confidences {asset: confidence}
            tau: Uncertainty parameter (typically 0.01 to 1.0)
            risk_aversion: Risk aversion parameter
            **optimization_kwargs: Additional optimization arguments
            
        Returns:
            Dictionary of optimal weights
        """
        # Market weights (proportional to market cap)
        w_market = market_caps / market_caps.sum()
        
        # Sample covariance matrix
        cov_matrix = self.returns.cov().values
        
        # Implied returns (reverse optimization)
        pi = risk_aversion * np.dot(cov_matrix, w_market.values)
        
        # Create view matrix P and view returns Q
        n_assets = len(self.returns.columns)
        view_assets = list(views.keys())
        n_views = len(view_assets)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)
        
        for i, asset in enumerate(view_assets):
            asset_idx = list(self.returns.columns).index(asset)
            P[i, asset_idx] = 1.0
            Q[i] = views[asset]
            # View uncertainty (inverse of confidence)
            omega_diag[i] = 1.0 / view_confidences[asset]
        
        omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau_cov = tau * cov_matrix
        
        # New expected returns
        M1 = np.linalg.inv(tau_cov)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau_cov), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
        
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # New covariance matrix
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Create modified returns with BL parameters
        # Generate synthetic returns that match BL statistics
        np.random.seed(42)
        n_samples = len(self.returns)
        synthetic_returns = np.random.multivariate_normal(
            mu_bl / self.periods_per_year,  # Daily returns
            cov_bl / self.periods_per_year,  # Daily covariance
            n_samples
        )
        
        bl_returns = pd.DataFrame(
            synthetic_returns,
            columns=self.returns.columns,
            index=self.returns.index
        )
        
        # Optimize using Black-Litterman inputs
        optimizer = PortfolioOptimizer(
            bl_returns,
            self.risk_free_rate,
            self.periods_per_year
        )
        
        # Use Sharpe optimization as default
        return optimizer.optimize_sharpe(**optimization_kwargs)