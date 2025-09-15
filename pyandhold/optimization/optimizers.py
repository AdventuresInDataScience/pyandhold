"""Portfolio optimization module."""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple, List, Union
from scipy.optimize import minimize
from ..metrics.risk import RiskMetrics
from ..metrics.performance import PerformanceMetrics


class PortfolioOptimizer:
    """Portfolio optimization with various objectives and constraints."""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize optimizer.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.n_assets = len(returns.columns)
        
        # Pre-calculate frequently used matrices
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
    def optimize_sharpe(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Maximize Sharpe ratio.
        
        Args:
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            
        Returns:
            Dictionary of optimal weights
        """
        def negative_sharpe(weights):
            returns = np.dot(weights, self.mean_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (returns - self.risk_free_rate/self.periods_per_year) / volatility
            return -sharpe * np.sqrt(self.periods_per_year)
        
        return self._optimize(negative_sharpe, weight_bounds, constraints)
    
    def optimize_min_variance(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Minimize portfolio variance.
        
        Args:
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            
        Returns:
            Dictionary of optimal weights
        """
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        return self._optimize(portfolio_variance, weight_bounds, constraints)
    
    def optimize_max_return(
        self,
        max_volatility: float,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Maximize return subject to volatility constraint.
        
        Args:
            max_volatility: Maximum allowed volatility (annualized)
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            
        Returns:
            Dictionary of optimal weights
        """
        def negative_return(weights):
            return -np.dot(weights, self.mean_returns)
        
        # Add volatility constraint
        if constraints is None:
            constraints = {}
        
        def volatility_constraint(weights):
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return max_volatility / np.sqrt(self.periods_per_year) - vol
        
        constraints['volatility'] = {
            'type': 'ineq',
            'fun': volatility_constraint
        }
        
        return self._optimize(negative_return, weight_bounds, constraints)
    
    def optimize_risk_parity(
        self,
        weight_bounds: Tuple[float, float] = (0, 1)
    ) -> Dict[str, float]:
        """
        Risk parity optimization - equal risk contribution.
        
        Args:
            weight_bounds: Min and max weight for each asset
            
        Returns:
            Dictionary of optimal weights
        """
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize variance of risk contributions
            return np.var(contrib)
        
        return self._optimize(risk_parity_objective, weight_bounds)
    
    def optimize_cvar(
        self,
        confidence_level: float = 0.95,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Minimize Conditional Value at Risk using CVXPy.
        
        Args:
            confidence_level: Confidence level for CVaR
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            
        Returns:
            Dictionary of optimal weights
        """
        n_samples, n_assets = self.returns.shape
        returns_matrix = self.returns.values
        
        # Decision variables
        weights = cp.Variable(n_assets)
        z = cp.Variable(n_samples)
        zeta = cp.Variable()
        
        # Portfolio returns
        portfolio_returns = returns_matrix @ weights
        
        # CVaR formulation
        cvar = zeta + (1 / (n_samples * (1 - confidence_level))) * cp.sum(z)
        
        # Constraints
        constraints_list = [
            weights >= weight_bounds[0],
            weights <= weight_bounds[1],
            cp.sum(weights) == 1,
            z >= 0,
            z >= -portfolio_returns - zeta
        ]
        
        # Add custom constraints if provided
        if constraints and 'min_assets' in constraints:
            # This would require binary variables - simplified here
            pass
        
        # Objective
        objective = cp.Minimize(cvar)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS)
        
        if problem.status == 'optimal':
            return dict(zip(self.returns.columns, weights.value))
        else:
            raise ValueError(f"Optimization failed: {problem.status}")
    
    def efficient_frontier(
        self,
        n_portfolios: int = 50,
        weight_bounds: Tuple[float, float] = (0, 1)
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_portfolios: Number of portfolios on frontier
            weight_bounds: Min and max weight for each asset
            
        Returns:
            DataFrame with frontier portfolios
        """
        # Get min and max return portfolios
        min_vol_weights = self.optimize_min_variance(weight_bounds)
        min_vol_return = np.dot(
            list(min_vol_weights.values()),
            self.mean_returns
        )
        
        max_return = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_vol_return, max_return, n_portfolios)
        
        frontier_data = []
        
        for target_return in target_returns:
            # Optimize for minimum variance given target return
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            constraints = {
                'return': {
                    'type': 'eq',
                    'fun': lambda w: np.dot(w, self.mean_returns) - target_return
                }
            }
            
            try:
                weights = self._optimize(portfolio_variance, weight_bounds, constraints)
                
                # Calculate metrics
                w_array = np.array(list(weights.values()))
                ret = np.dot(w_array, self.mean_returns) * self.periods_per_year
                vol = np.sqrt(np.dot(w_array.T, np.dot(self.cov_matrix, w_array))) * np.sqrt(self.periods_per_year)
                sharpe = (ret - self.risk_free_rate) / vol
                
                frontier_data.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    **{f'weight_{ticker}': weight for ticker, weight in weights.items()}
                })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def _optimize(
        self,
        objective_func,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Generic optimization function.
        
        Args:
            objective_func: Objective function to minimize
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            
        Returns:
            Dictionary of optimal weights
        """
        # Initial guess - equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds for each weight
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(self.n_assets))
        
        # Basic constraints - weights sum to 1
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Add custom constraints
        if constraints:
            for name, constraint in constraints.items():
                if name not in ['volatility', 'return']:  # These are handled separately
                    constraints_list.append(constraint)
                elif name == 'volatility' or name == 'return':
                    constraints_list.append(constraint)
        
        # Optimize
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Return as dictionary
        return dict(zip(self.returns.columns, result.x))