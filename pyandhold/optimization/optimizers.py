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
        constraints: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Maximize Sharpe ratio.
        
        Args:
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            verbose: Print optimization details
            
        Returns:
            Dictionary of optimal weights
        """
        def negative_sharpe(weights):
            weights = np.array(weights)
            
            # Add small penalty for numerical stability near boundaries
            if np.any(weights < 0) or np.sum(weights) == 0:
                return 1e6
                
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            if portfolio_variance <= 1e-12:  # Avoid division by zero
                return 1e6
                
            volatility = np.sqrt(portfolio_variance)
            sharpe = (portfolio_return - self.risk_free_rate/self.periods_per_year) / volatility
            return -sharpe * np.sqrt(self.periods_per_year)
        
        return self._optimize(negative_sharpe, weight_bounds, constraints, verbose)
    
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
            weights = np.array(weights)
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
            weights = np.array(weights)
            
            # Portfolio variance and volatility
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            if portfolio_var <= 1e-12:  # Avoid division by zero
                return 1e6
                
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Marginal risk contribution
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_contrib = portfolio_vol / self.n_assets
            
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
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
        constraints: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Generic optimization function with multiple solvers and initializations.
        
        Args:
            objective_func: Objective function to minimize
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            verbose: Print optimization details
            
        Returns:
            Dictionary of optimal weights
        """
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
        
        # Multiple initial guesses for better convergence
        initial_guesses = self._generate_initial_guesses(weight_bounds)
        
        # Try different solvers in order of preference
        solvers = ['SLSQP', 'trust-constr']
        
        best_result = None
        best_objective = float('inf')
        
        for solver in solvers:
            if verbose:
                print(f"\nTrying solver: {solver}")
                
            for i, x0 in enumerate(initial_guesses):
                try:
                    if verbose:
                        print(f"  Initial guess {i+1}: objective = {objective_func(x0):.6f}")
                    
                    # Solver-specific options
                    options = {'maxiter': 5000, 'ftol': 1e-12}
                    if solver == 'trust-constr':
                        options.update({'gtol': 1e-10, 'xtol': 1e-12})
                    elif solver == 'SLSQP':
                        options.update({'ftol': 1e-12, 'eps': 1e-8})
                    
                    result = minimize(
                        objective_func,
                        x0,
                        method=solver,
                        bounds=bounds,
                        constraints=constraints_list,
                        options=options
                    )
                    
                    if verbose:
                        print(f"    Result: success={result.success}, fun={result.fun:.6f}, message='{result.message}'")
                    
                    # Check if this is the best result so far
                    if result.success and result.fun < best_objective:
                        best_result = result
                        best_objective = result.fun
                        
                        if verbose:
                            print(f"    New best result! Objective: {best_objective:.6f}")
                            print(f"    Weights: {dict(zip(self.returns.columns, result.x))}")
                        
                        # If we get a very good result, we can stop early
                        if abs(best_objective) > 10:  # Only stop early for very good Sharpe ratios
                            break
                
                except Exception as e:
                    if verbose:
                        print(f"    Failed with error: {e}")
                    continue
            
            # If we found a good solution, we can stop trying other solvers
            if best_result is not None and best_result.success:
                break
        
        if best_result is None or not best_result.success:
            # Try one more time with a different approach - sequential optimization
            if verbose:
                print("\nAll standard methods failed, trying sequential quadratic programming with different settings...")
            
            # Try with looser tolerances
            x0 = self._generate_initial_guesses(weight_bounds)[0]  # Equal weights
            result = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                best_result = result
            else:
                raise ValueError(f"Optimization failed with all methods. Last message: {result.message if result else 'No result'}")
        
        # Normalize weights to ensure they sum to 1 (numerical precision)
        weights = best_result.x
        weights = weights / np.sum(weights)
        
        # Final check - ensure weights are reasonable
        if np.any(weights < -1e-6) or np.any(weights > 1 + 1e-6):
            raise ValueError("Optimization produced invalid weights outside bounds")
        
        # Clip small negative values to zero
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)  # Renormalize after clipping
        
        if verbose:
            print(f"\nFinal optimization result:")
            print(f"Objective: {best_result.fun:.6f}")
            print(f"Final weights: {dict(zip(self.returns.columns, weights))}")
        
        # Return as dictionary
        return dict(zip(self.returns.columns, weights))
    
    def _generate_initial_guesses(self, weight_bounds: Tuple[float, float]) -> List[np.ndarray]:
        """Generate multiple initial weight guesses for optimization."""
        initial_guesses = []
        np.random.seed(42)  # For reproducibility
        
        # 1. Equal weights
        equal_weight = np.ones(self.n_assets) / self.n_assets
        if weight_bounds[0] <= 1/self.n_assets <= weight_bounds[1]:
            initial_guesses.append(equal_weight)
        
        # 2. Random weights (normalized) - multiple attempts
        for seed in range(5):
            np.random.seed(42 + seed)
            random_weights = np.random.uniform(weight_bounds[0], weight_bounds[1], self.n_assets)
            # Ensure they can sum to 1 given bounds
            if np.sum(random_weights) > 0:
                random_weights = random_weights / np.sum(random_weights)
                # Check if normalized weights respect bounds
                if np.all(random_weights >= weight_bounds[0]) and np.all(random_weights <= weight_bounds[1]):
                    initial_guesses.append(random_weights)
        
        # 3. Return-weighted (if returns are available and meaningful)
        try:
            if np.any(self.mean_returns > 0) and not np.all(np.isclose(self.mean_returns, self.mean_returns[0])):
                # Use only positive returns for weighting
                pos_returns = np.maximum(self.mean_returns, 1e-6)  # Avoid zero/negative
                return_weights = pos_returns / np.sum(pos_returns)
                if np.all(return_weights >= weight_bounds[0]) and np.all(return_weights <= weight_bounds[1]):
                    initial_guesses.append(return_weights.values)
        except:
            pass
        
        # 4. Inverse volatility weighted
        try:
            asset_vols = np.sqrt(np.diag(self.cov_matrix))
            if np.all(asset_vols > 0):
                inv_vol_weights = (1 / asset_vols) / np.sum(1 / asset_vols)
                if np.all(inv_vol_weights >= weight_bounds[0]) and np.all(inv_vol_weights <= weight_bounds[1]):
                    initial_guesses.append(inv_vol_weights)
        except:
            pass
        
        # 5. Single asset allocations (corner solutions)
        if weight_bounds[1] >= 1.0:  # Only if we can put 100% in one asset
            for i in range(min(self.n_assets, 3)):  # Try first 3 assets
                single_asset = np.zeros(self.n_assets)
                single_asset[i] = 1.0
                initial_guesses.append(single_asset)
        
        # 6. Two-asset portfolios (if we have at least 2 assets)
        if self.n_assets >= 2 and weight_bounds[1] >= 0.5:
            for i in range(min(2, self.n_assets)):
                for j in range(i+1, min(i+3, self.n_assets)):
                    two_asset = np.zeros(self.n_assets)
                    two_asset[i] = 0.6
                    two_asset[j] = 0.4
                    if np.all(two_asset >= weight_bounds[0]) and np.all(two_asset <= weight_bounds[1]):
                        initial_guesses.append(two_asset)
        
        # 7. If no valid guesses found, create a fallback
        if not initial_guesses:
            # Create a valid guess within bounds
            fallback = np.full(self.n_assets, (weight_bounds[0] + weight_bounds[1]) / 2)
            fallback = fallback / np.sum(fallback)
            initial_guesses.append(fallback)
        
        return initial_guesses
    
    def debug_optimization(self, method: str = 'sharpe', verbose: bool = True) -> Dict:
        """
        Debug optimization with detailed output.
        
        Args:
            method: Optimization method to test ('sharpe', 'min_variance', 'risk_parity')
            verbose: Print detailed debugging information
            
        Returns:
            Dictionary with optimization results and diagnostic info
        """
        print(f"\n=== Debugging {method} optimization ===")
        print(f"Number of assets: {self.n_assets}")
        print(f"Asset names: {list(self.returns.columns)}")
        print(f"Mean returns (annualized %): {(self.mean_returns * self.periods_per_year * 100).round(2).to_dict()}")
        
        # Check covariance matrix properties
        cov_eigenvals = np.linalg.eigvals(self.cov_matrix)
        print(f"Covariance matrix condition number: {np.linalg.cond(self.cov_matrix):.2e}")
        print(f"Min eigenvalue: {np.min(cov_eigenvals):.2e}")
        print(f"Is positive definite: {np.all(cov_eigenvals > 0)}")
        
        # Asset volatilities
        asset_vols = np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(self.periods_per_year)
        print(f"Asset volatilities (annualized %): {(asset_vols * 100).round(2)}")
        
        # Run optimization based on method
        try:
            if method == 'sharpe':
                result = self.optimize_sharpe(verbose=verbose)
                print(f"\nOptimal Sharpe portfolio weights: {result}")
                
                # Calculate portfolio metrics
                weights = np.array(list(result.values()))
                port_return = np.dot(weights, self.mean_returns) * self.periods_per_year
                port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.periods_per_year)
                sharpe = (port_return - self.risk_free_rate) / port_vol
                
                print(f"Portfolio return (annualized %): {port_return * 100:.2f}")
                print(f"Portfolio volatility (annualized %): {port_vol * 100:.2f}")
                print(f"Sharpe ratio: {sharpe:.4f}")
                
            elif method == 'min_variance':
                result = self.optimize_min_variance(verbose=verbose)
                print(f"\nMin variance portfolio weights: {result}")
                
                weights = np.array(list(result.values()))
                port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(self.periods_per_year)
                print(f"Portfolio volatility (annualized %): {port_vol * 100:.2f}")
                
            elif method == 'risk_parity':
                result = self.optimize_risk_parity(verbose=verbose)
                print(f"\nRisk parity portfolio weights: {result}")
                
            return {'success': True, 'weights': result, 'method': method}
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {'success': False, 'error': str(e), 'method': method}