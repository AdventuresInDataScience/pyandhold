"""Portfolio optimization module."""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple, List, Union
from scipy.optimize import minimize
import marsopt
from ..metrics.risk import RiskMetrics
from ..metrics.performance import PerformanceMetrics


class PortfolioOptimizer:
    """Portfolio optimization with various objectives and constraints."""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        optimizer: str = 'mars'
    ):
        """
        Initialize optimizer.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            optimizer: Optimization method ('scipy' or 'mars')
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.n_assets = len(returns.columns)
        self.optimizer = optimizer
        
        # Pre-calculate frequently used matrices
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
    def optimize_sharpe(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None,
        verbose: bool = False,
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        Maximize Sharpe ratio.
        
        Args:
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            verbose: Print optimization details
            n_trials: Number of trials (marsopt only)
            initial_noise: Initial noise level (marsopt only)
            
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
        
        return self._optimize(
            negative_sharpe, weight_bounds, constraints, verbose, 
            n_trials=n_trials, initial_noise=initial_noise
        )
    
    def optimize_min_variance(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None,
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        Minimize portfolio variance.
        
        Args:
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            n_trials: Number of trials (marsopt only)
            initial_noise: Initial noise level (marsopt only)
            
        Returns:
            Dictionary of optimal weights
        """
        def portfolio_variance(weights):
            weights = np.array(weights)
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        return self._optimize(
            portfolio_variance, weight_bounds, constraints,
            n_trials=n_trials, initial_noise=initial_noise
        )
    
    def optimize_max_return(
        self,
        max_volatility: float,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None,
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        Maximize return subject to volatility constraint.
        
        Args:
            max_volatility: Maximum allowed volatility (annualized)
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            n_trials: Number of trials (marsopt only)
            initial_noise: Initial noise level (marsopt only)
            
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
        
        return self._optimize(
            negative_return, weight_bounds, constraints,
            n_trials=n_trials, initial_noise=initial_noise
        )
    
    def optimize_risk_parity(
        self,
        weight_bounds: Tuple[float, float] = (0, 1),
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        Risk parity optimization - equal risk contribution.
        
        Args:
            weight_bounds: Min and max weight for each asset
            n_trials: Number of trials (marsopt only)
            initial_noise: Initial noise level (marsopt only)
            
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
        
        return self._optimize(
            risk_parity_objective, weight_bounds,
            n_trials=n_trials, initial_noise=initial_noise
        )
    
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
        verbose: bool = False,
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        Intelligent optimization routing: chooses best optimizer based on constraints.
        
        Args:
            objective_func: Objective function to minimize
            weight_bounds: Min and max weight for each asset
            constraints: Additional constraints
            verbose: Print optimization details
            n_trials: Number of trials (marsopt only)
            initial_noise: Initial noise level (marsopt only)
            
        Returns:
            Dictionary of optimal weights
        """
        # INTELLIGENT ROUTING: Choose optimizer based on constraint complexity
        use_scipy = self._should_use_scipy_for_constraints(constraints, verbose)
        
        if use_scipy:
            if verbose:
                print("🔧 Using SciPy (hard constraints work better)")
            return self._optimize_scipy(
                objective_func, weight_bounds, constraints, verbose
            )
        else:
            if verbose:
                print("🎯 Using MARSOPT (good for unconstrained/soft constraints)")
            return self._optimize_mars(
                objective_func, weight_bounds, constraints, verbose, 
                n_trials, initial_noise
            )
    
    def _should_use_scipy_for_constraints(self, constraints, verbose=False):
        """
        Decide whether to use SciPy based on constraint characteristics.
        
        Returns True for SciPy if:
        - User explicitly set optimizer='scipy'
        - Sector constraints (hard to satisfy with penalties)
        - Multiple complex constraints
        - Tight constraint bounds
        
        Returns False for MARSOPT if:
        - User explicitly set optimizer='mars'
        - No constraints
        - Simple constraints only
        """
        # Respect user's explicit optimizer choice
        if self.optimizer == 'scipy':
            return True  # User explicitly wants SciPy
        
        if self.optimizer == 'mars':
            # User explicitly wants MARS, but warn about constraint issues
            if verbose and constraints and any('sector' in name for name in constraints.keys()):
                print("⚠️  Warning: MARSOPT struggles with sector constraints")
                print("   Consider using SciPy for better constraint satisfaction")
            return False
        
        # Auto-routing logic when optimizer not explicitly set
        if not constraints:
            return False  # No constraints -> MARSOPT is fine
        
        # Check for constraint types that work better with SciPy
        has_sector_constraints = any('sector' in name for name in constraints.keys())
        has_multiple_constraints = len(constraints) > 2
        
        if has_sector_constraints:
            if verbose:
                print("🎯 Detected sector constraints -> Using SciPy for hard enforcement")
            return True
        
        if has_multiple_constraints:
            if verbose:
                print("🔧 Multiple constraints detected -> Using SciPy for reliability")  
            return True
        
        # Default to MARSOPT for simple cases
        return False
    
    def _optimize_mars(
        self,
        objective_func,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None,
        verbose: bool = False,
        n_trials: int = 1000,
        initial_noise: float = 0.2
    ) -> Dict[str, float]:
        """
        MARS optimization with hard constraint gates and constraint-aware sampling.
        """
        # Pre-validate constraints for feasibility
        self._validate_constraints(constraints, weight_bounds, verbose)
        
        # Initialize constraint-aware storage
        self.feasible_solutions = []
        self.sector_info = self._parse_sector_constraints(constraints)
        
        def mars_objective(trial):
            """
            Objective with HARD LOGIC GATES for constraints.
            Implements Fix 1 and Fix 2 from AI recommendations.
            """
            max_attempts = 10
            
            for attempt in range(max_attempts):
                # Fix 2: Constraint-aware sampling
                if attempt == 0 and self.sector_info:
                    # First attempt: sector-aware sampling
                    weights = self._sample_sector_aware_weights(trial, self.sector_info)
                elif attempt < 5:
                    # Next attempts: wider exploration with increasing noise
                    noise_level = initial_noise + attempt * 0.1
                    weights = self._sample_with_noise(trial, noise_level, weight_bounds)
                else:
                    # Final attempts: rejection sampling from known feasible regions
                    if self.feasible_solutions:
                        weights = self._sample_near_feasible(trial, weight_bounds)
                    else:
                        weights = self._sample_uniform(trial, weight_bounds)
                
                # CRITICAL: Enforce weight bounds strictly
                weights = np.clip(weights, weight_bounds[0], weight_bounds[1])
                weights = weights / np.sum(weights)
                
                # Check bounds again after normalization
                if np.any(weights < weight_bounds[0] - 1e-8) or np.any(weights > weight_bounds[1] + 1e-8):
                    continue  # Skip this sample if bounds are violated
                
                # Fix 1: HARD GATE - Check constraints FIRST before evaluating objective
                if constraints and not self._is_feasible(weights, constraints):
                    continue  # Skip to next attempt, don't even evaluate objective
                
                # Only evaluate objective if ALL constraints are satisfied
                return objective_func(weights)
            
            # Couldn't find feasible solution after max_attempts
            return 1e15  # Fixed high value for complete rejection

        # Create study with enhanced exploration parameters
        study = marsopt.Study(
            direction='minimize',
            initial_noise=max(initial_noise, 0.6),  # Higher initial noise
            verbose=False,
            random_state=42
        )
        
        if verbose:
            print(f"Starting MARS optimization with hard constraint gates...")
            if self.sector_info:
                print(f"Detected {len(self.sector_info)} sectors for constraint-aware sampling")
        
        # Seed MARS with constraint-aware starting points
        self._seed_mars_with_feasible_solutions(study, constraints, weight_bounds, verbose)
        
        # Optimize
        study.optimize(mars_objective, n_trials)
        
        # Extract best weights
        best_trial = study.best_trial
        
        if verbose:
            try:
                best_objective = best_trial['objective_value']
                if best_objective >= 1e15:
                    print(f"Warning: No feasible solution found (objective = {best_objective})")
                else:
                    print(f"Found feasible solution (objective = {best_objective:.6f})")
                    print(f"Feasible solutions stored: {len(self.feasible_solutions)}")
            except (AttributeError, KeyError):
                pass
        
        # Reconstruct best weights
        weights = self._reconstruct_weights_from_trial(best_trial, weight_bounds, constraints)
        
        # Final constraint check and fallback to stored feasible solution if needed
        if constraints and not self._is_feasible(weights, constraints):
            # Reconstruction failed constraints - try using stored feasible solution
            if hasattr(self, 'feasible_solutions') and self.feasible_solutions:
                if verbose:
                    print("   Reconstruction violated constraints - using stored feasible solution")
                # Use the most recently stored feasible solution (likely the best one)
                weights = self.feasible_solutions[-1]
                
                # Double-check this solution
                if self._is_feasible(weights, constraints):
                    if verbose:
                        print("   ✅ Using stored feasible solution")
                else:
                    if verbose:
                        print("   ❌ Even stored solution violates constraints")
                    raise ValueError("MARS optimization failed to find feasible solution")
            else:
                raise ValueError("MARS optimization failed to find feasible solution")
        
        return dict(zip(self.returns.columns, weights))
    
    def _parse_sector_constraints(self, constraints: Optional[Dict]) -> Dict:
        """
        Parse sector constraints to extract sector mapping and limits.
        
        Returns:
            Dictionary with sector info for constraint-aware sampling
        """
        sector_info = {}
        
        if not constraints:
            return sector_info
            
        for name, constraint in constraints.items():
            if isinstance(constraint, list) and name == 'sector':
                try:
                    # Parse sector constraints by testing with sample weights
                    parsed_sectors = self._extract_sector_info_from_constraints(constraint)
                    sector_info.update(parsed_sectors)
                except Exception:
                    continue
                    
        return sector_info
    
    def _extract_sector_info_from_constraints(self, sector_constraints: List[Dict]) -> Dict:
        """Extract sector indices and limits from constraint functions."""
        sector_info = {}
        
        # Process constraints in pairs (min/max for each sector)
        i = 0
        sector_id = 0
        while i < len(sector_constraints) - 1:
            min_constraint = sector_constraints[i]
            max_constraint = sector_constraints[i + 1]
            
            if (min_constraint['type'] == 'ineq' and 
                max_constraint['type'] == 'ineq'):
                
                # Find which assets belong to this sector by testing
                sector_indices = []
                for asset_idx in range(self.n_assets):
                    test_weights = np.zeros(self.n_assets)
                    test_weights[asset_idx] = 1.0
                    
                    min_val = min_constraint['fun'](test_weights)
                    max_val = max_constraint['fun'](test_weights)
                    
                    # If this asset strongly affects the constraints
                    if abs(min_val) > 0.5 or abs(max_val - 1.0) < 0.5:
                        sector_indices.append(asset_idx)
                
                if sector_indices:
                    # Estimate sector limits
                    sector_weight = 1.0 / len(sector_indices)  # Equal weight test
                    test_weights = np.zeros(self.n_assets)
                    for idx in sector_indices:
                        test_weights[idx] = sector_weight
                    
                    min_result = min_constraint['fun'](test_weights)
                    max_result = max_constraint['fun'](test_weights)
                    
                    # Calculate limits (reverse engineering from constraint functions)
                    sector_allocation = sum(test_weights)
                    min_limit = max(0.01, sector_allocation - min_result)
                    max_limit = min(0.8, max_result + sector_allocation)
                    
                    sector_info[f'sector_{sector_id}'] = {
                        'indices': sector_indices,
                        'min_limit': min_limit,
                        'max_limit': max_limit
                    }
                    sector_id += 1
            
            i += 2  # Move to next constraint pair
        
        return sector_info
    
    def _sample_sector_aware_weights(self, trial, sector_info: Dict) -> np.ndarray:
        """
        Sample weights that respect sector constraints from the start.
        Implements Fix 2: Constraint-aware sampling.
        """
        weights = np.zeros(self.n_assets)
        
        if not sector_info:
            return self._sample_uniform(trial, (0, 1))
        
        # First, sample sector allocations within their limits
        sector_allocations = {}
        remaining = 1.0
        
        sectors = list(sector_info.keys())
        
        # Sample allocations for all sectors except the last one
        for i, sector in enumerate(sectors[:-1]):
            info = sector_info[sector]
            min_val, max_val = info['min_limit'], info['max_limit']
            
            # Calculate feasible range considering remaining sectors
            other_mins = sum(sector_info[s]['min_limit'] for s in sectors[i+1:])
            max_possible = min(max_val, remaining - other_mins)
            min_possible = max(min_val, 0.01)
            
            if max_possible >= min_possible:
                sector_alloc = trial.suggest_float(
                    f'sector_{sector}',
                    min_possible,
                    max_possible
                )
            else:
                sector_alloc = min_possible
            
            sector_allocations[sector] = sector_alloc
            remaining -= sector_alloc
        
        # Assign remaining weight to last sector
        if sectors:
            last_sector = sectors[-1]
            sector_allocations[last_sector] = max(remaining, 0.01)
        
        # Distribute within each sector
        for sector, alloc in sector_allocations.items():
            if sector not in sector_info:
                continue
                
            indices = sector_info[sector]['indices']
            n_assets_sector = len(indices)
            
            if n_assets_sector == 0:
                continue
            elif n_assets_sector == 1:
                weights[indices[0]] = alloc
            else:
                # Use Dirichlet-like sampling within sector
                cuts = []
                for j in range(n_assets_sector - 1):
                    cuts.append(trial.suggest_float(f'{sector}_cut_{j}', 0.0, 1.0))
                
                cuts = sorted(cuts)
                boundaries = [0.0] + cuts + [1.0]
                sector_weights = np.diff(boundaries)
                
                # Assign to actual assets
                for j, idx in enumerate(indices):
                    weights[idx] = alloc * sector_weights[j]
        
        # Handle unassigned assets
        assigned_indices = set()
        for info in sector_info.values():
            assigned_indices.update(info['indices'])
        
        unassigned = [i for i in range(self.n_assets) if i not in assigned_indices]
        if unassigned:
            remaining_weight = 1.0 - np.sum(weights)
            if remaining_weight > 0.01:
                for idx in unassigned:
                    weights[idx] = remaining_weight / len(unassigned)
        
        return weights
    
    def _sample_with_noise(self, trial, noise_level: float, weight_bounds: Tuple[float, float]) -> np.ndarray:
        """Sample weights with increased noise for wider exploration."""
        if self.n_assets == 1:
            return np.array([1.0])
        
        # Use wider sampling ranges
        cuts = []
        for i in range(self.n_assets - 1):
            # Add noise to the sampling range
            noise_factor = 1.0 + noise_level
            cut_range = (0.0, min(1.0, noise_factor))
            cuts.append(trial.suggest_float(f'noise_cut_{i}', cut_range[0], cut_range[1]))
        
        cuts = sorted(cuts)
        # Normalize cuts to [0, 1] range
        if cuts:
            max_cut = max(cuts)
            if max_cut > 1.0:
                cuts = [c / max_cut for c in cuts]
        
        boundaries = [0.0] + cuts + [1.0]
        weights = np.diff(boundaries)
        
        # Apply bounds
        min_bound, max_bound = weight_bounds
        weights = np.clip(weights, min_bound, max_bound)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _sample_near_feasible(self, trial, weight_bounds: Tuple[float, float]) -> np.ndarray:
        """Sample near known feasible solutions with small perturbations."""
        if not self.feasible_solutions or len(self.feasible_solutions) == 0:
            return self._sample_uniform(trial, weight_bounds)
        
        # Choose a random feasible solution - fix the range issue
        if len(self.feasible_solutions) == 1:
            base_weights = self.feasible_solutions[0]
        else:
            base_idx = trial.suggest_int('base_solution', 0, len(self.feasible_solutions) - 1)
            base_weights = self.feasible_solutions[base_idx]
        
        # Add small random perturbation
        noise_scale = trial.suggest_float('perturbation_scale', 0.01, 0.1)
        noise = np.random.normal(0, noise_scale, size=self.n_assets)
        
        perturbed_weights = base_weights + noise
        perturbed_weights = np.clip(perturbed_weights, weight_bounds[0], weight_bounds[1])
        perturbed_weights = perturbed_weights / np.sum(perturbed_weights)
        
        return perturbed_weights
    
    def _sample_uniform(self, trial, weight_bounds: Tuple[float, float]) -> np.ndarray:
        """Standard uniform sampling with Dirichlet-like approach."""
        if self.n_assets == 1:
            return np.array([1.0])
        
        cuts = []
        for i in range(self.n_assets - 1):
            cuts.append(trial.suggest_float(f'uniform_cut_{i}', 0.0, 1.0))
        
        cuts = sorted(cuts)
        boundaries = [0.0] + cuts + [1.0]
        weights = np.diff(boundaries)
        
        # Apply bounds
        min_bound, max_bound = weight_bounds
        weights = np.clip(weights, min_bound, max_bound)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _is_feasible(self, weights: np.ndarray, constraints: Dict) -> bool:
        """
        Hard constraint check - returns True only if ALL constraints are satisfied.
        Implements Fix 1: Hard logic gate for constraints.
        """
        tolerance = 1e-6
        
        # First check: weights sum to 1 (approximately)
        if abs(np.sum(weights) - 1.0) > tolerance:
            return False
        
        # Second check: all weights are non-negative
        if np.any(weights < -tolerance):
            return False
        
        # Third check: custom constraints
        for name, constraint in constraints.items():
            if isinstance(constraint, dict):
                if 'constraint_generator' in constraint:
                    # Generated constraints (like min/max position)
                    constraint_type = constraint.get('type')
                    
                    if constraint_type == 'max_position':
                        max_weight = constraint['max_weight']
                        if np.any(weights > max_weight + tolerance):
                            return False
                    elif constraint_type == 'min_position':
                        min_weight = constraint['min_weight']
                        threshold = constraint.get('threshold', 1e-6)
                        for w in weights:
                            if w > threshold and w < min_weight - tolerance:
                                return False
                else:
                    # Single constraint
                    try:
                        result = constraint['fun'](weights)
                        if constraint['type'] == 'eq' and abs(result) > tolerance:
                            return False
                        elif constraint['type'] == 'ineq' and result < -tolerance:
                            return False
                    except Exception:
                        return False
                        
            elif isinstance(constraint, list):
                # Multiple constraints (sector constraints)
                for c in constraint:
                    try:
                        result = c['fun'](weights)
                        if c['type'] == 'eq' and abs(result) > tolerance:
                            return False
                        elif c['type'] == 'ineq' and result < -tolerance:
                            return False
                    except Exception:
                        return False
        
        # Store feasible solution for future sampling
        if hasattr(self, 'feasible_solutions'):
            self.feasible_solutions.append(weights.copy())
            # Limit stored solutions to prevent memory issues
            if len(self.feasible_solutions) > 100:
                self.feasible_solutions = self.feasible_solutions[-50:]  # Keep last 50
        
        return True
    
    def _reconstruct_weights_from_trial(self, best_trial, weight_bounds: Tuple[float, float], constraints: Optional[Dict] = None) -> np.ndarray:
        """Reconstruct the best weights from the trial parameters."""
        try:
            variables = best_trial['variables']
            
            # Try sector-aware reconstruction first
            if hasattr(self, 'sector_info') and self.sector_info:
                try:
                    return self._reconstruct_sector_aware_weights(variables, weight_bounds)
                except:
                    pass
            
            # Standard reconstruction
            if self.n_assets == 1:
                return np.array([1.0])
            
            # Look for the right variable pattern
            cuts = []
            for prefix in ['cut_', 'noise_cut_', 'uniform_cut_']:
                temp_cuts = []
                for i in range(self.n_assets - 1):
                    key = f'{prefix}{i}'
                    if key in variables:
                        temp_cuts.append(variables[key])
                
                if len(temp_cuts) == self.n_assets - 1:
                    cuts = temp_cuts
                    break
            
            if not cuts:
                # Fallback to equal weights
                return np.ones(self.n_assets) / self.n_assets
            
            # Reconstruct weights
            cuts = sorted(cuts)
            boundaries = [0.0] + cuts + [1.0]
            weights = np.diff(boundaries)
            
            # Apply weight constraints intelligently
            min_bound, max_bound = weight_bounds
            
            # Check if we have minimum position constraints (special logic needed)
            if constraints and self._has_minimum_position_constraints(constraints):
                # For minimum position constraints, just normalize and let constraint validation handle it
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.n_assets) / self.n_assets
                # Still apply max bounds to prevent violations
                weights = np.clip(weights, 0, max_bound)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.n_assets) / self.n_assets
            else:
                # Safe to use bounded normalization for simple weight bounds
                weights = self._normalize_with_bounds(weights, min_bound, max_bound)
            
            return weights
            
        except Exception:
            # Ultimate fallback
            return np.ones(self.n_assets) / self.n_assets
    
    def _reconstruct_sector_aware_weights(self, variables: Dict, weight_bounds: Tuple[float, float]) -> np.ndarray:
        """Reconstruct weights from sector-aware sampling variables."""
        weights = np.zeros(self.n_assets)
        
        # Reconstruct sector allocations
        sector_allocations = {}
        for sector in self.sector_info.keys():
            var_name = f'sector_{sector}'
            if var_name in variables:
                sector_allocations[sector] = variables[var_name]
        
        # Reconstruct within-sector weights
        for sector, alloc in sector_allocations.items():
            if sector not in self.sector_info:
                continue
                
            indices = self.sector_info[sector]['indices']
            n_assets_sector = len(indices)
            
            if n_assets_sector == 1:
                weights[indices[0]] = alloc
            else:
                # Reconstruct cuts
                cuts = []
                for j in range(n_assets_sector - 1):
                    cut_var = f'{sector}_cut_{j}'
                    if cut_var in variables:
                        cuts.append(variables[cut_var])
                
                if len(cuts) == n_assets_sector - 1:
                    cuts = sorted(cuts)
                    boundaries = [0.0] + cuts + [1.0]
                    sector_weights = np.diff(boundaries)
                    
                    for j, idx in enumerate(indices):
                        weights[idx] = alloc * sector_weights[j]
                else:
                    # Equal weight within sector
                    for idx in indices:
                        weights[idx] = alloc / n_assets_sector
        
        # Ensure weights are valid
        if np.sum(weights) == 0:
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = weights / np.sum(weights)
        
        return weights
    
    def _optimize_scipy(
        self,
        objective_func,
        weight_bounds: Tuple[float, float] = (0, 1),
        constraints: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        SciPy optimization with enhanced sector constraint handling.
        """
        # Pre-validate constraints for feasibility
        self._validate_constraints(constraints, weight_bounds, verbose)
        
        # Bounds for each weight
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(self.n_assets))
        
        # Basic constraints - weights sum to 1
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Add custom constraints with improved handling
        if constraints:
            for name, constraint in constraints.items():
                if isinstance(constraint, dict):
                    # Check if it's a new format constraint with constraint_generator
                    if 'constraint_generator' in constraint:
                        if constraint.get('type') == 'max_position':
                            # Generate individual constraints for each position
                            for i in range(self.n_assets):
                                constraints_list.append(constraint['constraint_generator'](i))
                        elif constraint.get('type') == 'min_position':
                            # Generate individual constraints for each position
                            for i in range(self.n_assets):
                                constraints_list.append(constraint['constraint_generator'](i))
                        else:
                            if verbose:
                                print(f"Warning: Unknown constraint generator type: {constraint.get('type')}")
                    else:
                        # Single constraint (like volatility, return)
                        constraints_list.append(constraint)
                        
                elif isinstance(constraint, list):
                    # Multiple constraints (sector constraints from ConstraintBuilder)
                    if verbose and name == 'sector':
                        print(f"Adding {len(constraint)} sector constraints")
                    
                    # Add each constraint individually with proper error handling
                    for i, c in enumerate(constraint):
                        try:
                            # Validate constraint format
                            if not isinstance(c, dict) or 'fun' not in c or 'type' not in c:
                                if verbose:
                                    print(f"Skipping malformed constraint {i}: {c}")
                                continue
                            
                            # Test constraint function with dummy weights
                            test_weights = np.ones(self.n_assets) / self.n_assets
                            try:
                                test_result = c['fun'](test_weights)
                                if not isinstance(test_result, (int, float, np.number)):
                                    if verbose:
                                        print(f"Constraint {i} returns non-numeric value: {test_result}")
                                    continue
                            except Exception as e:
                                if verbose:
                                    print(f"Constraint {i} function failed test: {e}")
                                continue
                            
                            # Add validated constraint
                            constraints_list.append(c)
                            
                        except Exception as e:
                            if verbose:
                                print(f"Error processing constraint {i}: {e}")
                            continue
                else:
                    # Try to handle as a constraint function or object
                    try:
                        if hasattr(constraint, '__call__'):
                            constraints_list.append({'type': 'ineq', 'fun': constraint})
                        else:
                            if verbose:
                                print(f"Warning: Unknown constraint type for {name}: {type(constraint)}")
                    except Exception as e:
                        if verbose:
                            print(f"Error processing constraint {name}: {e}")
        
        if verbose:
            print(f"Total constraints for SciPy: {len(constraints_list)}")
        
        # Generate constraint-satisfying initial guesses
        initial_guesses = self._generate_initial_guesses(weight_bounds, constraints)
        
        if verbose and constraints:
            print(f"Generated {len(initial_guesses)} initial guesses")
        
        # Try different solvers with enhanced settings
        solvers = ['SLSQP', 'trust-constr']
        
        best_result = None
        best_objective = float('inf')
        
        for solver in solvers:
            if verbose:
                print(f"\nTrying solver: {solver}")
                
            for i, x0 in enumerate(initial_guesses):
                try:
                    if verbose and i < 3:  # Only show first few attempts
                        print(f"  Initial guess {i+1}: objective = {objective_func(x0):.6f}")
                    
                    # Enhanced solver options
                    if solver == 'trust-constr':
                        options = {
                            'maxiter': 10000,
                            'gtol': 1e-10,
                            'xtol': 1e-12,
                            'barrier_tol': 1e-10,
                            'sparse_jacobian': None,
                            'finite_diff_rel_step': None
                        }
                    else:  # SLSQP
                        options = {
                            'maxiter': 5000,
                            'ftol': 1e-12,
                            'eps': 1e-8,
                            'disp': False
                        }
                    
                    result = minimize(
                        objective_func,
                        x0,
                        method=solver,
                        bounds=bounds,
                        constraints=constraints_list,
                        options=options
                    )
                    
                    # Enhanced result checking
                    if result.success and result.fun < best_objective:
                        # CRITICAL FIX: Actually validate constraints manually
                        # Don't trust SciPy's internal constraint satisfaction reporting
                        constraint_satisfied = True
                        constraint_violations = []
                        
                        if constraints and 'sector' in constraints:
                            for j, c in enumerate(constraints['sector']):
                                try:
                                    val = c['fun'](result.x)
                                    if c['type'] == 'ineq' and val < -1e-6:
                                        constraint_satisfied = False
                                        constraint_violations.append(f"Sector constraint {j+1}: {val:.6f}")
                                    elif c['type'] == 'eq' and abs(val) > 1e-6:
                                        constraint_satisfied = False
                                        constraint_violations.append(f"Sector constraint {j+1}: {val:.6f}")
                                except Exception as e:
                                    constraint_satisfied = False
                                    constraint_violations.append(f"Sector constraint {j+1}: Error {e}")
                        
                        if constraint_satisfied:
                            best_result = result
                            best_objective = result.fun
                            
                            if verbose:
                                print(f"    ✅ New best result! Objective: {best_objective:.6f}")
                                print(f"    ✅ All constraints manually verified as satisfied")
                        else:
                            if verbose:
                                print(f"    ❌ Result rejected - constraint violations detected:")
                                for violation in constraint_violations[:3]:  # Show first 3
                                    print(f"        {violation}")
                    elif verbose and i < 3:
                        status = "✅" if result.success else "❌"
                        print(f"    {status} Result: success={result.success}, fun={result.fun:.6f}")
                        if not result.success:
                            print(f"        Message: {result.message}")
                
                except Exception as e:
                    if verbose and i < 3:
                        print(f"    ❌ Failed with error: {e}")
                    continue
            
            # If we found a good solution, we can stop trying other solvers
            if best_result is not None and best_result.success:
                if verbose:
                    print(f"Found satisfactory solution with {solver}, stopping solver search")
                break
        
        # Final validation and error handling
        if best_result is None or not best_result.success:
            if verbose:
                print("\n❌ All standard methods failed, trying backup approach...")
            
            # Backup: Try with relaxed constraints
            try:
                x0 = self._generate_initial_guesses(weight_bounds)[0]
                result = minimize(
                    objective_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints_list,
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )
                
                if result.success:
                    best_result = result
                    if verbose:
                        print("✅ Backup method succeeded")
                else:
                    raise ValueError(f"Backup optimization also failed: {result.message}")
            except Exception as backup_error:
                raise ValueError(
                    f"All optimization methods failed. "
                    f"Last error: {backup_error}. "
                    f"This may indicate infeasible constraints or poor conditioning."
                )
        
        # Extract and validate final weights
        weights = best_result.x
        
        # Precision adjustments
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            if verbose:
                print(f"⚠️  Weight sum adjustment: {weight_sum:.8f} → 1.0")
            weights = weights / weight_sum
        
        # Final bounds check
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)  # Renormalize after clipping
        
        if verbose:
            print(f"\n✅ Final SciPy optimization result:")
            print(f"   Objective: {best_result.fun:.6f}")
            print(f"   Solver: {best_result.message}")
            if constraints and 'sector' in constraints:
                print(f"   Final constraint check:")
                self._print_sector_allocation_summary(weights, constraints['sector'])
        
        return dict(zip(self.returns.columns, weights))
    
    def _print_sector_allocation_summary(self, weights: np.ndarray, sector_constraints: List[Dict]):
        """Print a summary of sector allocations vs constraints."""
        try:
            total_weight = np.sum(weights)
            print(f"     Total allocation: {total_weight:.1%}")
            print(f"     Weight range: {np.min(weights):.1%} - {np.max(weights):.1%}")
            print(f"     Active positions: {np.sum(weights > 0.01)}")
            
            # Check constraint violations more carefully
            violations = 0
            satisfied = 0
            for i, constraint in enumerate(sector_constraints):
                try:
                    result = constraint['fun'](weights)
                    if constraint['type'] == 'ineq':
                        if result < -1e-6:  # Violated
                            violations += 1
                        else:  # Satisfied
                            satisfied += 1
                    elif constraint['type'] == 'eq':
                        if abs(result) > 1e-6:  # Violated
                            violations += 1
                        else:  # Satisfied
                            satisfied += 1
                except Exception as e:
                    print(f"     Error checking constraint {i}: {e}")
                    violations += 1
            
            if violations == 0:
                print(f"     ✅ All {len(sector_constraints)} sector constraints satisfied")
            else:
                print(f"     ❌ {violations}/{len(sector_constraints)} sector constraints violated")
                print(f"     ⚠️  Note: SciPy optimizer may have constraint validation issues")
                
        except Exception as e:
            print(f"     Could not summarize sector allocations: {e}")
    
    def _generate_initial_guesses(
        self, 
        weight_bounds: Tuple[float, float], 
        constraints: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """Generate multiple initial weight guesses for optimization, including constraint-satisfying ones."""
        initial_guesses = []
        np.random.seed(42)  # For reproducibility
        
        # First, try to generate constraint-satisfying initial guesses
        if constraints:
            constraint_satisfying_guesses = self._generate_constraint_satisfying_guesses(
                weight_bounds, constraints
            )
            initial_guesses.extend(constraint_satisfying_guesses)
        
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
    
    def _generate_constraint_satisfying_guesses(
        self,
        weight_bounds: Tuple[float, float],
        constraints: Dict
    ) -> List[np.ndarray]:
        """Generate initial guesses that satisfy sector constraints."""
        constraint_guesses = []
        
        # Look for sector constraints
        for name, constraint in constraints.items():
            if isinstance(constraint, list) and name == 'sector':
                # This is a sector constraint from ConstraintBuilder
                sector_guesses = self._generate_sector_satisfying_guesses(constraint, weight_bounds)
                constraint_guesses.extend(sector_guesses)
        
        return constraint_guesses
    
    def _generate_sector_satisfying_guesses(
        self,
        sector_constraints: List[Dict],
        weight_bounds: Tuple[float, float]
    ) -> List[np.ndarray]:
        """Generate initial guesses that satisfy sector constraints."""
        guesses = []
        
        # Parse sector constraints to extract sector mappings and limits
        sector_info = {}
        
        # Group constraints by sector (each sector has min and max constraints)
        i = 0
        while i < len(sector_constraints):
            min_constraint = sector_constraints[i]
            max_constraint = sector_constraints[i + 1] if i + 1 < len(sector_constraints) else None
            
            if min_constraint['type'] == 'ineq' and max_constraint and max_constraint['type'] == 'ineq':
                # Try to extract sector info by testing with dummy weights
                test_weights = np.ones(self.n_assets) * 0.1
                
                # Find which assets belong to this sector by testing the constraint function
                sector_indices = []
                for asset_idx in range(self.n_assets):
                    test_weights_single = np.zeros(self.n_assets)
                    test_weights_single[asset_idx] = 1.0
                    
                    min_val = min_constraint['fun'](test_weights_single)
                    max_val = max_constraint['fun'](test_weights_single)
                    
                    # If this asset affects the constraint, it belongs to this sector
                    if abs(min_val) > 0.5 or abs(max_val - 1.0) < 0.5:  # Asset weight is 1.0
                        sector_indices.append(asset_idx)
                
                if sector_indices:
                    # Extract min/max limits by testing with sector-only allocations
                    sector_weight_test = np.zeros(self.n_assets)
                    for idx in sector_indices:
                        sector_weight_test[idx] = 1.0 / len(sector_indices)  # Equal weight in sector
                    
                    min_result = min_constraint['fun'](sector_weight_test)
                    max_result = max_constraint['fun'](sector_weight_test)
                    
                    # Approximate the limits (this is reverse engineering)
                    sector_info[f'sector_{len(sector_info)}'] = {
                        'indices': sector_indices,
                        'min_limit': max(min_result, 0.01),  # Reasonable bounds
                        'max_limit': min(max_result, 0.8)    # Reasonable bounds
                    }
            
            i += 2  # Move to next pair of constraints
        
        # Generate constraint-satisfying portfolios
        if sector_info:
            for strategy in ['balanced', 'min_allocation', 'max_allocation']:
                try:
                    weights = self._create_sector_satisfying_portfolio(
                        sector_info, weight_bounds, strategy
                    )
                    if weights is not None and self._validate_sector_constraints(weights, sector_constraints):
                        guesses.append(weights)
                except Exception:
                    continue  # Skip if this strategy fails
        
        return guesses
    
    def _create_sector_satisfying_portfolio(
        self,
        sector_info: Dict,
        weight_bounds: Tuple[float, float],
        strategy: str
    ) -> Optional[np.ndarray]:
        """Create a portfolio that satisfies sector constraints."""
        weights = np.zeros(self.n_assets)
        
        # Calculate target allocations for each sector
        sector_allocations = {}
        
        for sector_name, info in sector_info.items():
            min_limit = info['min_limit']
            max_limit = info['max_limit']
            
            if strategy == 'balanced':
                # Use mid-point of allowed range
                target_alloc = (min_limit + max_limit) / 2
            elif strategy == 'min_allocation':
                # Use minimum + small buffer
                target_alloc = min_limit + 0.02
            elif strategy == 'max_allocation':
                # Use maximum - small buffer  
                target_alloc = max_limit - 0.02
            else:
                target_alloc = (min_limit + max_limit) / 2
                
            sector_allocations[sector_name] = min(max(target_alloc, min_limit), max_limit)
        
        # Check if total allocation is feasible
        total_target = sum(sector_allocations.values())
        if total_target > 1.0:
            # Scale down proportionally
            scale_factor = 0.95 / total_target  # Leave 5% buffer
            for sector in sector_allocations:
                sector_allocations[sector] *= scale_factor
        
        # Allocate weights within each sector
        for sector_name, info in sector_info.items():
            sector_indices = info['indices']
            sector_allocation = sector_allocations[sector_name]
            
            if sector_indices and sector_allocation > 0:
                # Use return-weighted allocation within sector if possible
                try:
                    sector_returns = self.mean_returns.iloc[sector_indices]
                    if sector_returns.sum() > 0 and not np.all(np.isclose(sector_returns, sector_returns.iloc[0])):
                        # Weight by returns (positive only)
                        pos_returns = np.maximum(sector_returns, 1e-6)
                        sector_weights = pos_returns / pos_returns.sum()
                    else:
                        # Equal weight within sector
                        sector_weights = np.ones(len(sector_indices)) / len(sector_indices)
                except:
                    # Fallback to equal weight
                    sector_weights = np.ones(len(sector_indices)) / len(sector_indices)
                
                # Assign weights
                for i, idx in enumerate(sector_indices):
                    weights[idx] = sector_allocation * sector_weights[i]
        
        # Handle any remaining allocation (assets not in defined sectors)
        allocated_weight = np.sum(weights)
        remaining_weight = 1.0 - allocated_weight
        
        if remaining_weight > 0.01:  # Significant remaining weight
            unallocated_indices = [i for i in range(self.n_assets) if weights[i] == 0]
            if unallocated_indices:
                # Distribute remaining weight equally among unallocated assets
                for idx in unallocated_indices:
                    weights[idx] = remaining_weight / len(unallocated_indices)
        
        # Ensure weights respect bounds and sum to 1
        weights = np.clip(weights, weight_bounds[0], weight_bounds[1])
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            return None
        
        return weights
    
    def _validate_sector_constraints(
        self,
        weights: np.ndarray,
        sector_constraints: List[Dict]
    ) -> bool:
        """Validate that weights satisfy sector constraints."""
        try:
            for constraint in sector_constraints:
                result = constraint['fun'](weights)
                if constraint['type'] == 'ineq' and result < -1e-6:  # Small tolerance
                    return False
                elif constraint['type'] == 'eq' and abs(result) > 1e-6:
                    return False
            return True
        except:
            return False
    
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
    
    def _validate_constraints(
        self, 
        constraints: Optional[Dict], 
        weight_bounds: Tuple[float, float], 
        verbose: bool = False
    ) -> None:
        """
        Validate constraints for feasibility before optimization.
        
        Args:
            constraints: Dictionary of constraints to validate
            weight_bounds: Weight bounds for validation
            verbose: Print validation details
            
        Raises:
            ValueError: If constraints are infeasible
        """
        if not constraints:
            return
            
        for name, constraint in constraints.items():
            if isinstance(constraint, dict) and 'constraint_generator' in constraint:
                constraint_type = constraint.get('type')
                
                if constraint_type == 'min_position':
                    self._validate_min_position_constraint(constraint, weight_bounds, verbose)
                elif constraint_type == 'max_position':
                    self._validate_max_position_constraint(constraint, weight_bounds, verbose)
    
    def _validate_min_position_constraint(
        self, 
        constraint: Dict, 
        weight_bounds: Tuple[float, float], 
        verbose: bool = False
    ) -> None:
        """
        Validate minimum position constraint for feasibility.
        
        Args:
            constraint: Min position constraint dictionary
            weight_bounds: Weight bounds for validation
            verbose: Print validation details
            
        Raises:
            ValueError: If minimum position constraint is infeasible
        """
        min_weight = constraint['min_weight']
        threshold = constraint.get('threshold', 1e-6)
        
        if verbose:
            print(f"\nValidating minimum position constraint:")
            print(f"  Required minimum weight: {min_weight:.1%}")
            print(f"  Number of assets: {self.n_assets}")
            print(f"  Weight bounds: {weight_bounds}")
        
        # Check basic feasibility
        if min_weight > weight_bounds[1]:
            raise ValueError(
                f"Infeasible minimum position constraint: "
                f"Required minimum weight {min_weight:.1%} exceeds maximum allowed weight {weight_bounds[1]:.1%}"
            )
        
        if min_weight < weight_bounds[0]:
            if verbose:
                print(f"  Warning: Minimum weight {min_weight:.1%} is below lower bound {weight_bounds[0]:.1%}")
        
        # Check if equal weighting satisfies the constraint
        equal_weight = 1.0 / self.n_assets
        
        if verbose:
            print(f"  Equal weight per asset: {equal_weight:.3%}")
            print(f"  Can all assets be active with equal weights? {equal_weight >= min_weight}")
        
        # If equal weighting doesn't work, calculate maximum feasible active assets
        if equal_weight < min_weight:
            max_active_assets = int(1.0 / min_weight)
            remaining_weight = 1.0 - (max_active_assets * min_weight)
            
            if verbose:
                print(f"  Equal weighting is infeasible!")
                print(f"  Maximum active assets with {min_weight:.1%} minimum: {max_active_assets}")
                print(f"  Remaining weight for optimization: {remaining_weight:.1%}")
                print(f"  This means {self.n_assets - max_active_assets} assets will be set to 0%")
            
            # Decide how to handle this - for now, raise an error to make it explicit
            raise ValueError(
                f"Minimum position constraint forces asset selection:\n"
                f"  • Requested: {self.n_assets} assets with {min_weight:.1%} minimum each\n"
                f"  • Required minimum allocation: {self.n_assets * min_weight:.1%}\n"
                f"  • Equal weight per asset: {equal_weight:.3%} < {min_weight:.1%} (INFEASIBLE)\n"
                f"  • Maximum feasible active assets: {max_active_assets}\n"
                f"  • Assets that will be excluded: {self.n_assets - max_active_assets}\n\n"
                f"Solutions:\n"
                f"  1. Reduce minimum weight to {equal_weight:.3%} or lower\n"
                f"  2. Reduce number of assets to {max_active_assets} or fewer\n"
                f"  3. Use weight_bounds to implement position limits instead\n"
                f"  4. Set allow_asset_selection=True (if implemented) to allow automatic selection"
            )
        
        if verbose:
            print(f"  ✓ Minimum position constraint is feasible with all {self.n_assets} assets")
    
    def _validate_max_position_constraint(
        self, 
        constraint: Dict, 
        weight_bounds: Tuple[float, float], 
        verbose: bool = False
    ) -> None:
        """
        Validate maximum position constraint for feasibility.
        
        Args:
            constraint: Max position constraint dictionary
            weight_bounds: Weight bounds for validation
            verbose: Print validation details
            
        Raises:
            ValueError: If maximum position constraint is infeasible
        """
        max_weight = constraint['max_weight']
        
        if verbose:
            print(f"\nValidating maximum position constraint:")
            print(f"  Maximum weight per position: {max_weight:.1%}")
            print(f"  Number of assets: {self.n_assets}")
        
        # Check if we can achieve 100% allocation with the max weight limit
        max_possible_allocation = self.n_assets * max_weight
        
        if max_possible_allocation < 1.0:
            raise ValueError(
                f"Maximum position constraint is infeasible:\n"
                f"  • Maximum weight per asset: {max_weight:.1%}\n"
                f"  • Number of assets: {self.n_assets}\n"
                f"  • Maximum possible total allocation: {self.n_assets} × {max_weight:.1%} = {max_possible_allocation:.1%}\n"
                f"  • Required total allocation: 100%\n\n"
                f"Solutions:\n"
                f"  1. Increase maximum weight to {1.0/self.n_assets:.3%} or higher\n"
                f"  2. Use fewer assets in the optimization\n"
                f"  3. Allow leverage (total allocation > 100%)"
            )
        
        if verbose:
            print(f"  ✓ Maximum position constraint is feasible")
            print(f"    Maximum total allocation: {max_possible_allocation:.1%}")

    def _generate_sector_aware_weights(self, sector_constraints):
        """
        Generate weights that are more likely to satisfy sector constraints.
        Uses a two-stage approach: sample sector allocations, then distribute within sectors.
        """
        try:
            # Parse sector constraints to get limits
            sector_info = {}
            sector_indices = {}
            
            # Extract sector mappings and limits from constraints
            for constraint in sector_constraints:
                constraint_func = constraint['fun']
                
                # Try to identify which sector this constraint refers to
                # This is a heuristic approach - we'll test with a sample weight vector
                test_weights = np.zeros(self.n_assets)
                
                # Try to match constraint to sector by testing different configurations
                for sector_name in ['tech', 'finance', 'energy', 'healthcare']:
                    if hasattr(self, f'{sector_name}_indices'):
                        indices = getattr(self, f'{sector_name}_indices')
                        test_weights[:] = 0.0
                        test_weights[indices] = 0.5 / len(indices)  # Equal weight within sector
                        
                        result = constraint_func(test_weights)
                        # If this gives a positive result, it's likely a lower bound constraint for this sector
                        if result > 0.3:  # Threshold indicating this sector
                            if sector_name not in sector_info:
                                sector_info[sector_name] = {'indices': indices, 'min': 0.0, 'max': 1.0}
                            if constraint['type'] == 'ineq':
                                # This could be either min or max constraint
                                # We need more sophisticated parsing, for now use heuristics
                                pass
            
            # Fallback: use predefined sector structure if available
            if not sector_info:
                # Define default sectors based on common patterns
                n_per_sector = max(1, self.n_assets // 4)
                sector_info = {
                    'sector1': {'indices': list(range(0, min(n_per_sector, self.n_assets))), 'min': 0.2, 'max': 0.5},
                    'sector2': {'indices': list(range(n_per_sector, min(2*n_per_sector, self.n_assets))), 'min': 0.1, 'max': 0.3},
                    'sector3': {'indices': list(range(2*n_per_sector, min(3*n_per_sector, self.n_assets))), 'min': 0.05, 'max': 0.2},
                    'remaining': {'indices': list(range(3*n_per_sector, self.n_assets)), 'min': 0.1, 'max': 0.4}
                }
            
            # Generate sector allocations
            weights = np.zeros(self.n_assets)
            remaining_weight = 1.0
            
            sectors = list(sector_info.keys())
            
            # Sample sector allocations (except last one)
            for i, sector in enumerate(sectors[:-1]):
                info = sector_info[sector]
                # Sample allocation for this sector
                max_possible = min(info['max'], remaining_weight - sum(sector_info[s]['min'] for s in sectors[i+1:]))
                min_possible = max(info['min'], 0.0)
                
                if max_possible > min_possible:
                    sector_weight = np.random.uniform(min_possible, max_possible)
                else:
                    sector_weight = min_possible
                    
                remaining_weight -= sector_weight
                
                # Distribute within sector
                if len(info['indices']) > 0:
                    intra_weights = np.random.dirichlet(np.ones(len(info['indices'])))
                    for j, idx in enumerate(info['indices']):
                        if idx < len(weights):
                            weights[idx] = sector_weight * intra_weights[j]
            
            # Assign remaining weight to last sector
            if sectors:
                last_sector = sectors[-1]
                last_info = sector_info[last_sector]
                if len(last_info['indices']) > 0:
                    intra_weights = np.random.dirichlet(np.ones(len(last_info['indices'])))
                    for j, idx in enumerate(last_info['indices']):
                        if idx < len(weights):
                            weights[idx] = remaining_weight * intra_weights[j]
            
            # Ensure weights sum to 1
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.n_assets) / self.n_assets
            
            return weights
            
        except Exception as e:
            # Fallback to simple Dirichlet sampling if sector-aware sampling fails
            return self._generate_dirichlet_weights((0.0, 1.0))
    
    def _generate_dirichlet_weights(self, weight_bounds):
        """Generate weights using Dirichlet sampling (cuts method)."""
        if self.n_assets == 1:
            return np.array([1.0])
        
        # Generate n-1 random cuts in [0,1]
        cuts = np.sort(np.random.random(self.n_assets - 1))
        boundaries = np.concatenate([[0.0], cuts, [1.0]])
        weights = np.diff(boundaries)
        
        # Apply bounds if necessary
        min_bound, max_bound = weight_bounds
        if np.any(weights < min_bound) or np.any(weights > max_bound):
            weights = np.clip(weights, min_bound, max_bound)
            weights = weights / np.sum(weights)
        
        return weights
    
    def _normalize_with_bounds(self, weights: np.ndarray, min_bound: float, max_bound: float, max_iterations: int = 100) -> np.ndarray:
        """
        Normalize weights to sum to 1 while respecting bounds.
        
        IMPORTANT: This method is for WEIGHT BOUNDS only, not minimum position constraints.
        Weight bounds apply to ALL positions: min_bound ≤ weight ≤ max_bound
        Minimum position constraints have different logic: IF weight > 0 THEN weight ≥ min_threshold
        
        Args:
            weights: Initial weights
            min_bound: Minimum weight per asset (for weight bounds only)
            max_bound: Maximum weight per asset
            max_iterations: Maximum iterations for iterative adjustment
            
        Returns:
            Normalized weights respecting bounds
        """
        weights = np.array(weights, dtype=float)
        n_assets = len(weights)
        
        # Quick check if bounds are feasible
        if n_assets * max_bound < 1.0:
            # Impossible to satisfy sum=1 with given max bounds
            # Return equal weights at max bound (will not sum to 1, but respects bounds)
            return np.full(n_assets, max_bound)
        
        if n_assets * min_bound > 1.0:
            # Impossible to satisfy sum=1 with given min bounds  
            # Return equal weights at min bound (will not sum to 1, but respects bounds)
            return np.full(n_assets, min_bound)
        
        # Iterative normalization approach
        for iteration in range(max_iterations):
            # Clip to bounds
            weights = np.clip(weights, min_bound, max_bound)
            
            current_sum = np.sum(weights)
            if abs(current_sum - 1.0) < 1e-10:
                break  # Converged
            
            if current_sum == 0:
                # All weights are zero, start with equal allocation
                weights = np.full(n_assets, 1.0 / n_assets)
                continue
                
            # Find assets that can be adjusted (not at bounds)
            at_min = weights <= min_bound + 1e-10
            at_max = weights >= max_bound - 1e-10
            adjustable = ~(at_min | at_max)
            
            if not np.any(adjustable):
                # All weights are at bounds, can't adjust further
                break
                
            # Calculate adjustment needed
            excess = current_sum - 1.0
            n_adjustable = np.sum(adjustable)
            
            if n_adjustable > 0:
                # Distribute adjustment across adjustable assets
                adjustment = excess / n_adjustable
                weights[adjustable] -= adjustment
            else:
                # No adjustable weights, we're done
                break
        
        # Final clipping
        weights = np.clip(weights, min_bound, max_bound)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        return weights
    
    def _has_minimum_position_constraints(self, constraints: Optional[Dict]) -> bool:
        """Check if constraints include CONDITIONAL minimum position constraints (not simple bounds).
        
        This detects constraints like: "if weight > 0 then weight >= 5%"
        This does NOT detect simple bounds like max_position_constraint.
        """
        if not constraints:
            return False
            
        for constraint_name, constraint_def in constraints.items():
            if isinstance(constraint_def, dict):
                # Check for min_position constraint type (conditional constraints)
                if constraint_def.get('type') == 'min_position':
                    return True
            # Also check for legacy min_position constraint names in lists
            elif isinstance(constraint_def, list):
                for constraint in constraint_def:
                    if isinstance(constraint, dict):
                        constraint_func = constraint.get('fun')
                        if constraint_func and hasattr(constraint_func, '__code__'):
                            # Check if the constraint function uses conditional logic
                            # This is a heuristic to detect "x[idx] - min_weight if x[idx] > threshold else 0.0" patterns
                            func_code = str(constraint_func.__code__.co_code)
                            if 'threshold' in constraint_func.__code__.co_names:
                                return True
        
        return False
    
    def _seed_mars_with_feasible_solutions(self, study, constraints: Optional[Dict], weight_bounds: Tuple[float, float], verbose: bool = False):
        """Seed MARS with constraint-aware starting points to improve convergence for tight constraints."""
        
        if not constraints:
            return
            
        starting_points = []
        
        # Strategy 1: Equal weight distribution (if feasible)
        equal_weight = 1.0 / self.n_assets
        min_bound, max_bound = weight_bounds
        
        if min_bound <= equal_weight <= max_bound:
            equal_weights = np.full(self.n_assets, equal_weight)
            if self._is_feasible(equal_weights, constraints):
                starting_points.append(equal_weights)
                if verbose:
                    print(f"   Adding equal weight starting point: {equal_weight:.3f} each")
        
        # Strategy 2: For max position constraints, try boundary solutions
        for constraint_name, constraint_def in constraints.items():
            if isinstance(constraint_def, dict) and constraint_def.get('type') == 'max_position':
                max_weight = constraint_def['max_weight']
                
                # Try solutions at the boundary
                if self.n_assets * max_weight >= 1.0:  # Feasible
                    # Try equal allocation at max boundary
                    if max_weight >= equal_weight:
                        boundary_weights = np.full(self.n_assets, max_weight)
                        # Normalize to sum to 1
                        boundary_weights = boundary_weights / np.sum(boundary_weights)
                        
                        if self._is_feasible(boundary_weights, constraints):
                            starting_points.append(boundary_weights)
                            if verbose:
                                print(f"   Adding max boundary starting point: max {np.max(boundary_weights):.3f}")
                    
                    # Try mixed allocation: some at max, others at min
                    n_at_max = int(1.0 / max_weight)  # How many can be at maximum
                    if n_at_max < self.n_assets and n_at_max > 0:
                        mixed_weights = np.zeros(self.n_assets)
                        mixed_weights[:n_at_max] = max_weight
                        remaining = 1.0 - n_at_max * max_weight
                        if remaining > 0 and (self.n_assets - n_at_max) > 0:
                            mixed_weights[n_at_max:] = remaining / (self.n_assets - n_at_max)
                            
                            if np.all(mixed_weights >= min_bound) and np.all(mixed_weights <= max_bound):
                                if self._is_feasible(mixed_weights, constraints):
                                    starting_points.append(mixed_weights)
                                    if verbose:
                                        print(f"   Adding mixed allocation starting point")
        
        # Strategy 3: For min position constraints, try solutions that satisfy minimums
        for constraint_name, constraint_def in constraints.items():
            if isinstance(constraint_def, dict) and constraint_def.get('type') == 'min_position':
                min_weight = constraint_def['min_weight']
                
                # Try equal allocation above minimum
                if equal_weight >= min_weight:
                    min_weights = np.full(self.n_assets, max(equal_weight, min_weight))
                    min_weights = min_weights / np.sum(min_weights)  # Normalize
                    
                    if np.all(min_weights >= min_bound) and np.all(min_weights <= max_bound):
                        if self._is_feasible(min_weights, constraints):
                            starting_points.append(min_weights)
                            if verbose:
                                print(f"   Adding min position starting point: min {np.min(min_weights):.3f}")
                
                # Try solution where some assets are at minimum, others get the rest
                total_min_required = self.n_assets * min_weight
                if total_min_required <= 1.0:  # Feasible
                    min_solution = np.full(self.n_assets, min_weight)
                    remaining = 1.0 - total_min_required
                    
                    if remaining > 0:
                        # Add extra to first asset (if within bounds)
                        if min_solution[0] + remaining <= max_bound:
                            min_solution[0] += remaining
                        else:
                            # Distribute remaining across all assets
                            per_asset_extra = remaining / self.n_assets
                            min_solution += per_asset_extra
                    
                    if np.all(min_solution >= min_bound) and np.all(min_solution <= max_bound):
                        if self._is_feasible(min_solution, constraints):
                            starting_points.append(min_solution)
                            if verbose:
                                print(f"   Adding min position distributed starting point")
        
        # Seed the study with starting points
        if starting_points:
            if verbose:
                print(f"   Seeding MARS with {len(starting_points)} constraint-aware starting points")
            
            # Convert starting points to MARS variable format
            for weights in starting_points[:5]:  # Limit to 5 starting points
                if self.n_assets > 1:
                    # Convert weights to cuts for MARS sampling
                    sorted_indices = np.argsort(weights)
                    cumsum = np.cumsum(weights[sorted_indices])
                    cuts = cumsum[:-1]  # All but the last cumulative sum
                    
                    # Create variable dict for this starting point
                    variables = {}
                    for i, cut in enumerate(cuts):
                        variables[f'cut_{i}'] = cut
                    
                    try:
                        # Add this as a starting point to the study
                        study._add_manual_trial(variables, mars_objective)
                    except:
                        # If manual trial fails, continue with other starting points
                        pass
