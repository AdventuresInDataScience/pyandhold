"""Constraint builders for portfolio optimization."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ConstraintBuilder:
    """Build constraints for portfolio optimization."""
    
    @staticmethod
    def long_only_constraint():
        """All weights must be non-negative."""
        return {
            'type': 'ineq',
            'fun': lambda x: x
        }
    
    @staticmethod
    def max_position_constraint(max_weight: float):
        """Maximum weight per position."""
        def constraint_func(x):
            # Return list of constraints: max_weight - x_i >= 0 for all i
            return [max_weight - xi for xi in x]
        
        # For scipy, we need individual constraints for each position
        constraints = []
        
        def make_constraint(i):
            return {
                'type': 'ineq',
                'fun': lambda x, idx=i: max_weight - x[idx]
            }
        
        # We'll return a function that generates constraints based on the size of x
        return {
            'type': 'max_position',
            'max_weight': max_weight,
            'constraint_generator': make_constraint
        }
    
    @staticmethod
    def min_position_constraint(min_weight: float):
        """Minimum weight per position (for non-zero positions)."""
        def make_constraint(i, threshold=1e-6):
            return {
                'type': 'ineq',
                'fun': lambda x, idx=i: x[idx] - min_weight if x[idx] > threshold else 0.0
            }
        
        return {
            'type': 'min_position',
            'min_weight': min_weight,
            'constraint_generator': make_constraint
        }
    
    @staticmethod
    def sector_constraint(
        sector_mapping: Dict[str, List[int]],
        sector_limits: Dict[str, Tuple[float, float]]
    ):
        """Sector allocation constraints."""
        constraints = []
        
        for sector, indices in sector_mapping.items():
            if sector in sector_limits:
                min_alloc, max_alloc = sector_limits[sector]
                
                # Fix lambda closure issue by using default parameters
                # Min constraint: sector_allocation >= min_alloc
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=indices, min_val=min_alloc: sum(x[i] for i in idx) - min_val
                })
                
                # Max constraint: sector_allocation <= max_alloc
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, idx=indices, max_val=max_alloc: max_val - sum(x[i] for i in idx)
                })
        
        return constraints
    
    @staticmethod
    def cardinality_constraint(
        min_assets: int,
        max_assets: int,
        threshold: float = 0.01
    ):
        """Number of assets constraint (approximate for continuous optimization)."""
        def min_assets_constraint(x):
            return sum(1 for w in x if w > threshold) - min_assets
        
        def max_assets_constraint(x):
            return max_assets - sum(1 for w in x if w > threshold)
        
        return [
            {'type': 'ineq', 'fun': min_assets_constraint},
            {'type': 'ineq', 'fun': max_assets_constraint}
        ]
    
    @staticmethod
    def turnover_constraint(
        current_weights: np.ndarray,
        max_turnover: float
    ):
        """Maximum portfolio turnover constraint."""
        return {
            'type': 'ineq',
            'fun': lambda x: max_turnover - np.sum(np.abs(x - current_weights))
        }
    
    @staticmethod
    def leverage_constraint(
        min_leverage: float = 0.9,
        max_leverage: float = 1.1
    ):
        """Leverage constraints (sum of weights)."""
        return [
            {'type': 'ineq', 'fun': lambda x: np.sum(x) - min_leverage},
            {'type': 'ineq', 'fun': lambda x: max_leverage - np.sum(x)}
        ]
    
    @staticmethod
    def custom_linear_constraint(
        A: np.ndarray,
        b_lower: np.ndarray,
        b_upper: np.ndarray
    ):
        """Custom linear constraints: b_lower <= Ax <= b_upper."""
        constraints = []
        
        for i in range(len(b_lower)):
            # Lower bound
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, row=A[i], bound=b_lower[i]: np.dot(row, x) - bound
            })
            
            # Upper bound
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, row=A[i], bound=b_upper[i]: bound - np.dot(row, x)
            })
        
        return constraints