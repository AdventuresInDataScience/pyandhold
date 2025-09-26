#!/usr/bin/env python3
"""
Test script for the new regularized minimum position constraint.

This demonstrates the difference between:
1. Regular minimum constraint: enforces min weight for all active positions
2. Regularized minimum constraint: enforces either 0% or minimum weight (no small positions)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder


def generate_sample_returns(n_assets=10, n_periods=252):
    """Generate sample returns data for testing."""
    np.random.seed(42)
    
    # Asset tickers
    tickers = [f"ASSET_{i+1}" for i in range(n_assets)]
    
    # Generate some correlated returns
    mean_returns = np.random.uniform(0.08, 0.15, n_assets) / 252  # Daily returns
    volatilities = np.random.uniform(0.15, 0.25, n_assets) / np.sqrt(252)  # Daily vols
    
    # Create correlation matrix
    correlations = np.random.uniform(0.1, 0.4, (n_assets, n_assets))
    correlations = (correlations + correlations.T) / 2  # Make symmetric
    np.fill_diagonal(correlations, 1.0)
    
    # Generate returns
    returns = []
    for i in range(n_periods):
        daily_returns = np.random.multivariate_normal(mean_returns, np.diag(volatilities**2))
        returns.append(daily_returns)
    
    return pd.DataFrame(returns, columns=tickers)


def test_constraint_comparison():
    """Compare regular vs regularized minimum constraints."""
    
    print("=" * 80)
    print("TESTING REGULARIZED MINIMUM POSITION CONSTRAINT")
    print("=" * 80)
    
    # Generate sample data
    returns_data = generate_sample_returns(n_assets=8, n_periods=252)
    print(f"Generated returns data: {returns_data.shape[0]} periods, {returns_data.shape[1]} assets")
    print(f"Asset names: {list(returns_data.columns)}")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        returns=returns_data,
        risk_free_rate=0.03,
        periods_per_year=252,
        optimizer='scipy'  # Use SciPy for better constraint handling
    )
    
    constraint_builder = ConstraintBuilder()
    min_weight_limit = 0.05  # 5% minimum
    
    print(f"\nOptimization Setup:")
    print(f"  Risk-free rate: 3.0%")
    print(f"  Minimum weight threshold: {min_weight_limit:.1%}")
    print(f"  Using SciPy optimizer for better constraint handling")
    
    # Test 1: Unconstrained optimization
    print(f"\n" + "=" * 60)
    print("TEST 1: UNCONSTRAINED OPTIMIZATION (BASELINE)")
    print("=" * 60)
    
    try:
        unconstrained_weights = optimizer.optimize_sharpe()
        print("✅ Unconstrained optimization successful")
        
        weights_array = np.array(list(unconstrained_weights.values()))
        print(f"\nUnconstrained Results:")
        print(f"  Portfolio weights range: {np.min(weights_array):.3%} to {np.max(weights_array):.3%}")
        print(f"  Number of positions > 1%: {np.sum(weights_array > 0.01)}")
        print(f"  Number of positions < 1%: {np.sum((weights_array > 0.0001) & (weights_array <= 0.01))}")
        print(f"  Effectively zero positions: {np.sum(weights_array <= 0.0001)}")
        
        # Show all positions
        print(f"\nDetailed positions:")
        for ticker, weight in unconstrained_weights.items():
            status = "ZERO" if weight <= 0.0001 else ("SMALL" if weight <= 0.01 else "ACTIVE")
            print(f"    {ticker}: {weight:7.3%} ({status})")
            
    except Exception as e:
        print(f"❌ Unconstrained optimization failed: {e}")
        return
    
    # Test 2: Regular minimum constraint
    print(f"\n" + "=" * 60)
    print("TEST 2: REGULAR MINIMUM POSITION CONSTRAINT")
    print("=" * 60)
    print(f"Constraint: Any position > 0.01% must be >= {min_weight_limit:.1%}")
    
    try:
        min_pos_constraint = constraint_builder.min_position_constraint(min_weight_limit)
        
        regular_min_weights = optimizer.optimize_sharpe(
            constraints={'min_position': min_pos_constraint},
            verbose=True
        )
        print("✅ Regular minimum constraint optimization successful")
        
        weights_array = np.array(list(regular_min_weights.values()))
        print(f"\nRegular Minimum Constraint Results:")
        print(f"  Portfolio weights range: {np.min(weights_array):.3%} to {np.max(weights_array):.3%}")
        print(f"  Number of positions >= {min_weight_limit:.1%}: {np.sum(weights_array >= min_weight_limit)}")
        print(f"  Number of violating positions: {np.sum((weights_array > 0.0001) & (weights_array < min_weight_limit))}")
        print(f"  Effectively zero positions: {np.sum(weights_array <= 0.0001)}")
        
        # Show all positions
        print(f"\nDetailed positions:")
        for ticker, weight in regular_min_weights.items():
            status = "ZERO" if weight <= 0.0001 else ("VIOLATION" if 0.0001 < weight < min_weight_limit else "GOOD")
            print(f"    {ticker}: {weight:7.3%} ({status})")
            
    except Exception as e:
        print(f"❌ Regular minimum constraint optimization failed: {e}")
        print(f"This is expected if the constraint is too tight for all {len(returns_data.columns)} assets")
        regular_min_weights = None
    
    # Test 3: Regularized minimum constraint  
    print(f"\n" + "=" * 60)
    print("TEST 3: REGULARIZED MINIMUM POSITION CONSTRAINT")
    print("=" * 60)
    print(f"Constraint: Positions must be either ≤ 0.01% OR >= {min_weight_limit:.1%}")
    print("This avoids small, potentially overfitted positions")
    
    try:
        regularized_constraint = constraint_builder.regularized_min_position_constraint(
            min_weight=min_weight_limit,
            threshold=0.0001  # 0.01% threshold
        )
        
        regularized_weights = optimizer.optimize_sharpe(
            constraints={'regularized_min_position': regularized_constraint},
            verbose=True
        )
        print("✅ Regularized minimum constraint optimization successful")
        
        weights_array = np.array(list(regularized_weights.values()))
        print(f"\nRegularized Minimum Constraint Results:")
        print(f"  Portfolio weights range: {np.min(weights_array):.3%} to {np.max(weights_array):.3%}")
        print(f"  Number of positions >= {min_weight_limit:.1%}: {np.sum(weights_array >= min_weight_limit)}")
        print(f"  Number of zero/tiny positions (≤ 0.01%): {np.sum(weights_array <= 0.0001)}")
        print(f"  Number of 'forbidden' positions (0.01% < weight < {min_weight_limit:.1%}): {np.sum((weights_array > 0.0001) & (weights_array < min_weight_limit))}")
        
        # Show all positions
        print(f"\nDetailed positions:")
        for ticker, weight in regularized_weights.items():
            if weight <= 0.0001:
                status = "ZERO"
            elif weight >= min_weight_limit:
                status = "ACTIVE"
            else:
                status = "VIOLATION"
            print(f"    {ticker}: {weight:7.3%} ({status})")
            
    except Exception as e:
        print(f"❌ Regularized minimum constraint optimization failed: {e}")
        regularized_weights = None
    
    # Test 4: Compare portfolio metrics
    print(f"\n" + "=" * 60)
    print("PORTFOLIO METRICS COMPARISON")
    print("=" * 60)
    
    portfolios = {
        "Unconstrained": unconstrained_weights,
        "Regular Minimum": regular_min_weights,
        "Regularized Minimum": regularized_weights
    }
    
    print(f"{'Portfolio':<20} {'Return':<10} {'Volatility':<10} {'Sharpe':<10} {'Active Pos.':<12} {'Small Pos.':<12}")
    print("-" * 90)
    
    for name, weights in portfolios.items():
        if weights is None:
            print(f"{name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
            continue
            
        weights_array = np.array(list(weights.values()))
        
        # Calculate metrics
        portfolio_return = np.dot(weights_array, optimizer.mean_returns) * 252
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(optimizer.cov_matrix, weights_array))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - 0.03) / portfolio_vol
        
        active_positions = np.sum(weights_array > 0.01)
        small_positions = np.sum((weights_array > 0.0001) & (weights_array <= 0.01))
        
        print(f"{name:<20} {portfolio_return:<10.2%} {portfolio_vol:<10.2%} {sharpe_ratio:<10.2f} {active_positions:<12d} {small_positions:<12d}")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1. UNCONSTRAINED: Allows any weight, may have many small positions")
    print(f"2. REGULAR MINIMUM: Forces all active positions >= {min_weight_limit:.1%}, may be infeasible")
    print(f"3. REGULARIZED MINIMUM: Allows 0% OR >= {min_weight_limit:.1%}, avoids small positions")
    print(f"\nThe regularized constraint is useful for:")
    print(f"  • Avoiding overfitting from many small positions")
    print(f"  • Reducing transaction costs (fewer positions to manage)")
    print(f"  • Maintaining feasibility when regular minimum constraint is too tight")


def test_constraint_with_different_optimizers():
    """Test the regularized constraint with both SciPy and MARSOPT."""
    
    print(f"\n" + "=" * 80)
    print("TESTING REGULARIZED CONSTRAINT WITH DIFFERENT OPTIMIZERS")
    print("=" * 80)
    
    returns_data = generate_sample_returns(n_assets=6, n_periods=252)
    constraint_builder = ConstraintBuilder()
    
    regularized_constraint = constraint_builder.regularized_min_position_constraint(
        min_weight=0.08,  # 8% minimum
        threshold=0.001   # 0.1% threshold
    )
    
    for optimizer_name in ['scipy', 'mars']:
        print(f"\n" + "-" * 40)
        print(f"TESTING WITH {optimizer_name.upper()} OPTIMIZER")
        print("-" * 40)
        
        optimizer = PortfolioOptimizer(
            returns=returns_data,
            risk_free_rate=0.03,
            periods_per_year=252,
            optimizer=optimizer_name
        )
        
        try:
            weights = optimizer.optimize_sharpe(
                constraints={'regularized_min_position': regularized_constraint},
                verbose=True
            )
            
            weights_array = np.array(list(weights.values()))
            print(f"✅ {optimizer_name.upper()} optimization successful")
            print(f"   Active positions (>= 8%): {np.sum(weights_array >= 0.08)}")
            print(f"   Zero positions (<= 0.1%): {np.sum(weights_array <= 0.001)}")
            print(f"   Forbidden positions (0.1% < w < 8%): {np.sum((weights_array > 0.001) & (weights_array < 0.08))}")
            
        except Exception as e:
            print(f"❌ {optimizer_name.upper()} optimization failed: {e}")


if __name__ == "__main__":
    print("Testing Regularized Minimum Position Constraint")
    print("This demonstrates the new constraint that enforces either 0% or minimum weight")
    
    test_constraint_comparison()
    test_constraint_with_different_optimizers()
    
    print(f"\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
