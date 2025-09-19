#!/usr/bin/env python3
"""
Test script to verify that the constraint validation fix works correctly.
"""

import sys
import os
sys.path.append('.')

from pyandhold import DataDownloader, PortfolioOptimizer  
from pyandhold.optimization import ConstraintBuilder

def test_constraint_validation():
    """Test that constraint validation properly catches infeasible constraints."""
    
    print("TESTING IMPROVED CONSTRAINT VALIDATION")
    print("=" * 50)
    
    # Download test data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    
    try:
        prices, returns = downloader.download_prices_and_returns(
            tickers,
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
    except Exception as e:
        print(f"Failed to download data: {e}")
        return False
    
    optimizer = PortfolioOptimizer(returns)
    constraint_builder = ConstraintBuilder()
    
    # Test 1: Feasible constraint (should work)
    print("\n1. Testing FEASIBLE constraint (4 assets, 5% minimum)...")
    try:
        feasible_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        feasible_returns = returns[feasible_tickers] 
        feasible_optimizer = PortfolioOptimizer(feasible_returns)
        
        min_pos_constraint = constraint_builder.min_position_constraint(0.05)
        
        result = feasible_optimizer.optimize_sharpe(
            constraints={'min_position': min_pos_constraint}
        )
        print("   ✅ SUCCESS: Feasible constraint worked as expected")
        for k, v in result.items():
            print(f"     {k}: {v:.2%}")
        
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")
        return False
    
    # Test 2: Infeasible constraint (should fail with clear error)
    print("\n2. Testing INFEASIBLE constraint (15 assets, 10% minimum)...")
    print("   Expected: 15 × 10% = 150% > 100% = IMPOSSIBLE!")
    
    try:
        # This should fail because 15 × 10% = 150% > 100%
        infeasible_constraint = constraint_builder.min_position_constraint(0.10)
        
        result = optimizer.optimize_sharpe(
            constraints={'min_position': infeasible_constraint}
        )
        
        print("   ❌ PROBLEM: This should have failed but didn't!")
        print("   The old silent behavior is still happening!")
        return False
        
    except ValueError as e:
        print("   ✅ SUCCESS: Properly caught infeasible constraint!")
        print(f"   Error message: {str(e)[:200]}...")
        return True
        
    except Exception as e:
        print(f"   ❌ WRONG ERROR TYPE: {type(e).__name__}: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_constraint_validation()
    if success:
        print("\n🎉 CONSTRAINT VALIDATION FIX WORKING!")
    else:
        print("\n💥 CONSTRAINT VALIDATION FIX FAILED!")
    
    sys.exit(0 if success else 1)
