#!/usr/bin/env python3
"""
Quick verification test to confirm constraint validation is working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import pandas as pd
from pyandhold.data.downloader import DataDownloader
from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder

def test_constraint_validation():
    """Test that constraint validation is working correctly."""
    
    print("🔍 CONSTRAINT VALIDATION VERIFICATION")
    print("=" * 50)
    
    # Download data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # 4 assets
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    optimizer = PortfolioOptimizer(returns)
    constraint_builder = ConstraintBuilder()
    
    print("\n✅ TEST 1: FEASIBLE CONSTRAINT (4 assets, 5% minimum)")
    print("Expected: Should work, each asset can get ≥5%")
    
    try:
        min_pos_constraint = constraint_builder.min_position_constraint(0.05)
        weights_list = optimizer.optimize_sharpe(
            constraints={'min_position': min_pos_constraint},
            verbose=True
        )
        
        # Convert to dictionary
        if isinstance(weights_list, list):
            weights = {tickers[i]: weights_list[i] for i in range(len(tickers))}
        else:
            weights = weights_list
            
        print("\n✅ SUCCESS - Feasible constraint worked!")
        print("Final weights:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.2%}")
        print(f"Total: {sum(weights.values()):.1%}")
        
    except Exception as e:
        print(f"❌ UNEXPECTED FAILURE: {e}")
    
    print("\n" + "="*50)
    print("❌ TEST 2: INFEASIBLE CONSTRAINT (4 assets, 30% minimum)")  
    print("Expected: Should fail - 4×30% = 120% > 100%")
    
    try:
        min_pos_constraint = constraint_builder.min_position_constraint(0.30)
        weights_list = optimizer.optimize_sharpe(
            constraints={'min_position': min_pos_constraint},
            verbose=True
        )
        
        print("❌ UNEXPECTED SUCCESS - This should have failed!")
        
    except ValueError as e:
        print("✅ EXPECTED FAILURE - Constraint validation caught infeasible constraint!")
        print("Error details:")
        print(f"  {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR TYPE: {e}")

if __name__ == "__main__":
    test_constraint_validation()
