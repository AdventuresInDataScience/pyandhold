#!/usr/bin/env python3
"""
Alternative approach: Test sector constraints with SciPy optimizer which handles hard constraints better.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import pandas as pd
from pyandhold.data.downloader import DataDownloader
from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder

def test_sector_constraints_scipy():
    """Test sector constraints using SciPy optimizer (hard constraints)."""
    
    print("🎯 SECTOR CONSTRAINTS WITH SCIPY (HARD CONSTRAINTS)")
    print("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Create SciPy optimizer instead of MARS
    optimizer = PortfolioOptimizer(returns, optimizer='scipy')
    constraint_builder = ConstraintBuilder()
    
    # Define sectors and their indices
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA']
    finance_stocks = ['JPM', 'BAC', 'GS']
    energy_stocks = ['XOM', 'CVX']
    healthcare_stocks = ['JNJ', 'PFE', 'UNH']
    
    # Map tickers to indices
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
    
    sector_mapping = {
        'tech': [ticker_to_index[t] for t in tech_stocks if t in ticker_to_index],
        'finance': [ticker_to_index[t] for t in finance_stocks if t in ticker_to_index],
        'energy': [ticker_to_index[t] for t in energy_stocks if t in ticker_to_index],
        'healthcare': [ticker_to_index[t] for t in healthcare_stocks if t in ticker_to_index]
    }
    
    # Define sector limits
    sector_limits = {
        'tech': (0.30, 0.60),      # Tech: 30-60%
        'finance': (0.10, 0.25),   # Finance: 10-25%
        'energy': (0.05, 0.15),    # Energy: 5-15%
        'healthcare': (0.10, 0.20) # Healthcare: 10-20%
    }
    
    print("Using SciPy optimizer with HARD constraints...")
    print("Sector definitions:")
    for sector, stocks in [('tech', tech_stocks), ('finance', finance_stocks), 
                          ('energy', energy_stocks), ('healthcare', healthcare_stocks)]:
        print(f"  {sector}: {stocks}")
        if sector in sector_limits:
            min_limit, max_limit = sector_limits[sector]
            print(f"    Target: {min_limit:.0%}-{max_limit:.0%}")
    
    try:
        # Use SciPy with sector constraints
        sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
        
        print(f"\nGenerated {len(sector_constraints)} sector constraint functions")
        print("Optimizing with hard constraints...")
        
        sector_weights = optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            verbose=True
        )
        
        # Convert to dictionary if needed
        if isinstance(sector_weights, list):
            sector_weights = {tickers[i]: sector_weights[i] for i in range(len(tickers))}
        
        # Calculate sector allocations
        sector_allocs = {
            'tech': sum(sector_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(sector_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(sector_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(sector_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        print(f"\n✅ SCIPY OPTIMIZATION SUCCESS")
        print(f"Final sector allocations:")
        
        all_satisfied = True
        for sector, alloc in sector_allocs.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                sector_satisfied = min_limit <= alloc <= max_limit
                if not sector_satisfied:
                    all_satisfied = False
                    
                sector_status = "✅" if sector_satisfied else "❌"
                deviation = ""
                if not sector_satisfied:
                    if alloc < min_limit:
                        deviation = f" (short by {min_limit - alloc:.2%})"
                    else:
                        deviation = f" (over by {alloc - max_limit:.2%})"
                        
                print(f"  {sector_status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%}){deviation}")
            else:
                print(f"  ℹ️ {sector}: {alloc:.2%} (no limits)")
        
        overall_status = "✅ ALL SATISFIED" if all_satisfied else "❌ SOME VIOLATIONS"
        print(f"\n{overall_status}")
        
        print(f"\nTop holdings:")
        for ticker, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.01:
                print(f"  {ticker}: {weight:.2%}")
        
        return all_satisfied
        
    except Exception as e:
        print(f"❌ SCIPY SECTOR CONSTRAINT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sector_constraints_mars():
    """Test sector constraints using MARS optimizer (penalty-based)."""
    
    print("\n" + "=" * 60)
    print("🎯 SECTOR CONSTRAINTS WITH MARS (PENALTY-BASED)")
    print("=" * 60)
    
    # Download data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Create MARS optimizer 
    optimizer = PortfolioOptimizer(returns, optimizer='mars')
    constraint_builder = ConstraintBuilder()
    
    # Define sectors and their indices
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA']
    finance_stocks = ['JPM', 'BAC', 'GS']
    energy_stocks = ['XOM', 'CVX']
    healthcare_stocks = ['JNJ', 'PFE', 'UNH']
    
    # Map tickers to indices
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
    
    sector_mapping = {
        'tech': [ticker_to_index[t] for t in tech_stocks if t in ticker_to_index],
        'finance': [ticker_to_index[t] for t in finance_stocks if t in ticker_to_index],
        'energy': [ticker_to_index[t] for t in energy_stocks if t in ticker_to_index],
        'healthcare': [ticker_to_index[t] for t in healthcare_stocks if t in ticker_to_index]
    }
    
    # Define sector limits
    sector_limits = {
        'tech': (0.30, 0.60),      # Tech: 30-60%
        'finance': (0.10, 0.25),   # Finance: 10-25%
        'energy': (0.05, 0.15),    # Energy: 5-15%
        'healthcare': (0.10, 0.20) # Healthcare: 10-20%
    }
    
    print("Using MARS optimizer with extreme penalties...")
    
    try:
        sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
        
        sector_weights = optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            n_trials=15000,  # Maximum trials
            initial_noise=0.8,  # High diversity
            verbose=False  # Less spam
        )
        
        # Convert to dictionary if needed
        if isinstance(sector_weights, list):
            sector_weights = {tickers[i]: sector_weights[i] for i in range(len(tickers))}
        
        # Calculate sector allocations
        sector_allocs = {
            'tech': sum(sector_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(sector_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(sector_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(sector_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        print(f"\n✅ MARS OPTIMIZATION COMPLETE")
        print(f"Final sector allocations:")
        
        all_satisfied = True
        for sector, alloc in sector_allocs.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                sector_satisfied = min_limit <= alloc <= max_limit
                if not sector_satisfied:
                    all_satisfied = False
                    
                sector_status = "✅" if sector_satisfied else "❌"
                deviation = ""
                if not sector_satisfied:
                    if alloc < min_limit:
                        deviation = f" (short by {min_limit - alloc:.2%})"
                    else:
                        deviation = f" (over by {alloc - max_limit:.2%})"
                        
                print(f"  {sector_status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%}){deviation}")
            else:
                print(f"  ℹ️ {sector}: {alloc:.2%} (no limits)")
        
        overall_status = "✅ ALL SATISFIED" if all_satisfied else "❌ SOME VIOLATIONS"
        print(f"\n{overall_status}")
        
        return all_satisfied
        
    except Exception as e:
        print(f"❌ MARS SECTOR CONSTRAINT TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPARING SECTOR CONSTRAINT ENFORCEMENT")
    print("=" * 60)
    
    scipy_success = test_sector_constraints_scipy()
    mars_success = test_sector_constraints_mars()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SciPy (Hard Constraints): {'✅ SUCCESS' if scipy_success else '❌ FAILED'}")
    print(f"MARS (Penalty-Based): {'✅ SUCCESS' if mars_success else '❌ FAILED'}")
    
    if scipy_success and not mars_success:
        print("\n💡 RECOMMENDATION: Use SciPy optimizer for strict sector constraints!")
    elif mars_success:
        print("\n🎉 MARS penalty method is now working!")
    else:
        print("\n🤔 Both methods need further tuning...")
