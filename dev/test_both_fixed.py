#!/usr/bin/env python3
"""Test both optimizers with improved normalization handling."""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from pyandhold.data.downloader import DataDownloader
from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder

def test_both_optimizers_fixed():
    """Test both SciPy and MARSOPT with improved normalization."""
    
    print("🔬 TESTING BOTH OPTIMIZERS - IMPROVED NORMALIZATION")
    print("=" * 70)
    
    # Download data
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
        return False, False
    
    # Define sectors
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
    
    constraint_builder = ConstraintBuilder()
    sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
    
    # Test SciPy
    print("\n" + "🔧 TESTING IMPROVED SCIPY")
    print("-" * 40)
    
    scipy_success = False
    try:
        scipy_optimizer = PortfolioOptimizer(returns, optimizer='scipy')
        
        scipy_weights = scipy_optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            verbose=True
        )
        
        # Calculate sector allocations
        scipy_allocs = {
            'tech': sum(scipy_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(scipy_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(scipy_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(scipy_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        scipy_satisfied = all(
            sector_limits[sector][0] <= alloc <= sector_limits[sector][1]
            for sector, alloc in scipy_allocs.items()
            if sector in sector_limits
        )
        
        print(f"\n📊 SCIPY RESULTS:")
        for sector, alloc in scipy_allocs.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                status = "✅" if min_limit <= alloc <= max_limit else "❌"
                print(f"  {status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%})")
        
        scipy_success = scipy_satisfied
        
    except Exception as e:
        print(f"❌ SCIPY FAILED: {e}")
    
    # Test MARSOPT
    print("\n" + "🔧 TESTING IMPROVED MARSOPT") 
    print("-" * 40)
    
    mars_success = False
    try:
        mars_optimizer = PortfolioOptimizer(returns, optimizer='mars')
        
        mars_weights = mars_optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            n_trials=3000,
            initial_noise=0.5,
            verbose=True
        )
        
        # Calculate sector allocations
        mars_allocs = {
            'tech': sum(mars_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(mars_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(mars_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(mars_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        mars_satisfied = all(
            sector_limits[sector][0] <= alloc <= sector_limits[sector][1]
            for sector, alloc in mars_allocs.items()
            if sector in sector_limits
        )
        
        print(f"\n📊 MARSOPT RESULTS:")
        for sector, alloc in mars_allocs.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                status = "✅" if min_limit <= alloc <= max_limit else "❌"
                print(f"  {status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%})")
        
        mars_success = mars_satisfied
        
    except Exception as e:
        print(f"❌ MARSOPT FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"SciPy (Improved): {'✅ SUCCESS' if scipy_success else '❌ FAILED'}")
    print(f"MARSOPT (Fixed): {'✅ SUCCESS' if mars_success else '❌ FAILED'}")
    
    if scipy_success and mars_success:
        print("\n🎉 BOTH OPTIMIZERS NOW WORKING!")
    elif scipy_success and not mars_success:
        print("\n💡 SciPy fixed, MARSOPT still needs work")
    elif not scipy_success and mars_success:
        print("\n🚀 MARSOPT fixed, SciPy still has issues")
    else:
        print("\n💥 Both still need more work")
    
    return scipy_success, mars_success

if __name__ == "__main__":
    scipy_ok, mars_ok = test_both_optimizers_fixed()
    print(f"\nResults: SciPy={scipy_ok}, MARSOPT={mars_ok}")
