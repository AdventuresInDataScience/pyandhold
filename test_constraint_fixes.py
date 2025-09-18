#!/usr/bin/env python3
"""
Test script to validate the constraint fixes for sector constraints.
This tests the improvements made to handle sector constraint violations.
"""

import numpy as np
import pandas as pd
import importlib
import sys

# Force reload of ALL related modules to ensure we get the latest changes
modules_to_reload = [
    'pyandhold.optimization.optimizers',
    'pyandhold.optimization.constraints', 
    'pyandhold.data.downloader'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"Reloading {module_name}...")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"{module_name} not loaded yet")

from pyandhold.data.downloader import DataDownloader
from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.constraints import ConstraintBuilder

# Verify the fixes are in place
print("🔧 Verifying optimizer routing logic...")
test_returns = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
test_opt_scipy = PortfolioOptimizer(test_returns, optimizer='scipy')
test_opt_mars = PortfolioOptimizer(test_returns, optimizer='mars')
print(f"   SciPy optimizer.optimizer = '{test_opt_scipy.optimizer}'")
print(f"   MARS optimizer.optimizer = '{test_opt_mars.optimizer}'")
print(f"   _should_use_scipy_for_constraints method exists: {hasattr(test_opt_scipy, '_should_use_scipy_for_constraints')}")
print(f"   _normalize_with_bounds method exists: {hasattr(test_opt_scipy, '_normalize_with_bounds')}")
print("")


def test_constraint_fixes():
    """Test both MARS and SciPy optimizers with sector constraints."""
    
    print("🎯 TESTING CONSTRAINT FIXES")
    print("=" * 60)
    
    # Download test data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    
    try:
        prices, returns = downloader.download_prices_and_returns(
            tickers,
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        print(f"✅ Downloaded data for {len(tickers)} assets")
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return False
    
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
    
    # Define realistic sector limits
    sector_limits = {
        'tech': (0.30, 0.60),      # Tech: 30-60%
        'finance': (0.10, 0.25),   # Finance: 10-25%
        'energy': (0.05, 0.15),    # Energy: 5-15%
        'healthcare': (0.10, 0.20) # Healthcare: 10-20%
    }
    
    print(f"\n📋 Sector constraints:")
    for sector, (min_limit, max_limit) in sector_limits.items():
        stocks = [tech_stocks, finance_stocks, energy_stocks, healthcare_stocks]
        sector_stocks = stocks[list(sector_limits.keys()).index(sector)]
        print(f"   {sector}: {min_limit:.0%}-{max_limit:.0%} ({len(sector_stocks)} stocks)")
    
    constraint_builder = ConstraintBuilder()
    sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
    
    results = {}
    
    # Test 1: SciPy optimizer (should work well)
    print(f"\n🔧 Testing SciPy Optimizer...")
    try:
        scipy_optimizer = PortfolioOptimizer(returns, optimizer='scipy')
        
        scipy_weights = scipy_optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            verbose=True
        )
        
        # Calculate sector allocations
        sector_allocs_scipy = {
            'tech': sum(scipy_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(scipy_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(scipy_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(scipy_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        # Check constraint satisfaction
        scipy_violations = 0
        print(f"\n   📊 SciPy Sector Allocations:")
        for sector, alloc in sector_allocs_scipy.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                satisfied = min_limit <= alloc <= max_limit
                if not satisfied:
                    scipy_violations += 1
                
                status = "✅" if satisfied else "❌"
                print(f"      {status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%})")
        
        results['scipy'] = {
            'success': scipy_violations == 0,
            'violations': scipy_violations,
            'weights': scipy_weights,
            'allocations': sector_allocs_scipy
        }
        
        overall_status = "✅ SUCCESS" if scipy_violations == 0 else f"❌ {scipy_violations} VIOLATIONS"
        print(f"   {overall_status}")
        
    except Exception as e:
        print(f"   ❌ SciPy optimizer failed: {e}")
        results['scipy'] = {'success': False, 'error': str(e)}
    
    # Test 2: MARS optimizer with fixes
    print(f"\n🎯 Testing MARS Optimizer (with constraint fixes)...")
    try:
        mars_optimizer = PortfolioOptimizer(returns, optimizer='mars')
        
        mars_weights = mars_optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            n_trials=3000,
            initial_noise=0.6,
            verbose=True
        )
        
        # Calculate sector allocations
        sector_allocs_mars = {
            'tech': sum(mars_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(mars_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(mars_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(mars_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        # Check constraint satisfaction
        mars_violations = 0
        print(f"\n   📊 MARS Sector Allocations:")
        for sector, alloc in sector_allocs_mars.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                satisfied = min_limit <= alloc <= max_limit
                if not satisfied:
                    mars_violations += 1
                
                status = "✅" if satisfied else "❌"
                print(f"      {status} {sector}: {alloc:.2%} (target: {min_limit:.0%}-{max_limit:.0%})")
        
        results['mars'] = {
            'success': mars_violations == 0,
            'violations': mars_violations,
            'weights': mars_weights,
            'allocations': sector_allocs_mars
        }
        
        overall_status = "✅ SUCCESS" if mars_violations == 0 else f"❌ {mars_violations} VIOLATIONS"
        print(f"   {overall_status}")
        
    except Exception as e:
        print(f"   ❌ MARS optimizer failed: {e}")
        results['mars'] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📋 CONSTRAINT FIX TEST SUMMARY")
    print(f"=" * 60)
    
    for optimizer_name, result in results.items():
        if 'success' in result:
            status = "✅ SUCCESS" if result['success'] else f"❌ FAILED"
            print(f"{optimizer_name.upper()}: {status}")
            if not result['success'] and 'violations' in result:
                print(f"   Constraint violations: {result['violations']}")
            elif 'error' in result:
                print(f"   Error: {result['error']}")
        else:
            print(f"{optimizer_name.upper()}: ❌ NO RESULT")
    
    # Test success
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    if success_count == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! ({success_count}/{total_tests})")
        print(f"   Both optimizers now properly enforce sector constraints!")
        return True
    elif success_count > 0:
        print(f"\n⚠️  PARTIAL SUCCESS ({success_count}/{total_tests})")
        print(f"   Some improvements working, but more fixes needed")
        return False
    else:
        print(f"\n💥 ALL TESTS FAILED ({success_count}/{total_tests})")
        print(f"   Constraint fixes need more work")
        return False


def test_simple_constraints():
    """Test simpler constraints to validate basic functionality."""
    
    print(f"\n" + "=" * 60)
    print(f"🔬 TESTING SIMPLE CONSTRAINTS")
    print(f"=" * 60)
    
    # Simple test with fewer assets
    downloader = DataDownloader()
    simple_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    
    try:
        prices, returns = downloader.download_prices_and_returns(
            simple_tickers,
            start_date='2022-01-01',
            end_date='2023-12-31'
        )
    except Exception as e:
        print(f"❌ Failed to download simple test data: {e}")
        return False
    
    # Test max position constraint (should be easy)
    print(f"\n📏 Testing Max Position Constraint (25% limit)...")
    
    try:
        optimizer = PortfolioOptimizer(returns, optimizer='mars')
        weights = optimizer.optimize_sharpe(
            weight_bounds=(0.0, 0.25),  # Max 25% per position
            verbose=True
        )
        
        max_weight = max(weights.values())
        success = max_weight <= 0.25 + 1e-6  # Small tolerance
        
        status = "✅" if success else "❌"
        print(f"{status} Max position constraint: max weight = {max_weight:.2%}")
        
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {ticker}: {weight:.2%}")
            
        return success
        
    except Exception as e:
        print(f"❌ Simple constraint test failed: {e}")
        return False


def test_weight_bounds_diagnostic():
    """Diagnostic test to understand why weight bounds are violated."""
    
    print(f"\n" + "=" * 60)
    print(f"🔍 WEIGHT BOUNDS DIAGNOSTIC")
    print(f"=" * 60)
    
    # Test simple 4-asset case with 25% weight bounds
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    
    try:
        prices, returns = downloader.download_prices_and_returns(
            tickers,
            start_date='2022-01-01', 
            end_date='2023-12-31'
        )
    except Exception as e:
        print(f"❌ Failed to download diagnostic data: {e}")
        return False
    
    print(f"Testing 4 assets: {tickers}")
    print(f"Weight bounds: 0% - 25%")
    print(f"Expected: All weights ≤ 25%")
    
    # Test MARS with detailed output
    print(f"\n🎯 MARS Optimizer Diagnostic:")
    print(f"  Setting optimizer='mars' explicitly...")
    mars_optimizer = PortfolioOptimizer(returns, optimizer='mars')
    
    mars_weights = mars_optimizer.optimize_sharpe(
        weight_bounds=(0.0, 0.25),
        n_trials=1000,
        verbose=True
    )
    
    print(f"\n📊 MARS Results:")
    total_weight = 0
    max_weight = 0
    violations = 0
    
    for ticker, weight in sorted(mars_weights.items(), key=lambda x: x[1], reverse=True):
        violation = weight > 0.25
        if violation:
            violations += 1
        status = "❌" if violation else "✅"
        print(f"  {status} {ticker}: {weight:.4f} ({weight:.2%})")
        total_weight += weight
        max_weight = max(max_weight, weight)
    
    print(f"\n📋 MARS Validation:")
    print(f"  Total weight: {total_weight:.4f} ({total_weight:.2%})")
    print(f"  Max weight: {max_weight:.4f} ({max_weight:.2%})")
    print(f"  Violations: {violations}")
    print(f"  Bounds violation: {'YES' if max_weight > 0.25 else 'NO'}")
    
    # Test SciPy for comparison
    print(f"\n🔧 SciPy Optimizer Diagnostic:")
    print(f"  Setting optimizer='scipy' explicitly...")
    scipy_optimizer = PortfolioOptimizer(returns, optimizer='scipy')
    
    scipy_weights = scipy_optimizer.optimize_sharpe(
        weight_bounds=(0.0, 0.25),
        verbose=True
    )
    
    print(f"\n📊 SciPy Results:")
    total_weight = 0
    max_weight = 0
    violations = 0
    
    for ticker, weight in sorted(scipy_weights.items(), key=lambda x: x[1], reverse=True):
        violation = weight > 0.25
        if violation:
            violations += 1
        status = "❌" if violation else "✅"
        print(f"  {status} {ticker}: {weight:.4f} ({weight:.2%})")
        total_weight += weight
        max_weight = max(max_weight, weight)
    
    print(f"\n📋 SciPy Validation:")
    print(f"  Total weight: {total_weight:.4f} ({total_weight:.2%})")
    print(f"  Max weight: {max_weight:.4f} ({max_weight:.2%})")
    print(f"  Violations: {violations}")
    print(f"  Bounds violation: {'YES' if max_weight > 0.25 else 'NO'}")
    
    # Summary
    mars_success = max([w for w in mars_weights.values()]) <= 0.25 + 1e-6
    scipy_success = max([w for w in scipy_weights.values()]) <= 0.25 + 1e-6
    
    print(f"\n💡 Diagnostic Summary:")
    print(f"  MARS bounds respected: {'✅ YES' if mars_success else '❌ NO'}")
    print(f"  SciPy bounds respected: {'✅ YES' if scipy_success else '❌ NO'}")
    
    if not mars_success and not scipy_success:
        print(f"  🚨 Both optimizers violating bounds - check constraint setup")
    elif not mars_success:
        print(f"  🚨 MARS issue - check weight reconstruction in _optimize_mars")
    elif not scipy_success:
        print(f"  🚨 SciPy issue - check bounds constraints in _optimize_scipy")
    else:
        print(f"  🎉 Both optimizers working correctly!")
    
    print(f"\n🔧 OPTIMIZER ROUTING TEST:")
    print(f"  Expected behavior after fix:")
    print(f"  - MARS test should show: '🎯 Using MARSOPT'")
    print(f"  - SciPy test should show: '🔧 Using SciPy'")
    print(f"  - If both show MARSOPT, the fix didn't work")
    
    return mars_success and scipy_success


def test_direct_optimizer_routing():
    """Direct test of optimizer routing logic."""
    
    print(f"\n" + "=" * 60)
    print(f"🔧 DIRECT OPTIMIZER ROUTING TEST")
    print(f"=" * 60)
    
    # Create test data
    test_returns = pd.DataFrame(np.random.randn(100, 4), columns=['AAPL', 'MSFT', 'GOOGL', 'JPM'])
    
    print("Testing _should_use_scipy_for_constraints method directly:")
    
    # Test SciPy explicitly set
    scipy_opt = PortfolioOptimizer(test_returns, optimizer='scipy')
    should_use_scipy_explicit = scipy_opt._should_use_scipy_for_constraints(None, verbose=True)
    print(f"  optimizer='scipy', no constraints → should_use_scipy = {should_use_scipy_explicit}")
    
    # Test MARS explicitly set  
    mars_opt = PortfolioOptimizer(test_returns, optimizer='mars')
    should_use_scipy_mars = mars_opt._should_use_scipy_for_constraints(None, verbose=True)
    print(f"  optimizer='mars', no constraints → should_use_scipy = {should_use_scipy_mars}")
    
    # Test with weight bounds only (should respect explicit choice)
    print(f"\nTesting with weight bounds (0, 0.25):")
    
    # This should show different messages for scipy vs mars
    scipy_weights = scipy_opt.optimize_sharpe(
        weight_bounds=(0.0, 0.25),
        verbose=True
    )
    
    print(f"\nNow testing MARS explicitly:")
    mars_weights = mars_opt.optimize_sharpe(
        weight_bounds=(0.0, 0.25), 
        verbose=True
    )
    
    # Check if weights respect bounds
    scipy_max = max(scipy_weights.values())
    mars_max = max(mars_weights.values())
    
    print(f"\n📊 Direct routing test results:")
    print(f"  SciPy max weight: {scipy_max:.4f} ({scipy_max:.2%})")
    print(f"  MARS max weight: {mars_max:.4f} ({mars_max:.2%})")
    print(f"  SciPy bounds OK: {'✅' if scipy_max <= 0.25 + 1e-6 else '❌'}")
    print(f"  MARS bounds OK: {'✅' if mars_max <= 0.25 + 1e-6 else '❌'}")
    
    return scipy_max <= 0.25 + 1e-6 and mars_max <= 0.25 + 1e-6


if __name__ == "__main__":
    print("🚀 CONSTRAINT FIX VALIDATION")
    print("Testing improvements to sector constraint enforcement...")
    
    # Run direct routing test first
    direct_success = test_direct_optimizer_routing()
    
    # Run diagnostic test
    diagnostic_success = test_weight_bounds_diagnostic()
    
    # Run main test
    main_success = test_constraint_fixes()
    
    # Run simple test
    simple_success = test_simple_constraints()
    
    print(f"\n" + "=" * 60)
    print(f"🏁 FINAL RESULTS")
    print(f"=" * 60)
    print(f"Direct routing test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Weight bounds diagnostic: {'✅ PASS' if diagnostic_success else '❌ FAIL'}")
    print(f"Main test (sector constraints): {'✅ PASS' if main_success else '❌ FAIL'}")
    print(f"Simple test (position limits): {'✅ PASS' if simple_success else '❌ FAIL'}")
    
    if direct_success and diagnostic_success and main_success and simple_success:
        print(f"\n🎉 ALL CONSTRAINT FIXES WORKING!")
        print(f"   The portfolio optimizer now properly enforces constraints!")
    elif direct_success:
        print(f"\n🔧 CORE FIXES WORKING!")
        print(f"   Direct tests pass - issue may be in test setup")
    else:
        print(f"\n⚠️  MORE WORK NEEDED")
        print(f"   Core constraint fixes still have issues")
