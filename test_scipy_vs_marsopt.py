"""
Direct comparison: SciPy (hard constraints) vs MARSOPT (penalty-based)
Test sector constraints to see which method actually works.
"""

import pandas as pd
import numpy as np
from pyandhold import DataDownloader, PortfolioOptimizer
from pyandhold.optimization import ConstraintBuilder

def test_sector_constraints():
    print('🔬 SCIENTIFIC TEST: SciPy vs MARSOPT for Sector Constraints')
    print('=' * 70)
    
    # Download data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    optimizer = PortfolioOptimizer(returns)
    
    # Define sectors
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA']
    finance_stocks = ['JPM', 'BAC', 'GS']
    energy_stocks = ['XOM', 'CVX']
    healthcare_stocks = ['JNJ', 'PFE', 'UNH']
    
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
    
    sector_mapping = {
        'tech': [ticker_to_index[t] for t in tech_stocks if t in ticker_to_index],
        'finance': [ticker_to_index[t] for t in finance_stocks if t in ticker_to_index],
        'energy': [ticker_to_index[t] for t in energy_stocks if t in ticker_to_index],
        'healthcare': [ticker_to_index[t] for t in healthcare_stocks if t in ticker_to_index]
    }
    
    sector_limits = {
        'tech': (0.30, 0.60),      # Tech: 30-60%
        'finance': (0.10, 0.25),   # Finance: 10-25%  
        'energy': (0.05, 0.15),    # Energy: 5-15%
        'healthcare': (0.10, 0.20) # Healthcare: 10-20%
    }
    
    constraint_builder = ConstraintBuilder()
    sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
    
    def analyze_result(name, weights):
        """Analyze and report sector allocations."""
        tech_alloc = sum(weights.get(t, 0) for t in tech_stocks)
        finance_alloc = sum(weights.get(t, 0) for t in finance_stocks)
        energy_alloc = sum(weights.get(t, 0) for t in energy_stocks)
        healthcare_alloc = sum(weights.get(t, 0) for t in healthcare_stocks)
        
        print(f'\n📊 {name} RESULTS:')
        print(f'Tech:       {tech_alloc:.1%} (target: 30-60%)')
        print(f'Finance:    {finance_alloc:.1%} (target: 10-25%)')  
        print(f'Energy:     {energy_alloc:.1%} (target: 5-15%)')
        print(f'Healthcare: {healthcare_alloc:.1%} (target: 10-20%)')
        
        # Check satisfaction
        satisfied = (0.30 <= tech_alloc <= 0.60 and 
                    0.10 <= finance_alloc <= 0.25 and
                    0.05 <= energy_alloc <= 0.15 and
                    0.10 <= healthcare_alloc <= 0.20)
        
        print(f'Status: {"✅ SATISFIED" if satisfied else "❌ VIOLATED"}')
        
        if not satisfied:
            violations = []
            if not (0.30 <= tech_alloc <= 0.60):
                violations.append(f"Tech: {tech_alloc:.1%}")
            if not (0.10 <= finance_alloc <= 0.25):
                violations.append(f"Finance: {finance_alloc:.1%}")
            if not (0.05 <= energy_alloc <= 0.15):
                violations.append(f"Energy: {energy_alloc:.1%}")
            if not (0.10 <= healthcare_alloc <= 0.20):
                violations.append(f"Healthcare: {healthcare_alloc:.1%}")
            print(f'Violations: {", ".join(violations)}')
        
        return satisfied
    
    # TEST 1: SciPy with hard constraints
    print('\n🔧 Testing SciPy (Hard Constraints)...')
    try:
        scipy_weights = optimizer._optimize_scipy(
            objective_func=lambda w: -optimizer._calculate_sharpe_ratio(w),  # Minimize negative Sharpe
            constraints={'sector': sector_constraints},
            verbose=True
        )
        scipy_satisfied = analyze_result('SCIPY', scipy_weights)
    except Exception as e:
        print(f'❌ SciPy FAILED: {e}')
        scipy_satisfied = False
        scipy_weights = {}
    
    # TEST 2: MARSOPT with penalty-based constraints  
    print('\n🎯 Testing MARSOPT (Penalty-Based)...')
    try:
        marsopt_weights = optimizer._optimize_marsopt(
            objective_func=lambda w: -optimizer._calculate_sharpe_ratio(w),  # Minimize negative Sharpe
            constraints={'sector': sector_constraints},
            n_trials=2000,
            verbose=True
        )
        marsopt_satisfied = analyze_result('MARSOPT', marsopt_weights)
    except Exception as e:
        print(f'❌ MARSOPT FAILED: {e}')
        marsopt_satisfied = False
        marsopt_weights = {}
    
    # FINAL COMPARISON
    print('\n🏆 FINAL VERDICT:')
    print('=' * 50)
    if scipy_satisfied and not marsopt_satisfied:
        print('🥇 SciPy WINS: Only SciPy satisfied all constraints')
        print('   Hard constraints > Penalty-based constraints')
    elif marsopt_satisfied and not scipy_satisfied:
        print('🥇 MARSOPT WINS: Only MARSOPT satisfied all constraints')
        print('   Penalty-based worked better than expected')
    elif scipy_satisfied and marsopt_satisfied:
        print('🤝 TIE: Both methods satisfied constraints')
        print('   Both approaches work for sector constraints')
    else:
        print('💥 BOTH FAILED: Neither method satisfied constraints')
        print('   Sector constraints may be too restrictive')
    
    print('\n📈 RECOMMENDATION:')
    if scipy_satisfied:
        print('✅ Use SciPy for strict constraint satisfaction')
        print('   SciPy handles hard constraints properly')
    elif marsopt_satisfied:
        print('✅ MARSOPT worked - penalty method successful')
    else:
        print('⚠️  Consider relaxing sector constraint bounds')
        print('   Current constraints may be infeasible')

if __name__ == "__main__":
    test_sector_constraints()
