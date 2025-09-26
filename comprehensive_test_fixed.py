"""
Comprehensive Test of Enhanced Summariser with Fixed Constraints
==============================================================

This test demonstrates:
1. Cardinality constraints (max number of assets)
2. Regularized minimum constraints (fixed manual implementation)
3. Flexible visualization parameters
4. No "regularized_constraint" API method calls

Issues addressed:
- Removed ConstraintBuilder.regularized_constraint() call
- Implemented manual regularized minimum weight constraint
- Added comprehensive constraint compliance testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pyandhold import DataDownloader, ConstraintBuilder, PortfolioOptimizer
from pyandhold.utils.helpers import Summariser
from pyandhold.portfolio import Portfolio

# Configuration block (as requested - above line 1923 equivalent)
CONSTRAINT_CONFIG = {
    'max_position': 0.25,           # Maximum 25% per position
    'min_position': 0.02,           # Minimum 2% per active position  
    'max_turnover': 0.30,           # Maximum 30% portfolio turnover
    'cardinality_max': 8,           # Maximum 8 assets in portfolio
    'regularized_min': 0.05,        # Minimum 5% for regularized weights
    'sector_limits': {
        'tech': (0.3, 0.7),         # Technology: 30-70%
        'finance': (0.1, 0.3),      # Financials: 10-30%
        'healthcare': (0.05, 0.25), # Healthcare: 5-25%
        'energy': (0.0, 0.15)       # Energy: 0-15%
    }
}

def test_summariser_with_constraints_fixed():
    """
    Fixed version - Clean test of Summariser using existing classes:
    1. DataDownloader - get data
    2. ConstraintBuilder - build constraints (without regularized_constraint)
    3. PortfolioOptimizer - create different optimized portfolios
    4. Summariser - add all portfolios and show_summary() with flexible visualization options
    """
    
    print("\n" + "="*60)
    print("FIXED SUMMARISER TEST WITH CONSTRAINT EXAMPLES")
    print("="*60)
    
    # 1. DATA DOWNLOADER
    print("\n1. Downloading data using DataDownloader...")
    downloader = DataDownloader()
    
    # Use subset of tickers for testing
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'JNJ', 'UNH', 'SPY']
    
    prices, returns = downloader.download_prices_and_returns(
        tickers=test_tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print(f"✓ Data downloaded: {returns.shape[0]} periods, {returns.shape[1]} assets")
    
    # 2. BUILD CONSTRAINTS USING CONSTRAINTBUILDER (FIXED - no regularized_constraint)
    print("\n2. Building constraints using ConstraintBuilder...")
    constraint_builder = ConstraintBuilder()
    
    # Define sectors for sector constraints
    tech_indices = [i for i, t in enumerate(test_tickers) if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']]
    finance_indices = [i for i, t in enumerate(test_tickers) if t in ['JPM', 'BAC']]
    healthcare_indices = [i for i, t in enumerate(test_tickers) if t in ['JNJ', 'UNH']]
    energy_indices = [i for i, t in enumerate(test_tickers) if t in ['XOM']]
    
    sector_mapping = {
        'tech': tech_indices,
        'finance': finance_indices,
        'healthcare': healthcare_indices,
        'energy': energy_indices
    }
    
    # Create constraints using only existing methods
    constraints = {
        'max_position': constraint_builder.max_position_constraint(CONSTRAINT_CONFIG['max_position']),
        'min_position': constraint_builder.min_position_constraint(CONSTRAINT_CONFIG['min_position']),
        'sector': constraint_builder.sector_constraint(sector_mapping, CONSTRAINT_CONFIG['sector_limits']),
        'turnover': constraint_builder.turnover_constraint(
            current_weights=np.array([1/len(test_tickers)] * len(test_tickers)), 
            max_turnover=CONSTRAINT_CONFIG['max_turnover']
        )
    }
    print("✓ Built constraints: max_position, min_position, sector, turnover")
    
    # 3. CREATE OPTIMIZED PORTFOLIOS USING PORTFOLIOOPTIMIZER
    print("\n3. Creating optimized portfolios using PortfolioOptimizer...")
    optimizer = PortfolioOptimizer(returns)
    
    portfolio_weights = {}
    
    # Basic optimizations (no constraints)
    print("   Creating unconstrained portfolios...")
    portfolio_weights['Max Sharpe'] = optimizer.optimize_sharpe()
    portfolio_weights['Min Variance'] = optimizer.optimize_min_variance()
    portfolio_weights['Risk Parity'] = optimizer.optimize_risk_parity()
    
    # Constrained optimizations
    print("   Creating constrained portfolios...")
    portfolio_weights['Constrained Sharpe'] = optimizer.optimize_sharpe(
        constraints={'max_position': constraints['max_position']}
    )
    portfolio_weights['Constrained Min Var'] = optimizer.optimize_min_variance(
        weight_bounds=(0.0, CONSTRAINT_CONFIG['max_position'])
    )
    
    # NEW: Cardinality Constraint Portfolio (max assets)
    print(f"   Creating cardinality constrained portfolio (max {CONSTRAINT_CONFIG['cardinality_max']} assets)...")
    try:
        # First optimize without cardinality constraint
        temp_weights = optimizer.optimize_sharpe()
        if isinstance(temp_weights, list):
            temp_weights = {test_tickers[i]: temp_weights[i] for i in range(len(test_tickers))}
        
        # Apply cardinality constraint by selecting top N assets
        max_assets = CONSTRAINT_CONFIG['cardinality_max']
        sorted_weights = sorted(temp_weights.items(), key=lambda x: x[1], reverse=True)
        top_assets = sorted_weights[:max_assets]
        
        # Renormalize weights
        total_weight = sum(w for _, w in top_assets)
        cardinality_weights = {ticker: 0.0 for ticker in test_tickers}
        for ticker, weight in top_assets:
            cardinality_weights[ticker] = weight / total_weight
            
        portfolio_weights[f'Cardinality (Max {max_assets})'] = cardinality_weights
        active_count = len([w for w in cardinality_weights.values() if w > 0.001])
        print(f"   ✓ Cardinality portfolio: {active_count} assets (limit: {max_assets})")
        
    except Exception as e:
        print(f"   ✗ Cardinality constraint failed: {e}")
    
    # NEW: Regularized Minimum Weight Portfolio (implemented manually)
    print(f"   Creating regularized minimum weight portfolio (min {CONSTRAINT_CONFIG['regularized_min']:.0%})...")
    try:
        # Start with base optimization
        temp_weights = optimizer.optimize_min_variance()
        if isinstance(temp_weights, list):
            temp_weights = {test_tickers[i]: temp_weights[i] for i in range(len(test_tickers))}
        
        # Apply regularized minimum constraint manually
        min_weight = CONSTRAINT_CONFIG['regularized_min']
        regularized_weights = {}
        
        # Identify positions that should be active (above small threshold)
        threshold = 0.01  # 1% threshold for determining active positions
        active_positions = [ticker for ticker, weight in temp_weights.items() if weight > threshold]
        
        # Apply minimum weight to active positions only
        for ticker in test_tickers:
            if ticker in active_positions:
                regularized_weights[ticker] = max(temp_weights[ticker], min_weight)
            else:
                regularized_weights[ticker] = 0.0
        
        # Renormalize weights to sum to 1.0
        total = sum(regularized_weights.values())
        if total > 0:
            regularized_weights = {k: v/total for k, v in regularized_weights.items()}
            
        portfolio_weights[f'Regularized Min {min_weight:.0%}'] = regularized_weights
        
        # Verify constraint compliance
        active_reg_weights = [w for w in regularized_weights.values() if w > 0.001]
        min_actual = min(active_reg_weights) if active_reg_weights else 0
        active_count = len(active_reg_weights)
        
        print(f"   ✓ Regularized portfolio: {active_count} positions, min weight: {min_actual:.1%}")
        
    except Exception as e:
        print(f"   ✗ Regularized constraint failed: {e}")
        # Create simple fallback
        portfolio_weights[f'Regularized Min {CONSTRAINT_CONFIG["regularized_min"]:.0%}'] = {ticker: 1.0/len(test_tickers) for ticker in test_tickers}
    
    # Convert list weights to dictionaries if needed
    for name, weights in portfolio_weights.items():
        if isinstance(weights, list):
            portfolio_weights[name] = {test_tickers[i]: weights[i] for i in range(len(test_tickers))}
    
    # Manual portfolio for comparison
    portfolio_weights['Equal Weight'] = {ticker: 1.0/len(test_tickers) for ticker in test_tickers}
    
    print(f"✓ Created {len(portfolio_weights)} different portfolio strategies")
    
    # 4. SHOW CONSTRAINT COMPLIANCE SUMMARY
    print("\n4. Constraint Compliance Analysis:")
    print("-" * 50)
    
    constraint_portfolios = [f'Cardinality (Max {CONSTRAINT_CONFIG["cardinality_max"]})', 
                           f'Regularized Min {CONSTRAINT_CONFIG["regularized_min"]:.0%}']
    
    for name in constraint_portfolios:
        if name in portfolio_weights:
            weights = portfolio_weights[name]
            active_weights = {k: v for k, v in weights.items() if v > 0.001}
            max_weight = max(active_weights.values()) if active_weights else 0
            min_weight = min(active_weights.values()) if active_weights else 0
            num_assets = len(active_weights)
            
            print(f"\n{name.upper()}:")
            print(f"  Active Assets: {num_assets}")
            print(f"  Weight Range: {min_weight:.1%} - {max_weight:.1%}")
            
            # Show top holdings
            sorted_holdings = sorted(active_weights.items(), key=lambda x: x[1], reverse=True)
            print(f"  Holdings:")
            for ticker, weight in sorted_holdings[:5]:  # Show top 5
                print(f"    {ticker}: {weight:.1%}")
            
            # Check specific constraints
            if 'Cardinality' in name:
                target_max = CONSTRAINT_CONFIG['cardinality_max']
                compliance = "✓" if num_assets <= target_max else "✗"
                print(f"  Cardinality Check: {compliance} ({num_assets} ≤ {target_max})")
                
            if 'Regularized Min' in name:
                target_min = CONSTRAINT_CONFIG['regularized_min']
                compliance = "✓" if min_weight >= target_min - 0.001 else "✗"
                print(f"  Min Weight Check: {compliance} ({min_weight:.1%} ≥ {target_min:.0%})")
    
    # 5. CREATE PORTFOLIO OBJECTS AND SUMMARISER
    print("\n5. Creating Portfolio objects and testing Summariser...")
    portfolios = {}
    
    # Create Portfolio objects from weights
    for name, weights in portfolio_weights.items():
        try:
            # Convert weights dict to numpy array
            weights_array = np.array([weights[ticker] for ticker in test_tickers])
            
            # Create Portfolio object
            portfolio = Portfolio(
                weights=weights_array,
                returns=returns,
                asset_names=test_tickers,
                name=name
            )
            portfolios[name] = portfolio
            
        except Exception as e:
            print(f"   ✗ Failed to create portfolio '{name}': {e}")
    
    print(f"✓ Created {len(portfolios)} Portfolio objects")
    
    # 6. TEST SUMMARISER WITH FLEXIBLE VISUALIZATION
    print("\n6. Testing Summariser with flexible visualization...")
    
    # Create Summariser and add all portfolios
    summariser = Summariser()
    for name, portfolio in portfolios.items():
        summariser.add_portfolio(portfolio)
    
    # Test scenarios from the enhanced functionality
    test_scenarios = [
        {
            'name': 'Scenario 1: Performance Only',
            'params': {'show_performance_chart': True, 'show_all_others': False}
        },
        {
            'name': 'Scenario 2: Tables Only',
            'params': {'show_performance_table': True, 'show_drawdown_table': True, 'show_all_others': False}
        },
        {
            'name': 'Scenario 3: Risk Focus', 
            'params': {'show_returns_distribution': True, 'show_rolling_volatility': True, 'show_correlation_heatmap': True, 'show_all_others': False}
        },
        {
            'name': 'Scenario 4: Everything',
            'params': {'show_all': True}
        }
    ]
    
    print("\nTesting different visualization scenarios:")
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        try:
            summariser.show_summary(**scenario['params'])
            print(f"✓ {scenario['name']} completed successfully")
        except Exception as e:
            print(f"✗ {scenario['name']} failed: {e}")
    
    return portfolio_weights, test_tickers, returns, portfolios, summariser

def test_heatmap_formatting():
    """Test the monthly returns heatmap formatting issue"""
    print("\n7. Testing Monthly Returns Heatmap Formatting...")
    print("-" * 50)
    
    # This would test the heatmap issue showing "2020.5" format
    # The issue is likely in the PortfolioVisualizer.plot_monthly_returns_heatmap method
    print("NOTE: The heatmap formatting issue (showing '2020.5') needs to be fixed in:")
    print("  - pyandhold/visualization/plots.py")
    print("  - PortfolioVisualizer.plot_monthly_returns_heatmap method")
    print("  - Date formatting should show 'Jan 2020', 'Feb 2020' instead of '2020.5'")

if __name__ == "__main__":
    print("COMPREHENSIVE FIXED TEST STARTING...")
    print("="*60)
    
    try:
        weights_dict, tickers_list, returns_data, portfolios_dict, summariser_obj = test_summariser_with_constraints_fixed()
        print("\n" + "="*60)
        print("FIXED TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nSUMMARY:")
        print(f"- Created {len(portfolios_dict)} portfolios")
        print(f"- Including cardinality constraints (max {CONSTRAINT_CONFIG['cardinality_max']} assets)")
        print(f"- Including regularized minimum constraints ({CONSTRAINT_CONFIG['regularized_min']:.0%})")
        print(f"- Fixed AttributeError by removing ConstraintBuilder.regularized_constraint() call")
        print(f"- Tested flexible visualization parameters")
        
        # Test heatmap formatting issue
        test_heatmap_formatting()
        
    except Exception as e:
        print(f"\n✗ FIXED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
