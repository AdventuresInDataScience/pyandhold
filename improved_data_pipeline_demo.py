#!/usr/bin/env python3
"""
Demonstration of the improved data pipeline that preserves maximum historical data.

Problem Solved:
- OLD PIPELINE: Download → Align ALL → Optimize → Select subset (loses historical data)
- NEW PIPELINE: Download → Optimize → Select subset → Align SELECTED (preserves max data)

This approach ensures that when optimization selects a subset of assets, you retain
the maximum available historical data for those specific assets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyandhold import DataDownloader, PortfolioOptimizer, DataPreprocessor

def demonstrate_data_pipeline_improvement():
    """
    Compare the old vs new data pipeline approaches.
    """
    print("🔍 IMPROVED DATA PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Example universe: Mix of assets with different history lengths
    tickers = [
        'AAPL',  # Long history
        'MSFT',  # Long history  
        'GOOGL', # Medium history (IPO 2004)
        'META',  # Shorter history (IPO 2012)
        'TSLA',  # Shorter history (IPO 2010)
        'NVDA',  # Long history but became relevant later
        'AMZN',  # Long history
        'NFLX',  # Medium history
    ]
    
    start_date = '2000-01-01'  # Long lookback to show the effect
    end_date = '2023-12-31'
    
    downloader = DataDownloader()
    
    print(f"\n📊 Downloading data for {len(tickers)} assets from {start_date} to {end_date}...")
    
    # ========================================
    # OLD APPROACH: Align first, then optimize
    # ========================================
    print(f"\n🔴 OLD APPROACH: Download → Align ALL → Optimize")
    print("-" * 50)
    
    # Download data
    prices_old, returns_old = downloader.download_prices_and_returns(
        tickers, start_date, end_date
    )
    
    print(f"   Raw data shape: {returns_old.shape}")
    print(f"   Data per asset before alignment:")
    for ticker in tickers:
        if ticker in returns_old.columns:
            asset_data = returns_old[ticker].dropna()
            print(f"     {ticker}: {len(asset_data)} periods ({asset_data.index[0]} to {asset_data.index[-1]})")
    
    # OLD: Align all data to common date range
    aligned_returns_old = DataPreprocessor.align_data(returns_old)
    print(f"\n   After aligning ALL assets: {aligned_returns_old.shape}")
    print(f"   Common date range: {aligned_returns_old.index[0]} to {aligned_returns_old.index[-1]}")
    
    # Optimize on the aligned (trimmed) data
    optimizer_old = PortfolioOptimizer(aligned_returns_old)
    weights_old = optimizer_old.optimize_sharpe()
    
    # Show selected assets
    selected_old = {k: v for k, v in weights_old.items() if abs(v) > 0.01}
    print(f"   Selected assets (>1%): {list(selected_old.keys())}")
    print(f"   Final data periods available: {len(aligned_returns_old)}")
    
    # ========================================
    # NEW APPROACH: Optimize first, then align selected
    # ========================================
    print(f"\n🟢 NEW APPROACH: Download → Optimize → Align SELECTED")
    print("-" * 50)
    
    # Download with flexible alignment (preserve individual histories)
    prices_new, returns_new = downloader.download_with_flexible_alignment(
        tickers, start_date, end_date, align_all=False
    )
    
    print(f"   Raw data shape: {returns_new.shape}")
    print(f"   Individual asset histories preserved:")
    for ticker in tickers:
        if ticker in returns_new.columns:
            asset_data = returns_new[ticker].dropna()
            print(f"     {ticker}: {len(asset_data)} periods ({asset_data.index[0]} to {asset_data.index[-1]})")
    
    # Define optimization function
    def optimize_portfolio(returns_data):
        """Optimization function for the workflow."""
        # Handle the case where assets have different date ranges
        # For optimization, we'll use the overlapping period
        common_dates = returns_data.dropna().index
        if len(common_dates) < 100:  # Need reasonable amount of data
            # Fall back to pairwise approach or use available data
            aligned_for_opt = DataPreprocessor.align_data(returns_data)
            optimizer = PortfolioOptimizer(aligned_for_opt)
        else:
            optimizer = PortfolioOptimizer(returns_data.loc[common_dates])
        
        return optimizer.optimize_sharpe()
    
    # NEW: Use the optimize-then-align workflow
    final_prices, final_returns, final_weights = DataPreprocessor.optimize_then_align_workflow(
        prices_new, returns_new, optimize_portfolio, min_history=100
    )
    
    # Show results
    selected_new = {k: v for k, v in final_weights.items() if abs(v) > 0.01}
    print(f"   Selected assets (>1%): {list(selected_new.keys())}")
    print(f"   Final data periods available: {len(final_returns)}")
    
    # ========================================
    # COMPARISON
    # ========================================
    print(f"\n📈 COMPARISON RESULTS")
    print("=" * 60)
    
    data_improvement = len(final_returns) - len(aligned_returns_old)
    improvement_pct = (data_improvement / len(aligned_returns_old)) * 100
    
    print(f"Old approach data periods: {len(aligned_returns_old)}")
    print(f"New approach data periods: {len(final_returns)}")
    print(f"Additional data periods:   {data_improvement:+d} ({improvement_pct:+.1f}%)")
    
    if data_improvement > 0:
        print(f"🎉 SUCCESS: New approach preserves {data_improvement} additional periods of data!")
        
        # Show the additional date range
        old_start = aligned_returns_old.index[0]
        new_start = final_returns.index[0]
        if new_start < old_start:
            print(f"   Extra history: {new_start} to {old_start}")
    else:
        print(f"ℹ️  No improvement in this example (selected assets had similar history lengths)")
    
    # Show weight comparison
    print(f"\n💰 PORTFOLIO WEIGHTS COMPARISON")
    print(f"{'Asset':<8} {'Old Weight':<12} {'New Weight':<12} {'Difference':<12}")
    print("-" * 48)
    
    all_assets = set(list(weights_old.keys()) + list(final_weights.keys()))
    for asset in sorted(all_assets):
        old_w = weights_old.get(asset, 0)
        new_w = final_weights.get(asset, 0)
        diff = new_w - old_w
        
        if abs(old_w) > 0.001 or abs(new_w) > 0.001:  # Only show meaningful weights
            print(f"{asset:<8} {old_w:>10.1%} {new_w:>12.1%} {diff:>+10.1%}")
    
    return final_prices, final_returns, final_weights

def demonstrate_specific_use_case():
    """
    Show a specific example where this makes a big difference.
    """
    print(f"\n🎯 SPECIFIC USE CASE EXAMPLE")
    print("=" * 60)
    print("Scenario: You want to optimize among tech stocks, but TSLA has shorter history")
    print("Problem: Traditional approach trims AAPL/MSFT data to match TSLA's shorter history")
    print("Solution: Optimize first, then if TSLA isn't selected, keep full AAPL/MSFT history")
    
    tech_tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL']
    
    downloader = DataDownloader()
    
    # Use the new flexible approach
    prices, returns = downloader.download_with_flexible_alignment(
        tech_tickers, '2005-01-01', '2023-12-31', align_all=False
    )
    
    print(f"\nIndividual asset histories:")
    for ticker in tech_tickers:
        if ticker in returns.columns:
            asset_data = returns[ticker].dropna()
            first_date = asset_data.index[0]
            print(f"  {ticker}: {len(asset_data)} periods (starts {first_date})")
    
    # Define a simple optimization
    def tech_optimization(returns_data):
        aligned_data = DataPreprocessor.align_data(returns_data)  # Align for optimization
        optimizer = PortfolioOptimizer(aligned_data)
        return optimizer.optimize_sharpe(weight_bounds=(0.1, 0.4))  # Diversified
    
    # Apply the workflow
    final_prices, final_returns, weights = DataPreprocessor.optimize_then_align_workflow(
        prices, returns, tech_optimization, min_history=200
    )
    
    print(f"\nOptimization results:")
    for asset, weight in weights.items():
        if abs(weight) > 0.01:
            print(f"  {asset}: {weight:.1%}")
    
    print(f"\nFinal aligned data: {len(final_returns)} periods")
    print(f"Date range: {final_returns.index[0]} to {final_returns.index[-1]}")
    
    return final_prices, final_returns, weights

if __name__ == "__main__":
    # Run the demonstration
    try:
        demonstrate_data_pipeline_improvement()
        print("\n" + "="*60)
        demonstrate_specific_use_case()
        
        print(f"\n✅ SUMMARY")
        print("="*60)
        print("The new pipeline approach provides these benefits:")
        print("1. 📈 Preserves maximum historical data for selected assets")
        print("2. 🎯 Avoids trimming data based on irrelevant assets")  
        print("3. 🔧 Maintains flexibility with align_all parameter")
        print("4. 📊 Better backtesting with longer histories")
        print("5. 🚀 Easy to integrate with existing optimization workflows")
        
        print(f"\nKey Methods Added:")
        print("- DataDownloader.download_with_flexible_alignment()")
        print("- DataPreprocessor.align_selected_assets()")
        print("- DataPreprocessor.optimize_then_align_workflow()")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
