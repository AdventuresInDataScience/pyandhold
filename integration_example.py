#!/usr/bin/env python3
"""
Simple example of how to integrate the improved data pipeline into your existing workflow.

BEFORE (loses data):
    downloader = DataDownloader()
    prices, returns = downloader.download_prices_and_returns(tickers, start_date, end_date)
    aligned_returns = DataPreprocessor.align_data(returns)  # Trims to common range
    optimizer = PortfolioOptimizer(aligned_returns)
    weights = optimizer.optimize_sharpe()

AFTER (preserves data):
    Use the new optimize_then_align_workflow or download_with_flexible_alignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyandhold import DataDownloader, PortfolioOptimizer, DataPreprocessor

def old_workflow_example():
    """The traditional workflow that loses data."""
    print("🔴 OLD WORKFLOW (loses historical data)")
    print("-" * 50)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']
    
    # Traditional approach
    downloader = DataDownloader()
    prices, returns = downloader.download_prices_and_returns(
        tickers, '2010-01-01', '2023-12-31'
    )
    
    print(f"Raw data shape: {returns.shape}")
    
    # This trims ALL data to the common range (loses valuable history)
    aligned_returns = DataPreprocessor.align_data(returns)
    print(f"After alignment: {aligned_returns.shape}")
    print(f"Lost {returns.shape[0] - aligned_returns.shape[0]} periods due to early alignment")
    
    # Optimize on the trimmed data
    optimizer = PortfolioOptimizer(aligned_returns)
    weights = optimizer.optimize_sharpe()
    
    return weights, aligned_returns

def new_workflow_example():
    """The improved workflow that preserves data."""
    print("🟢 NEW WORKFLOW (preserves maximum data)")
    print("-" * 50)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']
    
    downloader = DataDownloader()
    
    # Method 1: Use the all-in-one workflow
    prices, returns = downloader.download_with_flexible_alignment(
        tickers, '2010-01-01', '2023-12-31', align_all=False
    )
    
    print(f"Raw data shape: {returns.shape}")
    
    def my_optimization(returns_data):
        # For optimization, we need aligned data, but only temporarily
        temp_aligned = DataPreprocessor.align_data(returns_data)
        optimizer = PortfolioOptimizer(temp_aligned)
        return optimizer.optimize_sharpe()
    
    # This preserves max data for the final selected assets
    final_prices, final_returns, weights = DataPreprocessor.optimize_then_align_workflow(
        prices, returns, my_optimization
    )
    
    return weights, final_returns

def manual_workflow_example():
    """Manual step-by-step for full control."""
    print("🔧 MANUAL WORKFLOW (step-by-step control)")
    print("-" * 50)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']
    
    downloader = DataDownloader()
    
    # Step 1: Download without aligning
    prices, returns = downloader.download_with_flexible_alignment(
        tickers, '2010-01-01', '2023-12-31', align_all=False
    )
    
    print(f"Downloaded data: {returns.shape}")
    
    # Step 2: Optimize on whatever overlapping data exists
    # (You could use align_data here just for optimization, or handle NaNs differently)
    temp_aligned = DataPreprocessor.align_data(returns)
    optimizer = PortfolioOptimizer(temp_aligned)
    weights = optimizer.optimize_sharpe()
    
    # Step 3: Select assets with meaningful weights
    selected_assets = [asset for asset, weight in weights.items() if abs(weight) > 0.01]
    print(f"Selected {len(selected_assets)} assets: {selected_assets}")
    
    # Step 4: Align ONLY the selected assets (preserves their maximum history)
    final_returns = DataPreprocessor.align_selected_assets(returns, selected_assets)
    final_prices = DataPreprocessor.align_selected_assets(prices, selected_assets)
    
    print(f"Final data shape: {final_returns.shape}")
    print(f"Preserved data range: {final_returns.index[0]} to {final_returns.index[-1]}")
    
    # Filter weights to selected assets
    filtered_weights = {asset: weights[asset] for asset in selected_assets}
    
    return filtered_weights, final_returns

if __name__ == "__main__":
    print("🔄 IMPROVED DATA PIPELINE INTEGRATION EXAMPLES")
    print("=" * 60)
    
    try:
        # Compare the approaches
        old_weights, old_data = old_workflow_example()
        print()
        new_weights, new_data = new_workflow_example()
        print()
        manual_weights, manual_data = manual_workflow_example()
        
        # Show the improvements
        print("\n📊 COMPARISON")
        print("=" * 60)
        print(f"Old workflow data periods:    {len(old_data)}")
        print(f"New workflow data periods:    {len(new_data)}")
        print(f"Manual workflow data periods: {len(manual_data)}")
        
        improvement = len(new_data) - len(old_data)
        if improvement > 0:
            print(f"\n🎉 New workflows preserve {improvement} additional periods!")
        
        print("\n💡 INTEGRATION TIPS:")
        print("1. Replace 'download_prices_and_returns' + 'align_data' with 'download_with_flexible_alignment'")
        print("2. Use 'optimize_then_align_workflow' for automatic handling")
        print("3. Use 'align_selected_assets' when you need manual control")
        print("4. Set align_all=False to preserve individual asset histories")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
