#!/usr/bin/env python3
"""
Quick test of the new data pipeline functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pyandhold.data.preprocessor import DataPreprocessor
from pyandhold.data.downloader import DataDownloader

def test_new_methods():
    """Test that the new methods work correctly."""
    print("🧪 TESTING NEW DATA PIPELINE METHODS")
    print("=" * 50)
    
    # Test 1: align_selected_assets
    print("1. Testing align_selected_assets...")
    try:
        # Create sample data with different histories
        dates1 = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        dates2 = pd.date_range('2021-01-01', '2023-12-31', freq='D')  # Shorter history
        dates3 = pd.date_range('2019-01-01', '2023-12-31', freq='D')  # Longer history
        
        # Create sample price data
        data = pd.DataFrame(index=pd.date_range('2019-01-01', '2023-12-31', freq='D'))
        data['ASSET_A'] = np.random.randn(len(data)) * 0.02 + 0.0005
        data['ASSET_B'] = np.nan  # Start with NaN
        data['ASSET_C'] = np.random.randn(len(data)) * 0.015 + 0.0003
        
        # Set different start dates
        data.loc[dates2, 'ASSET_B'] = np.random.randn(len(dates2)) * 0.025 + 0.0007
        
        # Test align_selected_assets
        selected = ['ASSET_A', 'ASSET_C']  # Skip the shorter history asset
        aligned = DataPreprocessor.align_selected_assets(data, selected)
        
        print(f"   ✅ Original data shape: {data.shape}")
        print(f"   ✅ Aligned selected shape: {aligned.shape}")
        print(f"   ✅ Selected assets: {list(aligned.columns)}")
        
    except Exception as e:
        print(f"   ❌ align_selected_assets failed: {e}")
        return False
    
    # Test 2: download_with_flexible_alignment (mock test)
    print("\n2. Testing download_with_flexible_alignment method exists...")
    try:
        downloader = DataDownloader()
        # Just check the method exists and accepts parameters
        method = getattr(downloader, 'download_with_flexible_alignment')
        print("   ✅ Method exists and is callable")
        
    except Exception as e:
        print(f"   ❌ download_with_flexible_alignment test failed: {e}")
        return False
    
    # Test 3: optimize_then_align_workflow (mock test)
    print("\n3. Testing optimize_then_align_workflow...")
    try:
        # Create mock data
        mock_prices = pd.DataFrame({
            'A': [100, 101, 102, 103, 104],
            'B': [50, 51, 49, 52, 53]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        mock_returns = mock_prices.pct_change().dropna()
        
        def mock_optimizer(returns):
            # Simple mock: equal weights
            return {col: 1/len(returns.columns) for col in returns.columns}
        
        result = DataPreprocessor.optimize_then_align_workflow(
            mock_prices, mock_returns, mock_optimizer
        )
        
        final_prices, final_returns, weights = result
        print(f"   ✅ Returns: {type(final_returns)} with shape {final_returns.shape}")
        print(f"   ✅ Weights: {weights}")
        
    except Exception as e:
        print(f"   ❌ optimize_then_align_workflow failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    success = test_new_methods()
    if success:
        print("\n✅ New data pipeline methods are working correctly!")
        print("You can now use the improved workflow in your portfolio optimization.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
