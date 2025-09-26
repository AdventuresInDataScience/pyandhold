"""Module for data preprocessing and cleaning."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List
from scipy import stats


class DataPreprocessor:
    """Handle data cleaning and preprocessing tasks."""
    
    @staticmethod
    def winsorize(
        data: pd.DataFrame,
        limits: Tuple[float, float] = (0.05, 0.05),
        axis: int = 0
    ) -> pd.DataFrame:
        """
        Winsorize data to remove outliers.
        
        Args:
            data: Input DataFrame
            limits: (lower, upper) percentile limits
            axis: Axis along which to winsorize
            
        Returns:
            Winsorized DataFrame
        """
        if axis == 0:
            return pd.DataFrame(
                stats.mstats.winsorize(data, limits=limits, axis=0),
                index=data.index,
                columns=data.columns
            )
        else:
            result = data.copy()
            for col in data.columns:
                result[col] = stats.mstats.winsorize(data[col], limits=limits)
            return result
    
    @staticmethod
    def align_data(
        data: pd.DataFrame,
        min_history: Optional[int] = None,
        handle_missing: str = 'drop'
    ) -> pd.DataFrame:
        """
        Align data to common date range where all assets have data.
        
        Args:
            data: Input DataFrame with asset prices/returns
            min_history: Minimum number of observations required
            handle_missing: How to handle missing data ('drop', 'forward_fill', 'interpolate')
            
        Returns:
            Aligned DataFrame
        """
        # Handle missing data
        if handle_missing == 'drop':
            data = data.dropna()
        elif handle_missing == 'forward_fill':
            data = data.fillna(method='ffill').dropna()
        elif handle_missing == 'interpolate':
            data = data.interpolate(method='linear').dropna()
        
        # Find common date range
        first_valid_idx = data.apply(lambda x: x.first_valid_index()).max()
        last_valid_idx = data.apply(lambda x: x.last_valid_index()).min()
        
        # Slice to common range
        data = data.loc[first_valid_idx:last_valid_idx]
        
        # Apply minimum history requirement
        if min_history and len(data) < min_history:
            raise ValueError(f"Insufficient data: {len(data)} rows, minimum required: {min_history}")
        
        return data
    
    @staticmethod
    def align_selected_assets(
        data: pd.DataFrame,
        selected_assets: List[str],
        min_history: Optional[int] = None,
        handle_missing: str = 'drop'
    ) -> pd.DataFrame:
        """
        Align data for only the selected assets, preserving maximum history for those assets.
        
        This method should be used AFTER optimization/asset selection to avoid losing
        valuable historical data due to other assets with shorter histories.
        
        Args:
            data: Input DataFrame with asset prices/returns
            selected_assets: List of asset columns to keep and align
            min_history: Minimum number of observations required
            handle_missing: How to handle missing data ('drop', 'forward_fill', 'interpolate')
            
        Returns:
            DataFrame with only selected assets, aligned to their common date range
        """
        # Filter to selected assets first
        if not all(asset in data.columns for asset in selected_assets):
            missing = [asset for asset in selected_assets if asset not in data.columns]
            raise ValueError(f"Selected assets not found in data: {missing}")
        
        selected_data = data[selected_assets].copy()
        
        # Now align only the selected assets
        return DataPreprocessor.align_data(
            selected_data, 
            min_history=min_history, 
            handle_missing=handle_missing
        )
    
    @staticmethod
    def remove_low_variance_assets(
        data: pd.DataFrame,
        threshold: float = 0.0001
    ) -> pd.DataFrame:
        """
        Remove assets with variance below threshold.
        
        Args:
            data: Input DataFrame
            threshold: Variance threshold
            
        Returns:
            DataFrame with low-variance assets removed
        """
        variances = data.var()
        keep_cols = variances[variances > threshold].index
        return data[keep_cols]
    
    @staticmethod
    def normalize_data(
        data: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize data using specified method.
        
        Args:
            data: Input DataFrame
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / (1.4826 * mad)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def optimize_then_align_workflow(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        optimization_func,
        min_history: Optional[int] = None,
        handle_missing: str = 'drop',
        min_weight_threshold: float = 0.001
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Recommended workflow: Optimize on unaligned data, then align selected assets.
        
        This preserves maximum historical data for the final selected portfolio assets.
        
        Args:
            prices: Raw price data (potentially unaligned)
            returns: Raw returns data (potentially unaligned) 
            optimization_func: Function that takes returns and returns optimal weights dict
            min_history: Minimum history required for final selected assets
            handle_missing: How to handle missing data in final alignment
            min_weight_threshold: Minimum weight to consider an asset "selected"
            
        Returns:
            Tuple of (aligned_prices, aligned_returns, optimal_weights)
        """
        print("🔄 Starting optimize-then-align workflow...")
        
        # Step 1: Get optimization results on raw (unaligned) data
        # This uses whatever data is available for each asset
        print(f"   1️⃣ Optimizing portfolio with {returns.shape[1]} assets...")
        optimal_weights = optimization_func(returns)
        
        # Step 2: Identify selected assets (those with meaningful weights)
        selected_assets = [
            asset for asset, weight in optimal_weights.items() 
            if abs(weight) > min_weight_threshold
        ]
        
        print(f"   2️⃣ Selected {len(selected_assets)} assets with weights > {min_weight_threshold:.1%}")
        print(f"        Selected assets: {selected_assets}")
        
        # Step 3: Align data for ONLY the selected assets
        print(f"   3️⃣ Aligning data for selected assets only...")
        aligned_prices = DataPreprocessor.align_selected_assets(
            prices, selected_assets, min_history, handle_missing
        )
        aligned_returns = DataPreprocessor.align_selected_assets(
            returns, selected_assets, min_history, handle_missing
        )
        
        # Step 4: Filter weights to only selected assets
        filtered_weights = {
            asset: weight for asset, weight in optimal_weights.items()
            if asset in selected_assets
        }
        
        print(f"   ✅ Final aligned data: {aligned_returns.shape[0]} periods, {aligned_returns.shape[1]} assets")
        print(f"   📈 Data range: {aligned_returns.index[0]} to {aligned_returns.index[-1]}")
        
        return aligned_prices, aligned_returns, filtered_weights