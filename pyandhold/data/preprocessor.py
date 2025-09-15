"""Module for data preprocessing and cleaning."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
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