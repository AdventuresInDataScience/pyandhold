"""Module for calculating return-based metrics."""

import pandas as pd
import numpy as np
from typing import Union, Optional


class ReturnMetrics:
    """Calculate various return-based metrics."""
    
    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices: DataFrame of prices
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame of returns
        """
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"Unknown return method: {method}")
    
    @staticmethod
    def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def annualized_return(
        returns: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate annualized returns.
        
        Args:
            returns: DataFrame of returns
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of annualized returns
        """
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        years = n_periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def cagr(
        prices: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            prices: DataFrame of prices
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of CAGR values
        """
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        n_periods = len(prices) - 1
        years = n_periods / periods_per_year
        return (end_price / start_price) ** (1 / years) - 1
    
    @staticmethod
    def rolling_returns(
        returns: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling returns.
        
        Args:
            returns: DataFrame of returns
            window: Rolling window size
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame of rolling returns
        """
        if min_periods is None:
            min_periods = window
        return returns.rolling(window=window, min_periods=min_periods).apply(
            lambda x: (1 + x).prod() - 1
        )
    
    @staticmethod
    def period_returns(
        returns: pd.DataFrame,
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Calculate returns by period (daily, weekly, monthly, yearly).
        
        Args:
            returns: DataFrame of returns
            freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            DataFrame of period returns
        """
        return returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)