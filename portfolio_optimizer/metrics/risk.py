"""Module for calculating risk metrics."""

import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy import stats


class RiskMetrics:
    """Calculate various risk metrics."""
    
    @staticmethod
    def volatility(
        returns: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate annualized volatility.
        
        Args:
            returns: DataFrame of returns
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of annualized volatilities
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def downside_deviation(
        returns: pd.DataFrame,
        mar: float = 0.0,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate downside deviation.
        
        Args:
            returns: DataFrame of returns
            mar: Minimum acceptable return
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of downside deviations
        """
        downside_returns = returns[returns < mar]
        return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(prices: pd.DataFrame) -> pd.Series:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            Series of maximum drawdowns
        """
        cumulative = (prices / prices.iloc[0])
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown series over time.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            DataFrame of drawdowns
        """
        cumulative = (prices / prices.iloc[0])
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max
    
    @staticmethod
    def value_at_risk(
        returns: pd.DataFrame,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> pd.Series:
        """
        Calculate Value at Risk.
        
        Args:
            returns: DataFrame of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            Series of VaR values
        """
        if method == 'historical':
            return returns.quantile(1 - confidence_level)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def conditional_value_at_risk(
        returns: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.Series:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: DataFrame of returns
            confidence_level: Confidence level
            
        Returns:
            Series of CVaR values
        """
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def ulcer_index(prices: pd.DataFrame) -> pd.Series:
        """
        Calculate Ulcer Index.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            Series of Ulcer Index values
        """
        drawdowns = RiskMetrics.drawdown_series(prices)
        return np.sqrt((drawdowns ** 2).mean())
    
    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix of returns."""
        return returns.corr()
    
    @staticmethod
    def covariance_matrix(
        returns: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """Calculate annualized covariance matrix."""
        return returns.cov() * periods_per_year