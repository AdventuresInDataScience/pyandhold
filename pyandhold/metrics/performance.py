"""Module for calculating performance metrics."""

import pandas as pd
import numpy as np
from typing import Union, Optional
from .returns import ReturnMetrics
from .risk import RiskMetrics


class PerformanceMetrics:
    """Calculate various performance metrics."""
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: DataFrame of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Sharpe ratios
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        return excess_returns.mean() * np.sqrt(periods_per_year) / returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: pd.DataFrame,
        mar: float = 0.0,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: DataFrame of returns
            mar: Minimum acceptable return (annualized)
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Sortino ratios
        """
        mar_daily = mar / periods_per_year
        excess_returns = returns - mar_daily
        downside_dev = RiskMetrics.downside_deviation(returns, mar_daily, periods_per_year)
        return excess_returns.mean() * np.sqrt(periods_per_year) / downside_dev
    
    @staticmethod
    def calmar_ratio(
        prices: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Calmar ratio.
        
        Args:
            prices: DataFrame of prices
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Calmar ratios
        """
        cagr = ReturnMetrics.cagr(prices, periods_per_year)
        max_dd = RiskMetrics.max_drawdown(prices).abs()
        return cagr / max_dd
    
    @staticmethod
    def omega_ratio(
        returns: pd.DataFrame,
        mar: float = 0.0,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Omega ratio.
        
        Args:
            returns: DataFrame of returns
            mar: Minimum acceptable return (annualized)
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Omega ratios
        """
        mar_daily = mar / periods_per_year
        gains = returns[returns > mar_daily] - mar_daily
        losses = mar_daily - returns[returns <= mar_daily]
        
        if losses.sum().sum() == 0:
            return pd.Series(np.inf, index=returns.columns)
        
        return gains.sum() / losses.sum()
    
    @staticmethod
    def treynor_ratio(
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Treynor ratio.
        
        Args:
            returns: DataFrame of returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Treynor ratios
        """
        excess_returns = returns.mean() - risk_free_rate / periods_per_year
        betas = pd.Series(index=returns.columns)
        
        for col in returns.columns:
            covariance = returns[col].cov(benchmark_returns)
            variance = benchmark_returns.var()
            betas[col] = covariance / variance
        
        return excess_returns * periods_per_year / betas
    
    @staticmethod
    def jensens_alpha(
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Jensen's alpha.
        
        Args:
            returns: DataFrame of returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Jensen's alphas
        """
        rf_daily = risk_free_rate / periods_per_year
        excess_returns = returns - rf_daily
        excess_benchmark = benchmark_returns - rf_daily
        
        alphas = pd.Series(index=returns.columns)
        
        for col in returns.columns:
            covariance = excess_returns[col].cov(excess_benchmark)
            variance = excess_benchmark.var()
            beta = covariance / variance
            
            expected_return = rf_daily + beta * excess_benchmark.mean()
            alphas[col] = (returns[col].mean() - expected_return) * periods_per_year
        
        return alphas
    
    @staticmethod
    def information_ratio(
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate Information ratio.
        
        Args:
            returns: DataFrame of returns
            benchmark_returns: Series of benchmark returns
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of Information ratios
        """
        active_returns = returns.sub(benchmark_returns, axis=0)
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        return active_returns.mean() * periods_per_year / tracking_error
    
    @staticmethod
    def k_ratio(
        prices: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate K-ratio (measures consistency of returns).
        
        Args:
            prices: DataFrame of prices
            periods_per_year: Number of periods in a year
            
        Returns:
            Series of K-ratios
        """
        log_prices = np.log(prices)
        x = np.arange(len(log_prices))
        
        k_ratios = pd.Series(index=prices.columns)
        
        for col in prices.columns:
            y = log_prices[col].values
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                k_ratios[col] = np.nan
                continue
                
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Linear regression
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            y_pred = slope * x_clean + intercept
            residuals = y_clean - y_pred
            std_error = np.std(residuals)
            
            if std_error == 0:
                k_ratios[col] = np.inf if slope > 0 else -np.inf
            else:
                k_ratios[col] = slope * np.sqrt(periods_per_year) / std_error
        
        return k_ratios