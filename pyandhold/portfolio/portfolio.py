"""Main Portfolio class for portfolio management."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from ..data.downloader import DataDownloader
from ..data.preprocessor import DataPreprocessor
from ..metrics.returns import ReturnMetrics
from ..metrics.risk import RiskMetrics
from ..metrics.performance import PerformanceMetrics


class Portfolio:
    """Main portfolio class for analysis and management."""
    
    def __init__(
        self,
        weights: Dict[str, float],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        benchmark: str = "^GSPC",
        rebalance_frequency: Optional[str] = None,
        transaction_cost: float = 0.001,
        initial_capital: float = 100000
    ):
        """
        Initialize Portfolio.
        
        Args:
            weights: Dictionary of ticker: weight pairs
            start_date: Start date for analysis
            end_date: End date for analysis
            benchmark: Benchmark ticker
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y', None)
            transaction_cost: Transaction cost as percentage
            initial_capital: Initial portfolio value
        """
        self.weights = weights
        self.tickers = list(weights.keys())
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Data containers
        self.prices = None
        self.returns = None
        self.benchmark_prices = None
        self.benchmark_returns = None
        self.portfolio_returns = None
        self.portfolio_value = None
        
        # Metrics
        self.metrics = {}
        
        # Initialize data downloader
        self.downloader = DataDownloader()
        
    def fetch_data(self):
        """Fetch price data for portfolio assets and benchmark."""
        # Download asset prices
        self.prices = self.downloader.download_data(
            self.tickers,
            self.start_date,
            self.end_date
        )
        
        # Calculate returns
        self.returns = ReturnMetrics.calculate_returns(self.prices)
        
        # Download benchmark data
        self.benchmark_prices = self.downloader.download_data(
            [self.benchmark],
            self.start_date,
            self.end_date
        ).iloc[:, 0]
        
        self.benchmark_returns = ReturnMetrics.calculate_returns(
            self.benchmark_prices.to_frame()
        ).iloc[:, 0]
        
        # Align data
        self.prices = DataPreprocessor.align_data(self.prices)
        self.returns = DataPreprocessor.align_data(self.returns)
        
    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns based on weights."""
        if self.returns is None:
            self.fetch_data()
        
        weights_array = np.array([self.weights[ticker] for ticker in self.returns.columns])
        self.portfolio_returns = (self.returns * weights_array).sum(axis=1)
        return self.portfolio_returns
    
    def calculate_portfolio_value(self) -> pd.Series:
        """Calculate portfolio value over time with rebalancing."""
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()
        
        if self.rebalance_frequency is None:
            # No rebalancing - simple buy and hold
            self.portfolio_value = self.initial_capital * (1 + self.portfolio_returns).cumprod()
        else:
            # With rebalancing
            value = self.initial_capital
            values = []
            dates = []
            
            # Group returns by rebalance period
            grouped = self.returns.resample(self.rebalance_frequency)
            
            for period_start, period_returns in grouped:
                if len(period_returns) == 0:
                    continue
                
                # Apply transaction costs at rebalancing
                value *= (1 - self.transaction_cost)
                
                # Calculate period returns with fixed weights
                weights_array = np.array([self.weights[ticker] for ticker in period_returns.columns])
                period_portfolio_returns = (period_returns * weights_array).sum(axis=1)
                
                # Update portfolio value for this period
                for date, ret in period_portfolio_returns.items():
                    value *= (1 + ret)
                    values.append(value)
                    dates.append(date)
            
            self.portfolio_value = pd.Series(values, index=dates)
        
        return self.portfolio_value
    
    def calculate_metrics(self) -> Dict:
        """Calculate all portfolio metrics."""
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()
        if self.portfolio_value is None:
            self.calculate_portfolio_value()
        
        # Convert Series to DataFrame for metrics functions
        portfolio_returns_df = self.portfolio_returns.to_frame('portfolio')
        portfolio_value_df = self.portfolio_value.to_frame('portfolio')
        
        # Return metrics
        self.metrics['total_return'] = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        self.metrics['cagr'] = ReturnMetrics.cagr(portfolio_value_df).iloc[0]
        self.metrics['annualized_return'] = ReturnMetrics.annualized_return(
            portfolio_returns_df
        ).iloc[0]
        
        # Risk metrics
        self.metrics['volatility'] = RiskMetrics.volatility(
            portfolio_returns_df
        ).iloc[0]
        self.metrics['max_drawdown'] = RiskMetrics.max_drawdown(
            portfolio_value_df
        ).iloc[0]
        self.metrics['ulcer_index'] = RiskMetrics.ulcer_index(
            portfolio_value_df
        ).iloc[0]
        self.metrics['var_95'] = RiskMetrics.value_at_risk(
            portfolio_returns_df, 0.95
        ).iloc[0]
        self.metrics['cvar_95'] = RiskMetrics.conditional_value_at_risk(
            portfolio_returns_df, 0.95
        ).iloc[0]
        
        # Performance metrics
        self.metrics['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(
            portfolio_returns_df
        ).iloc[0]
        self.metrics['sortino_ratio'] = PerformanceMetrics.sortino_ratio(
            portfolio_returns_df
        ).iloc[0]
        self.metrics['calmar_ratio'] = PerformanceMetrics.calmar_ratio(
            portfolio_value_df
        ).iloc[0]
        self.metrics['omega_ratio'] = PerformanceMetrics.omega_ratio(
            portfolio_returns_df
        ).iloc[0]
        
        # Benchmark-relative metrics
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(self.portfolio_returns.index)
            self.metrics['treynor_ratio'] = PerformanceMetrics.treynor_ratio(
                portfolio_returns_df,
                aligned_benchmark
            ).iloc[0]
            self.metrics['jensens_alpha'] = PerformanceMetrics.jensens_alpha(
                portfolio_returns_df,
                aligned_benchmark
            ).iloc[0]
            self.metrics['information_ratio'] = PerformanceMetrics.information_ratio(
                portfolio_returns_df,
                aligned_benchmark
            ).iloc[0]
        
        self.metrics['k_ratio'] = PerformanceMetrics.k_ratio(
            portfolio_value_df
        ).iloc[0]
        
        return self.metrics
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of portfolio assets."""
        if self.returns is None:
            self.fetch_data()
        return RiskMetrics.correlation_matrix(self.returns)
    
    def get_rolling_metrics(
        self,
        window: int = 252,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics.
        
        Args:
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if self.portfolio_returns is None:
            self.calculate_portfolio_returns()
        
        if metrics is None:
            metrics = ['sharpe_ratio', 'volatility', 'returns']
        
        rolling_data = pd.DataFrame(index=self.portfolio_returns.index)
        
        for metric in metrics:
            if metric == 'sharpe_ratio':
                rolling_data[metric] = self.portfolio_returns.rolling(window).apply(
                    lambda x: PerformanceMetrics.sharpe_ratio(pd.DataFrame(x))
                )
            elif metric == 'volatility':
                rolling_data[metric] = self.portfolio_returns.rolling(window).std() * np.sqrt(252)
            elif metric == 'returns':
                rolling_data[metric] = self.portfolio_returns.rolling(window).mean() * 252
        
        return rolling_data.dropna()