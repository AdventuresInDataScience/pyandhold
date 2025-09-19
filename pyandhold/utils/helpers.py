"""Utility functions for PyAndHold portfolio optimization."""

"""Utility functions for PyAndHold portfolio optimization."""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from scipy.stats import norm, t
import warnings

from ..data.downloader import DataDownloader
from ..data.preprocessor import DataPreprocessor
from ..data.universe import StockUniverse
from ..metrics.returns import ReturnMetrics
from ..metrics.risk import RiskMetrics
from ..metrics.performance import PerformanceMetrics
from ..portfolio.portfolio import Portfolio
from ..portfolio.backtester import Backtester

from ..optimization.optimizers import PortfolioOptimizer
from ..optimization.constraints import ConstraintBuilder
from ..optimization.robust import RobustOptimizer
from ..visualization import PortfolioVisualizer
class PortfolioHelpers:
    """Helper functions for portfolio analysis."""
    
    @staticmethod
    def calculate_rolling_beta(
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Calculate rolling beta."""
        covariance = asset_returns.rolling(window).cov(market_returns)
        variance = market_returns.rolling(window).var()
        return covariance / variance
    
    @staticmethod
    def calculate_tracking_error(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate tracking error."""
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def monte_carlo_simulation(
        returns: pd.DataFrame,
        weights: Dict[str, float],
        n_simulations: int = 1000,
        n_days: int = 252
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation for portfolio.
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            n_simulations: Number of simulations
            n_days: Number of days to simulate
            
        Returns:
            DataFrame with simulation results
        """
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Portfolio statistics
        weights_array = np.array([weights[ticker] for ticker in returns.columns])
        portfolio_mean = np.dot(weights_array, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        # Run simulations
        simulations = np.zeros((n_days, n_simulations))
        
        for i in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(
                portfolio_mean,
                portfolio_std,
                n_days
            )
            
            # Calculate cumulative returns
            simulations[:, i] = np.cumprod(1 + random_returns)
        
        # Create DataFrame
        simulation_df = pd.DataFrame(simulations)
        
        # Calculate statistics
        results = pd.DataFrame({
            'mean': simulation_df.mean(axis=1),
            'median': simulation_df.median(axis=1),
            'percentile_5': simulation_df.quantile(0.05, axis=1),
            'percentile_95': simulation_df.quantile(0.95, axis=1),
            'std': simulation_df.std(axis=1)
        })
        
        return results
    
    @staticmethod
    def calculate_kelly_criterion(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Kelly criterion for position sizing.
        
        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Optimal fraction to invest
        """
        excess_returns = returns - risk_free_rate / 252
        mean_excess = excess_returns.mean()
        variance = excess_returns.var()
        
        if variance == 0:
            return 0
        
        kelly_fraction = mean_excess / variance
        
        # Apply Kelly criterion with safety factor (typically use 25% of full Kelly)
        return min(max(kelly_fraction * 0.25, 0), 1)
    
    @staticmethod
    def calculate_portfolio_turnover(
        weights_history: pd.DataFrame,
        rebalance_frequency: str = 'M'
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            weights_history: DataFrame with weight history
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Annual turnover rate
        """
        # Calculate weight changes
        weight_changes = weights_history.diff().abs().sum(axis=1)
        
        # Annualize based on frequency
        frequency_map = {'D': 252, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}
        periods_per_year = frequency_map.get(rebalance_frequency, 12)
        
        # Average turnover
        avg_turnover = weight_changes.mean()
        annual_turnover = avg_turnover * periods_per_year / 2  # Divide by 2 for buys+sells
        
        return annual_turnover
    
    @staticmethod
    def stress_test_portfolio(
        returns: pd.DataFrame,
        weights: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Stress test portfolio under different scenarios.
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            scenarios: Dictionary of scenarios with asset shocks
            
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks to returns
            stressed_returns = returns.copy()
            for ticker, shock in shocks.items():
                if ticker in stressed_returns.columns:
                    stressed_returns[ticker] = stressed_returns[ticker] * (1 + shock)
            
            # Calculate portfolio performance
            weights_array = np.array([weights.get(ticker, 0) for ticker in stressed_returns.columns])
            portfolio_returns = (stressed_returns * weights_array).sum(axis=1)
            
            results.append({
                'Scenario': scenario_name,
                'Total Return': (1 + portfolio_returns).prod() - 1,
                'Max Drawdown': ((portfolio_returns + 1).cumprod() / (portfolio_returns + 1).cumprod().cummax() - 1).min(),
                'Volatility': portfolio_returns.std() * np.sqrt(252),
                'Worst Day': portfolio_returns.min(),
                'Best Day': portfolio_returns.max()
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_risk_contribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each asset.
        
        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            
        Returns:
            Array of risk contributions
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        return contrib / contrib.sum()
    
    @staticmethod
    def calculate_diversification_ratio(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate diversification ratio.
        
        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix
            
        Returns:
            Diversification ratio
        """
        # Individual volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))
        
        # Weighted average of individual volatilities
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        return weighted_avg_vol / portfolio_vol
    
class Summariser:
        pass  # Placeholder for summariser methods

    # get_optimized_weights
    # Config
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM',
            'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH', 'DIS', 'CURE',
            'NFLX', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'KO', 'MRK', 'ABT',
            'COKE', 'WMT', 'T', 'VZ', 'HD', 'LOW', 'MA', 'V', 'SPY', 'ADBE',
            'COST', 'CRM', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AMD', 'SBUX', 'MCD',
             'TQQQ', 'SOXL', 'TPL', 'M']

start_date='1980-01-01'
end_date='2025-12-31'

max_weight_limit = 0.3  # 30% max
min_weight_limit = 0.0  # 0% min
max_volatility = 0.9  # 90% maximum volatility constraint

initial_capital=100000

# Download data for constraint testing
downloader = DataDownloader()
prices, returns = downloader.download_prices_and_returns(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date
)

# Optimize with constraints
optimizer = PortfolioOptimizer(returns)
constraint_portfolios = {}
constraint_builder = ConstraintBuilder()

max_pos_constraint = constraint_builder.max_position_constraint(max_weight_limit) # MAX AND MIN POSITION example 
min_pos_constraint = constraint_builder.min_position_constraint(min_weight_limit)

# Combine both constraints
combined_constraints = {
    'max_position': max_pos_constraint,
    'min_position': min_pos_constraint
}

# Use optimize_max_return with required max_volatility parameter
optimized_weights = optimizer.optimize_max_return(
    max_volatility=max_volatility,
    constraints=combined_constraints
)

# add_portfolio

# show_summary
