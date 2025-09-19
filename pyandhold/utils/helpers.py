"""Utility functions for PyAndHold portfolio optimization."""

"""Utility functions for PyAndHold portfolio optimization."""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from scipy.stats import norm, t
import warnings

try:
    from IPython.display import display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(obj):
        print(obj)

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
from ..visualization.plots import PortfolioVisualizer
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
    
# #%%
# # get_optimized_weights
# # Eg Config
# tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM',
#             'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH', 'DIS', 'CURE',
#             'NFLX', 'INTC', 'CSCO', 'CMCSA', 'PEP', 'KO', 'MRK', 'ABT',
#             'COKE', 'WMT', 'T', 'VZ', 'HD', 'LOW', 'MA', 'V', 'SPY', 'ADBE',
#             'COST', 'CRM', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AMD', 'SBUX', 'MCD',
#              'TQQQ', 'SOXL', 'TPL', 'M']

# start_date='1980-01-01'
# end_date='2025-12-31'

# max_weight_limit = 0.3  # 30% max
# min_weight_limit = 0.0  # 0% min
# max_volatility = 0.9  # 90% maximum volatility constraint

# initial_capital=100000

# # Download data for constraint testing
# downloader = DataDownloader()
# prices, returns = downloader.download_prices_and_returns(
#     tickers=tickers,
#     start_date=start_date,
#     end_date=end_date
# )

# # Optimize with constraints
# optimizer = PortfolioOptimizer(returns)
# constraint_portfolios = {}
# constraint_builder = ConstraintBuilder()

# max_pos_constraint = constraint_builder.max_position_constraint(max_weight_limit) # MAX AND MIN POSITION example 
# min_pos_constraint = constraint_builder.min_position_constraint(min_weight_limit)

# # Combine both constraints
# combined_constraints = {
#     'max_position': max_pos_constraint,
#     'min_position': min_pos_constraint
# }

# # Use optimize_max_return with required max_volatility parameter
# optimized_weights = optimizer.optimize_max_return(
#     max_volatility=max_volatility,
#     constraints=combined_constraints
# )
# #%%
# # add_portfolio
# #%%
# # show_summary
# portfolio = Portfolio(
#         weights=weights,
#         start_date='2000-01-01',
#         end_date='2025-12-31',
#         rebalance_frequency='M',
#         initial_capital=100000
#     )
    
#     # Calculate everything
#     metrics = portfolio.calculate_metrics()
#     portfolio_value = portfolio.calculate_portfolio_value()
    
#     visualizer = PortfolioVisualizer()
    
#     # 1. Performance Line Chart
#     print("\n1. Creating performance line chart...")
#     perf_fig = visualizer.plot_performance(
#         portfolio_value,
#         title="Portfolio Performance Over Time"
#     )
#     perf_fig.show()
    
#     # 2. Returns Distribution Histogram with CDF
#     print("2. Creating returns distribution...")
#     dist_fig = visualizer.plot_returns_distribution(
#         portfolio.portfolio_returns,
#         title="Daily Returns Distribution"
#     )
#     dist_fig.show()
    
#     # 2b. Returns Statistics Table (Display as DataFrame)
#     print("2b. Creating returns statistics table...")
#     stats_df = visualizer.display_returns_statistics(
#         portfolio.portfolio_returns
#     )
#     print("Returns Statistics:")
#     display(stats_df)
    
#     # 3. Drawdown Chart
#     print("3. Creating drawdown chart...")
#     dd_fig = visualizer.plot_drawdown(
#         portfolio_value,
#         title="Portfolio Drawdown Analysis"
#     )
#     dd_fig.show()
    
#     # 3b. Top 10 Drawdown Periods Table (Display as DataFrame)
#     print("3b. Creating top 10 drawdown periods table...")
#     top_dd_df = visualizer.display_drawdown_periods(
#         portfolio_value,
#         top_n=10
#     )
#     print("Top 10 Drawdown Periods:")
#     display(top_dd_df)
    
#     # 4. Rolling Metrics Charts (Create individual charts)
#     print("4. Creating rolling metrics charts...")
    
#     returns = portfolio.portfolio_returns
#     print(f"  Portfolio returns available: {returns is not None}")
    
#     if returns is not None and len(returns) > 60:
#         print(f"  Returns data shape: {returns.shape}")
        
#         # Calculate rolling metrics manually for individual charts
#         window = 60
        
#         # 4a. Rolling Returns
#         print("  Creating 60-day rolling returns chart...")
#         rolling_returns = returns.rolling(window).mean()
#         returns_fig = visualizer.plot_single_metric_timeseries(
#             rolling_returns * 100,  # Convert to percentage
#             title="60-Day Rolling Mean Return",
#             y_label="Rolling Returns (%)",
#             color='blue',
#             show_mean=True
#         )
#         returns_fig.show()
        
#         # 4b. Rolling Volatility
#         print("  Creating 60-day rolling volatility chart...")
#         rolling_volatility = returns.rolling(window).std()
#         volatility_fig = visualizer.plot_single_metric_timeseries(
#             rolling_volatility * 100,  # Convert to percentage
#             title="60-Day Rolling Volatility",
#             y_label="Rolling Volatility (%)",
#             color='red',
#             show_mean=True
#         )
#         volatility_fig.show()
        
#         # 4c. Rolling Sharpe Ratio
#         print("  Creating 60-day rolling Sharpe ratio chart...")
#         rolling_mean = returns.rolling(window).mean()
#         rolling_std = returns.rolling(window).std()
#         rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
#         sharpe_fig = visualizer.plot_single_metric_timeseries(
#             rolling_sharpe,
#             title="60-Day Rolling Sharpe Ratio",
#             y_label="Rolling Sharpe Ratio",
#             color='green',
#             show_mean=True
#         )
#         sharpe_fig.show()
        
#         print("  ✓ All 60-day rolling metrics charts created successfully!")
#     else:
#         print("  ✗ Not enough data for 60-day rolling metrics")
    
#     # 4b. Extended rolling metrics with 126-day window
#     print("4b. Creating extended rolling metrics...")
    
#     if returns is not None and len(returns) > 126:
#         print(f"  Creating 126-day rolling metrics with {len(returns)} data points...")
#         window = 126
        
#         # 4b.1. 126-day Rolling Returns
#         print("  Creating 126-day rolling returns chart...")
#         rolling_returns_126 = returns.rolling(window).mean()
#         returns_126_fig = visualizer.plot_single_metric_timeseries(
#             rolling_returns_126 * 100,
#             title="126-Day Rolling Mean Return",
#             y_label="Rolling Returns (%)",
#             color='blue',
#             show_mean=True
#         )
#         returns_126_fig.show()
        
#         # 4b.2. 126-day Rolling Volatility
#         print("  Creating 126-day rolling volatility chart...")
#         rolling_volatility_126 = returns.rolling(window).std()
#         volatility_126_fig = visualizer.plot_single_metric_timeseries(
#             rolling_volatility_126 * 100,
#             title="126-Day Rolling Volatility",
#             y_label="Rolling Volatility (%)",
#             color='red',
#             show_mean=True
#         )
#         volatility_126_fig.show()
        
#         # 4b.3. 126-day Rolling Sharpe Ratio
#         print("  Creating 126-day rolling Sharpe ratio chart...")
#         rolling_mean_126 = returns.rolling(window).mean()
#         rolling_std_126 = returns.rolling(window).std()
#         rolling_sharpe_126 = rolling_mean_126 / rolling_std_126 * np.sqrt(252)
#         sharpe_126_fig = visualizer.plot_single_metric_timeseries(
#             rolling_sharpe_126,
#             title="126-Day Rolling Sharpe Ratio",
#             y_label="Rolling Sharpe Ratio",
#             color='green',
#             show_mean=True
#         )
#         sharpe_126_fig.show()
        
#         # 4b.4. 126-day Rolling Max Drawdown
#         print("  Creating 126-day rolling max drawdown chart...")
#         def rolling_max_drawdown(series, window):
#             def max_drawdown(x):
#                 if len(x) < 2:
#                     return 0
#                 cumulative = (1 + x).cumprod()
#                 running_max = cumulative.expanding().max()
#                 drawdown = (cumulative - running_max) / running_max
#                 return drawdown.min()
#             return series.rolling(window).apply(max_drawdown)
        
#         rolling_max_dd = rolling_max_drawdown(returns, window)
#         max_dd_fig = visualizer.plot_single_metric_timeseries(
#             rolling_max_dd * 100,  # Convert to percentage
#             title="126-Day Rolling Maximum Drawdown",
#             y_label="Rolling Max Drawdown (%)",
#             color='orange',
#             show_mean=True
#         )
#         max_dd_fig.show()
        
#         # 4b.5. 126-day Rolling Sortino Ratio (if you want it)
#         print("  Creating 126-day rolling Sortino ratio chart...")
#         def rolling_sortino(series, window, risk_free_rate=0):
#             def sortino_ratio(x):
#                 if len(x) < 2:
#                     return np.nan
#                 excess_returns = x - risk_free_rate/252  # Daily risk-free rate
#                 downside_returns = excess_returns[excess_returns < 0]
#                 if len(downside_returns) == 0:
#                     return np.nan
#                 downside_deviation = np.sqrt(np.mean(downside_returns**2))
#                 if downside_deviation == 0:
#                     return np.nan
#                 return (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)
#             return series.rolling(window).apply(sortino_ratio)
        
#         rolling_sortino_126 = rolling_sortino(returns, window)
#         sortino_fig = visualizer.plot_single_metric_timeseries(
#             rolling_sortino_126,
#             title="126-Day Rolling Sortino Ratio",
#             y_label="Rolling Sortino Ratio",
#             color='purple',
#             show_mean=True
#         )
#         sortino_fig.show()
        
#         print("  ✓ All 126-day rolling metrics charts created successfully!")
#     else:
#         print("  ✗ Not enough data for 126-day rolling metrics")
    
#     # 5. Correlation Heatmap
#     print("5. Creating correlation heatmap...")
#     corr_matrix = portfolio.get_correlation_matrix()
#     corr_fig = visualizer.plot_correlation_heatmap(
#         corr_matrix,
#         title="Asset Correlation Matrix"
#     )
#     corr_fig.show()
    
#     # 6. Weights Pie Chart
#     print("6. Creating weights pie chart...")
#     pie_fig = visualizer.plot_weights_pie(
#         weights,
#         title="Portfolio Allocation"
#     )
#     pie_fig.show()
    
#     # 6b. Monthly Returns Heatmap Table
#     print("6b. Creating monthly returns heatmap table...")
#     monthly_table_fig = visualizer.plot_monthly_returns_table(
#         portfolio.portfolio_returns,
#         title="Monthly Returns Heatmap"
#     )
#     monthly_table_fig.show()
    
#     # 7. Period Returns (Daily, Weekly, Monthly, Yearly)
#     print("7. Creating period returns analysis...")
#     period_fig = visualizer.plot_period_returns(
#         portfolio.portfolio_returns.to_frame(),
#         title="Returns by Period"
#     )
#     period_fig.update_layout(
#         width=1000,
#         height=800,
#         title_font_size=16
#     )
#     period_fig.show()
    
#     # 8. Cumulative returns chart
#     print("8. Creating cumulative returns chart...")
#     cum_returns_fig = visualizer.plot_cumulative_returns(
#         portfolio.portfolio_returns,
#         title="Cumulative Returns"
#     )
#     cum_returns_fig.show()
    
#     return portfolio, metrics

# portfolio, metrics = example_single_portfolio_all_visualizations()

# portfolio
# display(pd.DataFrame(metrics.items(), columns=["metric", "value"]))

# #%%
class Summariser:
    """Helper class for comparing multiple portfolio strategies."""
    
    def __init__(
        self,
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        rebalance_frequency: str = 'M',
        initial_capital: float = 100000,
        benchmark: str = "^GSPC"  # Default to S&P 500
    ):
        """
        Initialize the Summariser with common portfolio parameters.
        
        Args:
            start_date: Portfolio backtest start date
            end_date: Portfolio backtest end date
            rebalance_frequency: Rebalancing frequency
            initial_capital: Initial investment amount
            benchmark: Benchmark ticker symbol (optional)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.portfolios = {}  # Will store {name: weights} pairs
        self.portfolio_objects = {}  # Will store {name: Portfolio} objects
        self.metrics = {}  # Will store {name: metrics} dictionaries
        
    def add_portfolio(self, name: str, weights: Dict[str, float]):
        """
        Add a portfolio to the comparison.
        
        Args:
            name: Portfolio name/label
            weights: Dictionary of ticker:weight pairs
        """
        self.portfolios[name] = weights
        return self  # Allow method chaining
        
    def _create_portfolios(self):
        """Create Portfolio objects for each set of weights."""
        for name, weights in self.portfolios.items():
            portfolio = Portfolio(
                weights=weights,
                start_date=self.start_date,
                end_date=self.end_date,
                rebalance_frequency=self.rebalance_frequency,
                initial_capital=self.initial_capital,
                benchmark=self.benchmark
            )
            self.portfolio_objects[name] = portfolio
            self.metrics[name] = portfolio.calculate_metrics()
            
    def show_summary(self, metrics_to_show: List[str] = None):
        """
        Display a summary comparison of all portfolios.
        
        Args:
            metrics_to_show: List of metrics to include in comparison
        """
        # Create portfolios if they haven't been created
        if not self.portfolio_objects:
            self._create_portfolios()
            
        # Define default metrics if not specified
        if metrics_to_show is None:
            metrics_to_show = [
                'total_return', 'cagr', 'volatility', 'sharpe_ratio', 
                'sortino_ratio', 'max_drawdown', 'calmar_ratio'
            ]
            
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.metrics.items():
            row = {'Portfolio': name}
            for metric in metrics_to_show:
                if metric in metrics:
                    value = metrics[metric]
                    row[metric] = value
            comparison_data.append(row)
            
        # Convert to DataFrame and display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format columns appropriately
        for col in comparison_df.columns:
            if col == 'Portfolio':
                continue
            if 'return' in col.lower() or 'drawdown' in col.lower():
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}")
            elif 'ratio' in col.lower():
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.3f}")
            elif 'volatility' in col.lower():
                comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}")
                
        print("\n=== Portfolio Comparison ===")
        display(comparison_df)
        
        # Generate performance comparison chart
        self.plot_performance_comparison()
        
    def plot_performance_comparison(self):
        """Plot comparative performance of all portfolios."""
        if not self.portfolio_objects:
            self._create_portfolios()
            
        portfolio_values = {}
        for name, portfolio in self.portfolio_objects.items():
            portfolio_values[name] = portfolio.calculate_portfolio_value()
            
        visualizer = PortfolioVisualizer()
        fig = visualizer.plot_multiple_portfolios_comparison(
            portfolio_values,
            chart_type="performance",
            title="Portfolio Performance Comparison"
        )
        fig.show()
        
    def plot_drawdown_comparison(self):
        """Plot comparative drawdowns of all portfolios."""
        if not self.portfolio_objects:
            self._create_portfolios()
            
        portfolio_values = {}
        for name, portfolio in self.portfolio_objects.items():
            portfolio_values[name] = portfolio.calculate_portfolio_value()
            
        visualizer = PortfolioVisualizer()
        fig = visualizer.plot_multiple_portfolios_comparison(
            portfolio_values,
            chart_type="drawdown",
            title="Portfolio Drawdown Comparison"
        )
        fig.show()