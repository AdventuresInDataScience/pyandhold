## 📚 Examples

### Example 1: Basic Portfolio Analysis

#%%
from pyandhold import Portfolio

# Define portfolio weights
weights = {
    'AAPL': 0.25,
    'MSFT': 0.25,
    'GOOGL': 0.25,
    'AMZN': 0.25
}

# Create portfolio
portfolio = Portfolio(
    weights=weights,
    start_date='2020-01-01',
    end_date='2023-12-31',
    rebalance_frequency='Q',  # Quarterly rebalancing
    initial_capital=100000
)

# Calculate metrics
metrics = portfolio.calculate_metrics()

print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annual Return (CAGR): {metrics['cagr']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")

#%% ### Example 2: Portfolio Optimization

from pyandhold import DataDownloader, PortfolioOptimizer

# Download data
downloader = DataDownloader()
prices = downloader.download_data(
    ['SPY', 'TLT', 'GLD', 'VNQ', 'DBC'],  # Stocks, Bonds, Gold, REITs, Commodities
    start_date='2000-01-01',
    end_date='2023-12-31'
)

# Calculate returns
returns = prices.pct_change().dropna()

# Initialize optimizer
optimizer = PortfolioOptimizer(returns)

# Method 1: Maximize Sharpe Ratio
weights_sharpe = optimizer.optimize_sharpe(
    weight_bounds=(0.05, 0.40)  # Min 5%, Max 40% per asset
)

# Method 2: Minimum Variance
weights_minvar = optimizer.optimize_min_variance(
    weight_bounds=(0.0, 1.0)
)

# Method 3: Risk Parity
weights_riskparity = optimizer.optimize_risk_parity()

# Method 4: Maximum Return with Volatility Constraint
weights_maxret = optimizer.optimize_max_return(
    max_volatility=0.15,  # 15% annual volatility limit
    weight_bounds=(0.0, 0.5)
)

print("Optimal Weights (Max Sharpe):")
for ticker, weight in weights_sharpe.items():
    print(f"  {ticker}: {weight:.1%}")


#%% ### Example 3: Robust Optimization with Cross-Validation

from pyandhold import DataDownloader, RobustOptimizer

# Download data
downloader = DataDownloader()
returns = downloader.download_returns(
    ['QQQ', 'IWM', 'EFA', 'EEM', 'AGG'],
    start_date='2018-01-01'
)

# Robust optimization
robust_opt = RobustOptimizer(returns)

# Cross-validation (reduces overfitting)
cv_weights = robust_opt.cross_validation_optimize(
    optimization_method='sharpe',
    n_splits=5,
    weight_bounds=(0.1, 0.4)
)

# Bootstrap (parameter uncertainty)
bootstrap_weights = robust_opt.bootstrap_optimize(
    optimization_method='min_variance',
    n_iterations=100,
    weight_bounds=(0.05, 0.5)
)

# Shrinkage (covariance estimation)
shrinkage_weights = robust_opt.shrinkage_covariance_optimize(
    optimization_method='sharpe',
    shrinkage_factor=0.2
)

#%% ### Example 4: Advanced Backtesting
### Example 4: Advanced Backtesting

from pyandhold import DataDownloader, Backtester

# Download data
downloader = DataDownloader()
prices = downloader.download_data(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    start_date='2020-01-01'
)

# Initialize backtester
backtester = Backtester(
    data=prices,
    initial_capital=100000,
    commission=0.001,  # 0.1% commission
    slippage=0.0005    # 0.05% slippage
)

# Strategy 1: Fixed weights with monthly rebalancing
weights = {'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.2, 'AMZN': 0.15, 'META': 0.1}
results = backtester.backtest_fixed_weights(
    weights=weights,
    rebalance_frequency='M',
    leverage=1.0
)

# Calculate performance metrics
metrics = backtester.calculate_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Total Commission Paid: ${metrics['total_commission']:.2f}")

# Strategy 2: Dynamic strategy with custom logic
def momentum_strategy(historical_data, lookback=20):
    """Simple momentum strategy: overweight recent winners."""
    recent_returns = historical_data.iloc[-lookback:].mean()
    positive_returns = recent_returns[recent_returns > 0]
    
    if len(positive_returns) == 0:
        # Equal weight if no positive returns
        weights = {col: 1/len(historical_data.columns) 
                  for col in historical_data.columns}
    else:
        # Weight by positive returns
        total = positive_returns.sum()
        weights = {col: positive_returns.get(col, 0) / total 
                  for col in historical_data.columns}
    
    return weights

# Backtest dynamic strategy
dynamic_results = backtester.backtest_dynamic_strategy(
    strategy_func=momentum_strategy,
    lookback_period=60,
    rebalance_frequency='M',
    lookback=20
)
#%% ### Example 5: Using Pre-defined Universes
### Example 5: Using Pre-defined Universes

from pyandhold import StockUniverse, Portfolio, PortfolioOptimizer

# Get sector ETFs
sector_etfs = StockUniverse.get_sector_etfs()
print("Available sectors:", list(sector_etfs.keys()))

# Download sector data
downloader = DataDownloader()
prices = downloader.download_data(
    list(sector_etfs.values()),
    start_date='2020-01-01'
)

# Optimize sector allocation
returns = prices.pct_change().dropna()
optimizer = PortfolioOptimizer(returns)
optimal_weights = optimizer.optimize_sharpe()

# Create sector-rotated portfolio
portfolio = Portfolio(
    weights=optimal_weights,
    start_date='2020-01-01',
    rebalance_frequency='M'
)

metrics = portfolio.calculate_metrics()
#%% ### Example 6: Monte Carlo Simulation
### Example 6: Monte Carlo Simulation

from pyandhold import DataDownloader, PortfolioHelpers

# Download data
downloader = DataDownloader()
returns = downloader.download_returns(
    ['SPY', 'TLT', 'GLD'],
    start_date='2015-01-01'
)

# Define portfolio weights
weights = {'SPY': 0.6, 'TLT': 0.3, 'GLD': 0.1}

# Run Monte Carlo simulation
simulation_results = PortfolioHelpers.monte_carlo_simulation(
    returns=returns,
    weights=weights,
    n_simulations=1000,
    n_days=252  # 1 year forward
)

print(f"Expected 1-year return: {simulation_results['mean'].iloc[-1] - 1:.2%}")
print(f"5th percentile: {simulation_results['percentile_5'].iloc[-1] - 1:.2%}")
print(f"95th percentile: {simulation_results['percentile_95'].iloc[-1] - 1:.2%}")

#%% ### Example 7: Efficient Frontier Visualization
### Example 7: Efficient Frontier Visualization

from pyandhold import DataDownloader, PortfolioOptimizer
from pyandhold.visualization import PortfolioVisualizer

# Download data
downloader = DataDownloader()
returns = downloader.download_returns(
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    start_date='2020-01-01'
)

# Calculate efficient frontier
optimizer = PortfolioOptimizer(returns)
frontier = optimizer.efficient_frontier(
    n_portfolios=50,
    weight_bounds=(0, 1)
)

# Visualize
visualizer = PortfolioVisualizer()
fig = visualizer.plot_efficient_frontier(frontier)
fig.show()

#%%
## 🎯 Interactive Dashboard

Launch the Streamlit dashboard for a full GUI experience:

```bash
streamlit run pyandhold/visualization/dashboard.py
```

Features:
- Asset selection (manual, indices, sectors)
- Date range selection
- Multiple optimization methods
- Real-time portfolio analysis
- Interactive charts and metrics
- Strategy comparison
- Export results
