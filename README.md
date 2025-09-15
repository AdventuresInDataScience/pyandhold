# Portfolio Optimizer

A comprehensive Python library for portfolio optimization and analysis with advanced features including robust optimization, backtesting, and interactive visualization.

## 📁 Repository Structure

```
portfolio_optimizer/
├── setup.py                    # Package setup and dependencies
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── portfolio_optimizer/        # Main package
│   ├── __init__.py
│   ├── data/                   # Data acquisition and preprocessing
│   │   ├── downloader.py      # Yahoo Finance data downloader
│   │   ├── preprocessor.py    # Data cleaning and alignment
│   │   └── universe.py        # Stock universe definitions (S&P500, etc.)
│   ├── metrics/                # Performance measurement
│   │   ├── returns.py         # Return calculations (CAGR, rolling, etc.)
│   │   ├── risk.py            # Risk metrics (VaR, drawdown, volatility)
│   │   └── performance.py     # Performance ratios (Sharpe, Sortino, etc.)
│   ├── optimization/           # Portfolio optimization engines
│   │   ├── optimizers.py      # Core optimization methods
│   │   ├── constraints.py     # Constraint builders
│   │   └── robust.py          # Robust optimization techniques
│   ├── portfolio/              # Portfolio management
│   │   ├── portfolio.py       # Main Portfolio class
│   │   └── backtester.py      # Backtesting engine
│   ├── visualization/          # Plotting and dashboards
│   │   ├── plots.py           # Plotly visualizations
│   │   └── dashboard.py       # Streamlit interactive dashboard
│   └── utils/                  # Utility functions
│       └── helpers.py          # Monte Carlo, stress testing, etc.
└── tests/                      # Test suite
    ├── test_data.py
    ├── test_metrics.py
    ├── test_optimization.py
    └── test_portfolio.py
```

## 🚀 How It Works

### Architecture Overview

The library follows a modular architecture with clear separation of concerns:

1. **Data Layer** (`data/`): Handles all data acquisition and preprocessing
   - Downloads historical prices from Yahoo Finance
   - Provides pre-defined stock universes (S&P 500, sector ETFs)
   - Cleans data (winsorization, alignment, normalization)

2. **Metrics Layer** (`metrics/`): Calculates portfolio analytics
   - Returns: Simple, log, cumulative, rolling, period-based
   - Risk: Volatility, VaR, CVaR, max drawdown, Ulcer Index
   - Performance: Sharpe, Sortino, Calmar, Omega, Information Ratio, etc.

3. **Optimization Layer** (`optimization/`): Portfolio weight optimization
   - Traditional: Mean-variance, maximum Sharpe, risk parity
   - Advanced: CVaR optimization, custom constraints
   - Robust: Cross-validation, bootstrap, shrinkage estimators

4. **Portfolio Layer** (`portfolio/`): Portfolio management and analysis
   - Portfolio class: Tracks positions, calculates returns, handles rebalancing
   - Backtester: Historical simulation with costs and slippage

5. **Visualization Layer** (`visualization/`): Interactive charts and dashboards
   - Plotly charts: Performance, drawdown, correlation, efficient frontier
   - Streamlit dashboard: Full GUI for non-programmers

### Data Flow

```
Historical Data → Preprocessing → Returns Calculation
                                          ↓
                              Optimization Engine ← Constraints
                                          ↓
                                  Optimal Weights
                                          ↓
                              Portfolio Construction
                                          ↓
                         Backtesting/Forward Testing
                                          ↓
                            Performance Analytics
                                          ↓
                              Visualization/Reports
```

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/portfolio_optimizer.git
cd portfolio_optimizer

# Install dependencies
pip install -r requirements.txt

# Install package
python setup.py install

# Or for development
pip install -e .
```

## 📚 Examples

### Example 1: Basic Portfolio Analysis

```python
from portfolio_optimizer import Portfolio

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
```

### Example 2: Portfolio Optimization

```python
from portfolio_optimizer import DataDownloader, PortfolioOptimizer

# Download data
downloader = DataDownloader()
prices = downloader.download_data(
    ['SPY', 'TLT', 'GLD', 'VNQ', 'DBC'],  # Stocks, Bonds, Gold, REITs, Commodities
    start_date='2019-01-01',
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
```

### Example 3: Robust Optimization with Cross-Validation

```python
from portfolio_optimizer import DataDownloader, RobustOptimizer

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
```

### Example 4: Advanced Backtesting

```python
from portfolio_optimizer import DataDownloader, Backtester

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
```

### Example 5: Using Pre-defined Universes

```python
from portfolio_optimizer import StockUniverse, Portfolio, PortfolioOptimizer

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
```

### Example 6: Monte Carlo Simulation

```python
from portfolio_optimizer import DataDownloader, PortfolioHelpers

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
```

### Example 7: Efficient Frontier Visualization

```python
from portfolio_optimizer import DataDownloader, PortfolioOptimizer
from portfolio_optimizer.visualization import PortfolioVisualizer

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
```

## 🎯 Interactive Dashboard

Launch the Streamlit dashboard for a full GUI experience:

```bash
streamlit run portfolio_optimizer/visualization/dashboard.py
```

Features:
- Asset selection (manual, indices, sectors)
- Date range selection
- Multiple optimization methods
- Real-time portfolio analysis
- Interactive charts and metrics
- Strategy comparison
- Export results

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_optimization.py -v

# Run with coverage
pytest tests/ --cov=portfolio_optimizer --cov-report=html
```

## 📊 Available Metrics

### Return Metrics
- Total Return
- CAGR (Compound Annual Growth Rate)
- Annualized Return
- Period Returns (Daily, Weekly, Monthly, Yearly)
- Rolling Returns

### Risk Metrics
- Volatility (Standard Deviation)
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Ulcer Index
- Downside Deviation
- Beta

### Performance Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Treynor Ratio
- Jensen's Alpha
- Information Ratio
- K-Ratio

## 🔧 Configuration

### Custom Constraints

```python
from portfolio_optimizer import PortfolioOptimizer, ConstraintBuilder

# Sector constraints
sector_mapping = {
    'tech': [0, 1],      # First two assets are tech
    'finance': [2, 3]    # Next two are finance
}
sector_limits = {
    'tech': (0.2, 0.6),     # Tech: 20-60%
    'finance': (0.1, 0.4)   # Finance: 10-40%
}

constraints = ConstraintBuilder.sector_constraint(sector_mapping, sector_limits)

# Cardinality constraint (limit number of assets)
cardinality = ConstraintBuilder.cardinality_constraint(
    min_assets=3,
    max_assets=8
)

# Apply constraints
optimizer = PortfolioOptimizer(returns)
weights = optimizer.optimize_sharpe(constraints={'custom': constraints})
```

## 📈 Extending the Library

The modular design makes it easy to add new features:

```python
# Custom metric
class CustomMetrics:
    @staticmethod
    def custom_ratio(returns, benchmark):
        # Your implementation
        return custom_value

# Custom optimization
class CustomOptimizer(PortfolioOptimizer):
    def optimize_custom(self, **kwargs):
        # Your optimization logic
        return weights

# Custom visualization
class CustomVisualizer(PortfolioVisualizer):
    def plot_custom(self, data):
        # Your plotting logic
        return fig
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Yahoo Finance for market data
- NumPy, Pandas, SciPy for numerical computing
- CVXPY for convex optimization
- Plotly for interactive visualizations
- Streamlit for the dashboard framework