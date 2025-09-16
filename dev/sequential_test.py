"""
Comprehensive examples for pyandhold package.
Demonstrates all visualization types, portfolio comparisons, constraints, and features.
"""
#%% imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

# Import pyandhold components
from pyandhold import Portfolio, DataDownloader, PortfolioOptimizer
from pyandhold.optimization import RobustOptimizer, ConstraintBuilder
from pyandhold.portfolio import Backtester
from pyandhold.data import StockUniverse, DataPreprocessor
from pyandhold.visualization import PortfolioVisualizer
from pyandhold.utils import PortfolioHelpers
from pyandhold.metrics import ReturnMetrics, RiskMetrics, PerformanceMetrics

#%% 1
# ==============================================================================
# SECTION 1: SINGLE PORTFOLIO VISUALIZATIONS
# ==============================================================================

def example_single_portfolio_all_visualizations():
    """Demonstrate all visualization types for a single portfolio."""
    
    print("\n" + "="*60)
    print("SINGLE PORTFOLIO - ALL VISUALIZATIONS")
    print("="*60)
    
    # Create portfolio
    weights = {
        'AAPL': 0.10,
        'MSFT': 0.10,
        'COST': 0.10,
        'AMZN': 0.10,
        'COKE': 0.05,
        'KO': 0.05,
        'GLD': 0.10,
        'UNH': 0.10,
        'HD': 0.10,
        'MA': 0.05,
        'V': 0.05,
        'META': 0.10
    }
    
    portfolio = Portfolio(
        weights=weights,
        start_date='2000-01-01',
        end_date='2025-12-31',
        rebalance_frequency='M',
        initial_capital=100000
    )
    
    # Calculate everything
    metrics = portfolio.calculate_metrics()
    portfolio_value = portfolio.calculate_portfolio_value()
    
    visualizer = PortfolioVisualizer()
    
    # 1. Performance Line Chart
    print("\n1. Creating performance line chart...")
    perf_fig = visualizer.plot_performance(
        portfolio_value,
        title="Portfolio Performance Over Time"
    )
    perf_fig.show()
    
    # 2. Returns Distribution Histogram with CDF
    print("2. Creating returns distribution...")
    dist_fig = visualizer.plot_returns_distribution(
        portfolio.portfolio_returns,
        title="Daily Returns Distribution"
    )
    dist_fig.show()
    
    # 2b. Returns Statistics Table (Display as DataFrame)
    print("2b. Creating returns statistics table...")
    stats_df = visualizer.display_returns_statistics(
        portfolio.portfolio_returns
    )
    print("Returns Statistics:")
    display(stats_df)
    
    # 3. Drawdown Chart
    print("3. Creating drawdown chart...")
    dd_fig = visualizer.plot_drawdown(
        portfolio_value,
        title="Portfolio Drawdown Analysis"
    )
    dd_fig.show()
    
    # 3b. Top 10 Drawdown Periods Table (Display as DataFrame)
    print("3b. Creating top 10 drawdown periods table...")
    top_dd_df = visualizer.display_drawdown_periods(
        portfolio_value,
        top_n=10
    )
    print("Top 10 Drawdown Periods:")
    display(top_dd_df)
    
    # 4. Rolling Metrics Charts (Create individual charts)
    print("4. Creating rolling metrics charts...")
    
    returns = portfolio.portfolio_returns
    print(f"  Portfolio returns available: {returns is not None}")
    
    if returns is not None and len(returns) > 60:
        print(f"  Returns data shape: {returns.shape}")
        
        # Calculate rolling metrics manually for individual charts
        window = 60
        
        # 4a. Rolling Returns
        print("  Creating 60-day rolling returns chart...")
        rolling_returns = returns.rolling(window).mean()
        returns_fig = visualizer.plot_single_metric_timeseries(
            rolling_returns * 100,  # Convert to percentage
            title="60-Day Rolling Mean Return",
            y_label="Rolling Returns (%)",
            color='blue',
            show_mean=True
        )
        returns_fig.show()
        
        # 4b. Rolling Volatility
        print("  Creating 60-day rolling volatility chart...")
        rolling_volatility = returns.rolling(window).std()
        volatility_fig = visualizer.plot_single_metric_timeseries(
            rolling_volatility * 100,  # Convert to percentage
            title="60-Day Rolling Volatility",
            y_label="Rolling Volatility (%)",
            color='red',
            show_mean=True
        )
        volatility_fig.show()
        
        # 4c. Rolling Sharpe Ratio
        print("  Creating 60-day rolling Sharpe ratio chart...")
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
        sharpe_fig = visualizer.plot_single_metric_timeseries(
            rolling_sharpe,
            title="60-Day Rolling Sharpe Ratio",
            y_label="Rolling Sharpe Ratio",
            color='green',
            show_mean=True
        )
        sharpe_fig.show()
        
        print("  ✓ All 60-day rolling metrics charts created successfully!")
    else:
        print("  ✗ Not enough data for 60-day rolling metrics")
    
    # 4b. Extended rolling metrics with 126-day window
    print("4b. Creating extended rolling metrics...")
    
    if returns is not None and len(returns) > 126:
        print(f"  Creating 126-day rolling metrics with {len(returns)} data points...")
        window = 126
        
        # 4b.1. 126-day Rolling Returns
        print("  Creating 126-day rolling returns chart...")
        rolling_returns_126 = returns.rolling(window).mean()
        returns_126_fig = visualizer.plot_single_metric_timeseries(
            rolling_returns_126 * 100,
            title="126-Day Rolling Mean Return",
            y_label="Rolling Returns (%)",
            color='blue',
            show_mean=True
        )
        returns_126_fig.show()
        
        # 4b.2. 126-day Rolling Volatility
        print("  Creating 126-day rolling volatility chart...")
        rolling_volatility_126 = returns.rolling(window).std()
        volatility_126_fig = visualizer.plot_single_metric_timeseries(
            rolling_volatility_126 * 100,
            title="126-Day Rolling Volatility",
            y_label="Rolling Volatility (%)",
            color='red',
            show_mean=True
        )
        volatility_126_fig.show()
        
        # 4b.3. 126-day Rolling Sharpe Ratio
        print("  Creating 126-day rolling Sharpe ratio chart...")
        rolling_mean_126 = returns.rolling(window).mean()
        rolling_std_126 = returns.rolling(window).std()
        rolling_sharpe_126 = rolling_mean_126 / rolling_std_126 * np.sqrt(252)
        sharpe_126_fig = visualizer.plot_single_metric_timeseries(
            rolling_sharpe_126,
            title="126-Day Rolling Sharpe Ratio",
            y_label="Rolling Sharpe Ratio",
            color='green',
            show_mean=True
        )
        sharpe_126_fig.show()
        
        # 4b.4. 126-day Rolling Max Drawdown
        print("  Creating 126-day rolling max drawdown chart...")
        def rolling_max_drawdown(series, window):
            def max_drawdown(x):
                if len(x) < 2:
                    return 0
                cumulative = (1 + x).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown.min()
            return series.rolling(window).apply(max_drawdown)
        
        rolling_max_dd = rolling_max_drawdown(returns, window)
        max_dd_fig = visualizer.plot_single_metric_timeseries(
            rolling_max_dd * 100,  # Convert to percentage
            title="126-Day Rolling Maximum Drawdown",
            y_label="Rolling Max Drawdown (%)",
            color='orange',
            show_mean=True
        )
        max_dd_fig.show()
        
        # 4b.5. 126-day Rolling Sortino Ratio (if you want it)
        print("  Creating 126-day rolling Sortino ratio chart...")
        def rolling_sortino(series, window, risk_free_rate=0):
            def sortino_ratio(x):
                if len(x) < 2:
                    return np.nan
                excess_returns = x - risk_free_rate/252  # Daily risk-free rate
                downside_returns = excess_returns[excess_returns < 0]
                if len(downside_returns) == 0:
                    return np.nan
                downside_deviation = np.sqrt(np.mean(downside_returns**2))
                if downside_deviation == 0:
                    return np.nan
                return (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)
            return series.rolling(window).apply(sortino_ratio)
        
        rolling_sortino_126 = rolling_sortino(returns, window)
        sortino_fig = visualizer.plot_single_metric_timeseries(
            rolling_sortino_126,
            title="126-Day Rolling Sortino Ratio",
            y_label="Rolling Sortino Ratio",
            color='purple',
            show_mean=True
        )
        sortino_fig.show()
        
        print("  ✓ All 126-day rolling metrics charts created successfully!")
    else:
        print("  ✗ Not enough data for 126-day rolling metrics")
    
    # 5. Correlation Heatmap
    print("5. Creating correlation heatmap...")
    corr_matrix = portfolio.get_correlation_matrix()
    corr_fig = visualizer.plot_correlation_heatmap(
        corr_matrix,
        title="Asset Correlation Matrix"
    )
    corr_fig.show()
    
    # 6. Weights Pie Chart
    print("6. Creating weights pie chart...")
    pie_fig = visualizer.plot_weights_pie(
        weights,
        title="Portfolio Allocation"
    )
    pie_fig.show()
    
    # 6b. Monthly Returns Heatmap Table
    print("6b. Creating monthly returns heatmap table...")
    monthly_table_fig = visualizer.plot_monthly_returns_table(
        portfolio.portfolio_returns,
        title="Monthly Returns Heatmap"
    )
    monthly_table_fig.show()
    
    # 7. Period Returns (Daily, Weekly, Monthly, Yearly)
    print("7. Creating period returns analysis...")
    period_fig = visualizer.plot_period_returns(
        portfolio.portfolio_returns.to_frame(),
        title="Returns by Period"
    )
    period_fig.update_layout(
        width=1000,
        height=800,
        title_font_size=16
    )
    period_fig.show()
    
    # 8. Cumulative returns chart
    print("8. Creating cumulative returns chart...")
    cum_returns_fig = visualizer.plot_cumulative_returns(
        portfolio.portfolio_returns,
        title="Cumulative Returns"
    )
    cum_returns_fig.show()
    
    return portfolio, metrics

portfolio, metrics = example_single_portfolio_all_visualizations()

portfolio
metrics

#%%
# ==============================================================================
# SECTION 2: MULTIPLE PORTFOLIO COMPARISON
# ==============================================================================

def example_multiple_portfolio_comparison():
    """Compare multiple portfolios with different strategies."""
    
    print("\n" + "="*60)
    print("MULTIPLE PORTFOLIO COMPARISON")
    print("="*60)
    
    # Download data once
    downloader = DataDownloader()
    tickers = ['SPY', 'TLT', 'GLD', 'VNQ', 'DBC', 'EEM']
    prices = downloader.download_data(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    returns = downloader.download_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Create different portfolio strategies
    portfolios = {}
    
    # 1. Equal Weight
    portfolios['Equal Weight'] = Portfolio(
        weights={ticker: 1/len(tickers) for ticker in tickers},
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 2. Risk Parity
    optimizer = PortfolioOptimizer(returns)
    rp_weights = optimizer.optimize_risk_parity()
    portfolios['Risk Parity'] = Portfolio(
        weights=rp_weights,
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 3. Max Sharpe
    sharpe_weights = optimizer.optimize_sharpe()
    portfolios['Max Sharpe'] = Portfolio(
        weights=sharpe_weights,
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 4. Min Variance
    minvar_weights = optimizer.optimize_min_variance()
    portfolios['Min Variance'] = Portfolio(
        weights=minvar_weights,
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 5. 60/40 Portfolio (SPY/TLT only)
    portfolios['60/40'] = Portfolio(
        weights={'SPY': 0.6, 'TLT': 0.4, 'GLD': 0, 'VNQ': 0, 'DBC': 0, 'EEM': 0},
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # Calculate values for all portfolios
    portfolio_values = {}
    all_metrics = {}
    
    for name, portfolio in portfolios.items():
        portfolio_values[name] = portfolio.calculate_portfolio_value()
        all_metrics[name] = portfolio.calculate_metrics()
    
    # 1. Compare performance on single chart
    print("\n1. Creating multi-portfolio performance comparison...")
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (name, values) in enumerate(portfolio_values.items()):
        fig.add_trace(go.Scatter(
            x=values.index,
            y=values.values,
            mode='lines',
            name=name,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        title="Portfolio Strategy Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    fig.show()
    
    # 2. Compare cumulative returns
    print("2. Creating cumulative returns comparison...")
    fig = go.Figure()
    
    for i, (name, values) in enumerate(portfolio_values.items()):
        cum_returns = (values / 100000) - 1  # Initial capital = 100000
        fig.add_trace(go.Scatter(
            x=values.index,
            y=cum_returns * 100,
            mode='lines',
            name=name,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified'
    )
    fig.show()
    
    # 3. Compare drawdowns
    print("3. Creating drawdown comparison...")
    fig = go.Figure()
    
    for i, (name, values) in enumerate(portfolio_values.items()):
        cumulative = values / values.iloc[0]
        running_max = cumulative.cummax()
        drawdown = ((cumulative - running_max) / running_max) * 100
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name=name,
            line=dict(color=colors[i], width=1.5)
        ))
    
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )
    fig.show()
    
    # 4. Compare rolling Sharpe ratios
    print("4. Creating rolling Sharpe comparison...")
    fig = go.Figure()
    
    window = 126  # 6-month rolling
    for i, (name, portfolio) in enumerate(portfolios.items()):
        if portfolio.portfolio_returns is None:
            portfolio.calculate_portfolio_returns()
        
        rolling_sharpe = portfolio.portfolio_returns.rolling(window).apply(
            lambda x: PerformanceMetrics.sharpe_ratio(pd.DataFrame(x, columns=['returns'])).iloc[0]
            if len(x) > 5 else np.nan  # Avoid calculation with too few data points
        )
        
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name=name,
            line=dict(color=colors[i], width=1.5)
        ))
    
    fig.update_layout(
        title="126-Day Rolling Sharpe Ratio Comparison",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified'
    )
    fig.show()
    
    return portfolios, all_metrics

portfolios, all_metrics = example_multiple_portfolio_comparison()

#%%
# ==============================================================================
# SECTION 3: PORTFOLIO RETURNS TABLES
# ==============================================================================

def example_portfolio_returns_tables():
    """Create various return tables and dataframes."""
    
    print("\n" + "="*60)
    print("PORTFOLIO RETURNS TABLES")
    print("="*60)
    
    # Create sample portfolio
    portfolio = Portfolio(
        weights={'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3},
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    portfolio.calculate_portfolio_returns()
    
    # 1. Daily returns table
    print("\n1. Daily Returns (first 10 days):")
    daily_returns_df = pd.DataFrame({
        'Date': portfolio.portfolio_returns.index[:10],
        'Daily Return': portfolio.portfolio_returns.values[:10],
        'Return %': (portfolio.portfolio_returns.values[:10] * 100)
    })
    daily_returns_df['Return %'] = daily_returns_df['Return %'].round(2)
    print(daily_returns_df.to_string(index=False))
    
    # 2. Monthly returns table
    print("\n2. Monthly Returns:")
    monthly_returns = portfolio.portfolio_returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly_df = pd.DataFrame({
        'Month': monthly_returns.index.strftime('%Y-%m'),
        'Return': monthly_returns.values,
        'Return %': (monthly_returns.values * 100)
    })
    monthly_df['Return %'] = monthly_df['Return %'].round(2)
    print(monthly_df.to_string(index=False))
    
    # 3. Annual returns table
    print("\n3. Annual Returns:")
    annual_returns = portfolio.portfolio_returns.resample('Y').apply(
        lambda x: (1 + x).prod() - 1
    )
    annual_df = pd.DataFrame({
        'Year': annual_returns.index.year,
        'Return': annual_returns.values,
        'Return %': (annual_returns.values * 100)
    })
    annual_df['Return %'] = annual_df['Return %'].round(2)
    print(annual_df.to_string(index=False))
    
    # 4. Statistical summary table
    print("\n4. Statistical Summary:")
    stats_df = pd.DataFrame({
        'Metric': ['Mean Daily Return', 'Std Dev', 'Skewness', 'Kurtosis', 
                   'Best Day', 'Worst Day', 'Positive Days %'],
        'Value': [
            f"{portfolio.portfolio_returns.mean():.4f}",
            f"{portfolio.portfolio_returns.std():.4f}",
            f"{portfolio.portfolio_returns.skew():.4f}",
            f"{portfolio.portfolio_returns.kurtosis():.4f}",
            f"{portfolio.portfolio_returns.max():.4f}",
            f"{portfolio.portfolio_returns.min():.4f}",
            f"{(portfolio.portfolio_returns > 0).mean()*100:.2f}%"
        ]
    })
    print(stats_df.to_string(index=False))
    
    # 5. Calendar returns heatmap data
    print("\n5. Calendar Returns Matrix (2023):")
    returns_2023 = portfolio.portfolio_returns['2023']
    
    # Create pivot table for calendar view
    returns_2023_df = pd.DataFrame({
        'Date': returns_2023.index,
        'Year': returns_2023.index.year,
        'Month': returns_2023.index.month,
        'Day': returns_2023.index.day,
        'Return': returns_2023.values * 100
    })
    
    # Sample monthly summary
    monthly_summary = returns_2023_df.groupby('Month')['Return'].agg(['mean', 'sum', 'std'])
    monthly_summary.columns = ['Avg Daily %', 'Total %', 'Volatility %']
    monthly_summary = monthly_summary.round(2)
    print(monthly_summary)
    
    return portfolio

portfolio_tables = example_portfolio_returns_tables()

#%%
# ==============================================================================
# SECTION 4: WEIGHTS AND CONSTRAINTS EXAMPLES
# ==============================================================================

def example_weights_and_constraints():
    """Demonstrate various weight and constraint configurations."""
    
    print("\n" + "="*60)
    print("WEIGHTS AND CONSTRAINTS EXAMPLES")
    print("="*60)
    
    # Get data
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
    returns = downloader.download_returns(
        tickers, 
        start_date='2021-01-01', 
        end_date='2023-12-31'
    )
    
    optimizer = PortfolioOptimizer(returns)
    
    # 1. Basic weight bounds
    print("\n1. Basic Weight Bounds (10% min, 30% max):")
    weights_bounded = optimizer.optimize_sharpe(
        weight_bounds=(0.10, 0.30)
    )
    for ticker, weight in weights_bounded.items():
        print(f"  {ticker}: {weight:.1%}")
    
    # 2. Long-only constraint (no shorting)
    print("\n2. Long-only portfolio (no negative weights):")
    weights_long = optimizer.optimize_min_variance(
        weight_bounds=(0, 1)
    )
    for ticker, weight in sorted(weights_long.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {ticker}: {weight:.1%}")
    
    # 3. Sector constraints
    print("\n3. Sector Constraints:")
    
    # Define sectors
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    other_stocks = ['AMZN', 'TSLA', 'JPM']
    
    sector_mapping = {
        'tech': [i for i, t in enumerate(tickers) if t in tech_stocks],
        'other': [i for i, t in enumerate(tickers) if t in other_stocks]
    }
    
    sector_limits = {
        'tech': (0.4, 0.7),   # Tech must be 40-70%
        'other': (0.2, 0.5)   # Others must be 20-50%
    }
    
    constraints = ConstraintBuilder.sector_constraint(sector_mapping, sector_limits)
    
    # Apply constraints (simplified for example)
    print("  Tech sector: 40-70% allocation")
    print("  Other sector: 20-50% allocation")
    
    # 4. Cardinality constraints (number of assets)
    print("\n4. Cardinality Constraints (3-5 assets):")
    
    # Optimize then filter to top N assets
    all_weights = optimizer.optimize_sharpe()
    sorted_weights = sorted(all_weights.items(), key=lambda x: x[1], reverse=True)
    
    # Keep only top 4 assets
    top_n = 4
    concentrated_weights = {t: w for t, w in sorted_weights[:top_n]}
    
    # Renormalize
    total = sum(concentrated_weights.values())
    concentrated_weights = {t: w/total for t, w in concentrated_weights.items()}
    
    print(f"  Portfolio concentrated in {top_n} assets:")
    for ticker, weight in concentrated_weights.items():
        print(f"    {ticker}: {weight:.1%}")
    
    # 5. Maximum position size
    print("\n5. Maximum Position Size (no single stock > 25%):")
    weights_capped = optimizer.optimize_sharpe(
        weight_bounds=(0, 0.25)
    )
    for ticker, weight in sorted(weights_capped.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker}: {weight:.1%}")
    
    # 6. Equal risk contribution (risk parity)
    print("\n6. Risk Parity (equal risk contribution):")
    weights_rp = optimizer.optimize_risk_parity()
    
    # Calculate risk contributions
    cov_matrix = returns.cov()
    weights_array = np.array([weights_rp[t] for t in tickers])
    risk_contrib = PortfolioHelpers.calculate_risk_contribution(weights_array, cov_matrix.values)
    
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: Weight={weights_rp[ticker]:.1%}, Risk Contrib={risk_contrib[i]:.1%}")
    
    return concentrated_weights

concentrated_weights = example_weights_and_constraints()

#%%
# ==============================================================================
# SECTION 5: LEVERAGE AND FEES EXAMPLES
# ==============================================================================

def example_leverage_and_fees():
    """Demonstrate leverage and fee impacts."""
    
    print("\n" + "="*60)
    print("LEVERAGE AND FEES EXAMPLES")
    print("="*60)
    
    # Base portfolio
    base_weights = {'SPY': 0.6, 'TLT': 0.4}
    
    # Test different scenarios
    scenarios = [
        ('No Leverage, No Fees', 1.0, 0.0000, None),
        ('No Leverage, With Fees', 1.0, 0.0010, 'M'),
        ('1.5x Leverage, No Fees', 1.5, 0.0000, None),
        ('1.5x Leverage, With Fees', 1.5, 0.0010, 'M'),
        ('2x Leverage, High Fees', 2.0, 0.0025, 'W'),
    ]
    
    results = []
    downloader = DataDownloader()
    prices = downloader.download_data(
        list(base_weights.keys()),
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    for scenario_name, leverage, fees, rebalance in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Leverage: {leverage}x")
        print(f"  Transaction Cost: {fees:.2%}")
        print(f"  Rebalancing: {rebalance or 'None'}")
        
        # Create backtester
        backtester = Backtester(
            data=prices,
            initial_capital=100000,
            commission=fees,
            slippage=0.0005
        )
        
        # Apply leverage to weights
        leveraged_weights = {k: v * leverage for k, v in base_weights.items()}
        
        # Backtest
        results_df = backtester.backtest_fixed_weights(
            weights=leveraged_weights,
            rebalance_frequency=rebalance,
            leverage=leverage
        )
        
        metrics = backtester.calculate_metrics()
        
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Total Fees Paid: ${metrics['total_commission']:.2f}")
        
        results.append({
            'Scenario': scenario_name,
            'Leverage': leverage,
            'Fees': fees,
            'Return': metrics['total_return'],
            'Sharpe': metrics['sharpe_ratio'],
            'MaxDD': metrics['max_drawdown'],
            'Total Fees': metrics['total_commission']
        })
    
    # Create comparison table
    print("\n" + "-"*60)
    print("LEVERAGE AND FEES IMPACT SUMMARY")
    comparison_df = pd.DataFrame(results)
    comparison_df['Return'] = (comparison_df['Return'] * 100).round(2)
    comparison_df['MaxDD'] = (comparison_df['MaxDD'] * 100).round(2)
    comparison_df['Sharpe'] = comparison_df['Sharpe'].round(3)
    comparison_df['Total Fees'] = comparison_df['Total Fees'].round(2)
    print(comparison_df.to_string(index=False))
    
    # Visualize impact
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Returns', 'Sharpe Ratio', 'Max Drawdown', 'Total Fees']
    )
    
    x = [s[0] for s in scenarios]
    
    fig.add_trace(go.Bar(x=x, y=comparison_df['Return'], name='Return %'), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=comparison_df['Sharpe'], name='Sharpe'), row=1, col=2)
    fig.add_trace(go.Bar(x=x, y=comparison_df['MaxDD'], name='Max DD %'), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=comparison_df['Total Fees'], name='Fees $'), row=2, col=2)
    
    fig.update_layout(height=800, title="Leverage and Fees Impact Analysis", showlegend=False)
    fig.show()
    
    return comparison_df

leverage_comparison = example_leverage_and_fees()

#%%
# ==============================================================================
# SECTION 6: EFFICIENT FRONTIER WITH CONSTRAINTS
# ==============================================================================

def example_efficient_frontier_with_constraints():
    """Show efficient frontier with various constraints."""
    
    print("\n" + "="*60)
    print("EFFICIENT FRONTIER WITH CONSTRAINTS")
    print("="*60)
    
    # Get data
    downloader = DataDownloader()
    tickers = ['SPY', 'TLT', 'GLD', 'VNQ', 'EEM']
    returns = downloader.download_returns(
        tickers, 
        start_date='2019-01-01', 
        end_date='2023-12-31'
    )
    
    # Create multiple frontiers with different constraints
    optimizer = PortfolioOptimizer(returns)
    
    # 1. Unconstrained frontier
    frontier_unconstrained = optimizer.efficient_frontier(
        n_portfolios=30,
        weight_bounds=(0, 1)
    )
    
    # 2. Constrained frontier (max 40% per asset)
    frontier_constrained = optimizer.efficient_frontier(
        n_portfolios=30,
        weight_bounds=(0, 0.4)
    )
    
    # 3. Long-short frontier
    frontier_long_short = optimizer.efficient_frontier(
        n_portfolios=30,
        weight_bounds=(-0.2, 1.2)  # Allow 20% shorting
    )
    
    # Visualize all frontiers
    fig = go.Figure()
    
    # Add unconstrained
    fig.add_trace(go.Scatter(
        x=frontier_unconstrained['volatility'] * 100,
        y=frontier_unconstrained['return'] * 100,
        mode='lines+markers',
        name='Unconstrained',
        line=dict(color='blue', width=2)
    ))
    
    # Add constrained
    fig.add_trace(go.Scatter(
        x=frontier_constrained['volatility'] * 100,
        y=frontier_constrained['return'] * 100,
        mode='lines+markers',
        name='Max 40% per asset',
        line=dict(color='red', width=2)
    ))
    
    # Add long-short
    fig.add_trace(go.Scatter(
        x=frontier_long_short['volatility'] * 100,
        y=frontier_long_short['return'] * 100,
        mode='lines+markers',
        name='Long-Short (±20%)',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Mark special portfolios
    # Max Sharpe from unconstrained
    max_sharpe_idx = frontier_unconstrained['sharpe_ratio'].idxmax()
    fig.add_trace(go.Scatter(
        x=[frontier_unconstrained.loc[max_sharpe_idx, 'volatility'] * 100],
        y=[frontier_unconstrained.loc[max_sharpe_idx, 'return'] * 100],
        mode='markers',
        name='Max Sharpe',
        marker=dict(color='gold', size=15, symbol='star')
    ))
    
    # Min variance from constrained
    min_var_idx = frontier_constrained['volatility'].idxmin()
    fig.add_trace(go.Scatter(
        x=[frontier_constrained.loc[min_var_idx, 'volatility'] * 100],
        y=[frontier_constrained.loc[min_var_idx, 'return'] * 100],
        mode='markers',
        name='Min Variance',
        marker=dict(color='purple', size=12, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Efficient Frontiers with Different Constraints",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        hovermode='closest',
        width=900,
        height=600
    )
    fig.show()
    
    # Print optimal portfolios
    print("\nOptimal Portfolio Weights:")
    
    print("\n1. Max Sharpe (Unconstrained):")
    for col in [c for c in frontier_unconstrained.columns if c.startswith('weight_')]:
        ticker = col.replace('weight_', '')
        weight = frontier_unconstrained.loc[max_sharpe_idx, col]
        if abs(weight) > 0.01:
            print(f"  {ticker}: {weight:.1%}")
    
    print("\n2. Min Variance (Constrained):")
    for col in [c for c in frontier_constrained.columns if c.startswith('weight_')]:
        ticker = col.replace('weight_', '')
        weight = frontier_constrained.loc[min_var_idx, col]
        if abs(weight) > 0.01:
            print(f"  {ticker}: {weight:.1%}")
    
    return frontier_unconstrained, frontier_constrained

frontier_unc, frontier_con = example_efficient_frontier_with_constraints()
#%%
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" PYANDHOLD - COMPREHENSIVE EXAMPLES ".center(80))
    print("="*80)
    
    # Run all examples
    try:
        # Section 1: Single Portfolio Visualizations
        portfolio, metrics = example_single_portfolio_all_visualizations()
        
        # Section 2: Multiple Portfolio Comparison
        portfolios, all_metrics = example_multiple_portfolio_comparison()
        
        # Section 3: Returns Tables
        portfolio_tables = example_portfolio_returns_tables()
        
        # Section 4: Weights and Constraints
        concentrated_weights = example_weights_and_constraints()
        
        # Section 5: Leverage and Fees
        leverage_comparison = example_leverage_and_fees()
        
        # Section 6: Efficient Frontier with Constraints
        frontier_unc, frontier_con = example_efficient_frontier_with_constraints()
        
        print("\n" + "="*80)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY ".center(80))
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()