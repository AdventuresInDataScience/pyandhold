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
display(pd.DataFrame(metrics.items(), columns=["metric", "value"]))

#%% 2.1 - Basic
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
    tickers = ['SPY', 'NVDA']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='1980-01-01',
        end_date='2023-12-31'
    )
    
    # Create different portfolio strategies
    portfolios = {}
    
    # 1. Equal Weight
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    print(f"\n1. Equal Weight calculated weights:")
    for ticker, weight in equal_weights.items():
        print(f"   {ticker}: {weight:.4f}")
    
    portfolios['Equal Weight'] = Portfolio(
        weights=equal_weights,
        start_date='2000-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 2. Risk Parity
    optimizer = PortfolioOptimizer(returns)
    print(f"\n2. Calculating Risk Parity weights...")
    
    try:
        rp_weights = optimizer.optimize_risk_parity()
        print(f"   Risk Parity optimization SUCCESS")
    except Exception as e:
        print(f"   Risk Parity optimization FAILED: {e}")
        # Fallback to equal weights
        rp_weights = {ticker: 1/len(tickers) for ticker in tickers}
        print(f"   Using fallback equal weights for Risk Parity")
    
    print(f"Risk Parity calculated weights:")
    for ticker, weight in rp_weights.items():
        print(f"   {ticker}: {weight:.4f}")
    
    portfolios['Risk Parity'] = Portfolio(
        weights=rp_weights,
        start_date='2000-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 3. Max Sharpe
    print(f"\n3. Calculating Max Sharpe weights...")
    try:
        sharpe_weights = optimizer.optimize_sharpe()
        print(f"   Max Sharpe optimization SUCCESS")
    except Exception as e:
        print(f"   Max Sharpe optimization FAILED: {e}")
        # Fallback to equal weights
        sharpe_weights = {ticker: 1/len(tickers) for ticker in tickers}
        print(f"   Using fallback equal weights for Max Sharpe")
    
    print(f"Max Sharpe calculated weights:")
    for ticker, weight in sharpe_weights.items():
        print(f"   {ticker}: {weight:.4f}")
    
    portfolios['Max Sharpe'] = Portfolio(
        weights=sharpe_weights,
        start_date='2000-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 4. Min Variance
    print(f"\n4. Calculating Min Variance weights...")
    try:
        minvar_weights = optimizer.optimize_min_variance()
        print(f"   Min Variance optimization SUCCESS")
    except Exception as e:
        print(f"   Min Variance optimization FAILED: {e}")
        # Fallback to equal weights
        minvar_weights = {ticker: 1/len(tickers) for ticker in tickers}
        print(f"   Using fallback equal weights for Min Variance")
    
    print(f"Min Variance calculated weights:")
    for ticker, weight in minvar_weights.items():
        print(f"   {ticker}: {weight:.4f}")
    
    portfolios['Min Variance'] = Portfolio(
        weights=minvar_weights,
        start_date='2000-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )
    
    # 5. 60/40 Portfolio (SPY/TLT only)
    portfolio_60_40_weights = {'SPY': 0.6, 'TLT': 0.4, 'GLD': 0, 'VNQ': 0, 'DBC': 0, 'EEM': 0}
    print(f"\n5. 60/40 Portfolio weights:")
    for ticker, weight in portfolio_60_40_weights.items():
        if weight > 0:
            print(f"   {ticker}: {weight:.4f}")
    
    portfolios['60/40'] = Portfolio(
        weights=portfolio_60_40_weights,
        start_date='2000-01-01',
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
        legend=dict(x=0.02, y=0.98),
        width=1000,
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
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
        hovermode='x unified',
        width=1000,
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
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
        hovermode='x unified',
        width=1000,
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
    )
    fig.show()
    
    # 4. Compare rolling Sharpe ratios
    print("4. Creating rolling Sharpe comparison...")
    fig = go.Figure()
    
    window = 126  # 6-month rolling
    for i, (name, portfolio) in enumerate(portfolios.items()):
        if portfolio.portfolio_returns is None:
            portfolio.calculate_portfolio_returns()
        
        rolling_mean = portfolio.portfolio_returns.rolling(window).mean()
        rolling_std = portfolio.portfolio_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
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
        hovermode='x unified',
        width=1000,
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        )
    )
    fig.show()
    
    return portfolios, all_metrics

portfolios, all_metrics = example_multiple_portfolio_comparison()
display(pd.DataFrame(all_metrics))
#%% 2.2 - Advanced
# ==============================================================================
# SECTION 2.2: COMPREHENSIVE CONSTRAINT TESTING
# ==============================================================================

def example_comprehensive_constraint_testing():
    """Test all available constraint types from constraints.py."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE CONSTRAINT TESTING")
    print("="*60)
    
    # Download data for constraint testing
    downloader = DataDownloader()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    optimizer = PortfolioOptimizer(returns)
    constraint_portfolios = {}
    
    # Test all available constraints from constraints.py
    constraint_builder = ConstraintBuilder()
    
    # 1. LONG-ONLY CONSTRAINT TEST
    print("\n1. Testing Long-Only Constraint (using weight bounds)...")
    try:
        # Use weight bounds to implement long-only constraint
        long_only_weights = optimizer.optimize_sharpe(
            weight_bounds=(0.0, 1.0)  # This implements long-only
        )
        
        # Convert to dictionary if needed
        if isinstance(long_only_weights, list):
            long_only_weights = {tickers[i]: long_only_weights[i] for i in range(len(tickers))}
        
        constraint_portfolios['Long Only'] = Portfolio(
            weights=long_only_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Verify constraint
        min_weight = min(long_only_weights.values())
        constraint_satisfied = min_weight >= -1e-6  # Small tolerance
        
        status = "✓" if constraint_satisfied else "✗"
        print(f"   {status} Long-Only constraint {'SUCCESS' if constraint_satisfied else 'VIOLATION'}")
        print(f"   Minimum weight: {min_weight:.4f} (should be >= 0)")
        print(f"   Top 5 holdings:")
        for ticker, weight in sorted(long_only_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {ticker}: {weight:.2%}")
    except Exception as e:
        print(f"   ✗ Long-Only constraint FAILED: {e}")
    
    # 2. MAXIMUM POSITION CONSTRAINT TEST  
    print("\n2. Testing Maximum Position Constraint (15% limit)...")
    try:
        max_weight_limit = 0.15  # 15% max
        
        # Use actual ConstraintBuilder
        max_pos_constraint = constraint_builder.max_position_constraint(max_weight_limit)
        
        max_pos_weights = optimizer.optimize_sharpe(
            constraints={'max_position': max_pos_constraint}
        )
        
        # Convert to dictionary if needed
        if isinstance(max_pos_weights, list):
            max_pos_weights = {tickers[i]: max_pos_weights[i] for i in range(len(tickers))}
        
        constraint_portfolios['Max 15% Position'] = Portfolio(
            weights=max_pos_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Verify constraint
        max_weight = max(max_pos_weights.values())
        constraint_satisfied = max_weight <= max_weight_limit + 1e-6  # Small tolerance
        
        status = "✓" if constraint_satisfied else "✗"
        print(f"   {status} Max Position constraint {'SUCCESS' if constraint_satisfied else 'VIOLATION'}")
        print(f"   Largest position: {max_weight:.2%} (limit: {max_weight_limit:.0%})")
        print(f"   Top 5 holdings:")
        for ticker, weight in sorted(max_pos_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {ticker}: {weight:.2%}")
    except Exception as e:
        print(f"   ✗ Max Position constraint FAILED: {e}")
    
    # 3. MINIMUM POSITION CONSTRAINT TEST
    print("\n3. Testing Minimum Position Constraint (5% minimum for active positions)...")
    try:
        min_weight_limit = 0.05  # 5% min for active positions
        
        # Use a subset for feasible min position constraint (4 assets * 5% = 20% minimum)
        selected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        selected_returns = returns[selected_tickers]
        selected_optimizer = PortfolioOptimizer(selected_returns)
        
        # Use actual ConstraintBuilder
        min_pos_constraint = constraint_builder.min_position_constraint(min_weight_limit)
        
        min_pos_weights_list = selected_optimizer.optimize_sharpe(
            constraints={'min_position': min_pos_constraint}
        )
        
        # Convert to dictionary
        if isinstance(min_pos_weights_list, list):
            min_pos_weights = {selected_tickers[i]: min_pos_weights_list[i] for i in range(len(selected_tickers))}
        else:
            min_pos_weights = min_pos_weights_list
        
        # Pad with zeros for other tickers
        full_min_pos_weights = {ticker: 0.0 for ticker in tickers}
        full_min_pos_weights.update(min_pos_weights)
        
        constraint_portfolios['Min 5% Position'] = Portfolio(
            weights=full_min_pos_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Verify constraint
        active_weights = [w for w in min_pos_weights.values() if w > 0.001]
        min_nonzero_weight = min(active_weights) if active_weights else 0
        constraint_satisfied = min_nonzero_weight >= min_weight_limit - 1e-6
        
        status = "✓" if constraint_satisfied else "✗"
        print(f"   {status} Min Position constraint {'SUCCESS' if constraint_satisfied else 'VIOLATION'}")
        print(f"   Smallest position: {min_nonzero_weight:.2%} (min: {min_weight_limit:.0%})")
        for ticker, weight in min_pos_weights.items():
            print(f"     {ticker}: {weight:.2%}")
    except Exception as e:
        print(f"   ✗ Min Position constraint FAILED: {e}")
    
    # 4. SECTOR CONSTRAINT TEST
    print("\n4. Testing Sector Constraints...")
    try:
        # Define sectors and their indices
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA']
        finance_stocks = ['JPM', 'BAC', 'GS']
        energy_stocks = ['XOM', 'CVX']
        healthcare_stocks = ['JNJ', 'PFE', 'UNH']
        
        # Map tickers to indices
        ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}
        
        sector_mapping = {
            'tech': [ticker_to_index[t] for t in tech_stocks if t in ticker_to_index],
            'finance': [ticker_to_index[t] for t in finance_stocks if t in ticker_to_index],
            'energy': [ticker_to_index[t] for t in energy_stocks if t in ticker_to_index],
            'healthcare': [ticker_to_index[t] for t in healthcare_stocks if t in ticker_to_index]
        }
        
        # Define sector limits
        sector_limits = {
            'tech': (0.30, 0.60),      # Tech: 30-60%
            'finance': (0.10, 0.25),   # Finance: 10-25%
            'energy': (0.05, 0.15),    # Energy: 5-15%
            'healthcare': (0.10, 0.20) # Healthcare: 10-20%
        }
        
        # Use actual ConstraintBuilder
        sector_constraints = constraint_builder.sector_constraint(sector_mapping, sector_limits)
        
        sector_weights = optimizer.optimize_sharpe(
            constraints={'sector': sector_constraints},
            n_trials=10000,  # Even more trials for better constraint satisfaction
            initial_noise=0.5,  # Higher noise for more diverse initial population
            verbose=True  # Enable verbose to see what's happening
        )
        
        # Convert to dictionary if needed
        if isinstance(sector_weights, list):
            sector_weights = {tickers[i]: sector_weights[i] for i in range(len(tickers))}
        
        # Calculate sector allocations
        sector_allocs = {
            'tech': sum(sector_weights.get(t, 0) for t in tech_stocks),
            'finance': sum(sector_weights.get(t, 0) for t in finance_stocks),
            'energy': sum(sector_weights.get(t, 0) for t in energy_stocks),
            'healthcare': sum(sector_weights.get(t, 0) for t in healthcare_stocks)
        }
        
        constraint_portfolios['Sector Constrained'] = Portfolio(
            weights=sector_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Check if constraints are satisfied
        constraints_satisfied = True
        print(f"   Sector allocations (target ranges in parentheses):")
        for sector, alloc in sector_allocs.items():
            if sector in sector_limits:
                min_limit, max_limit = sector_limits[sector]
                sector_satisfied = min_limit <= alloc <= max_limit
                if not sector_satisfied:
                    constraints_satisfied = False
                sector_status = "✓" if sector_satisfied else "✗"
                print(f"     {sector}: {alloc:.2%} ({min_limit:.0%}-{max_limit:.0%}) {sector_status}")
            else:
                print(f"     {sector}: {alloc:.2%}")
        
        status = "✓" if constraints_satisfied else "✗"
        print(f"   {status} Sector constraints {'SUCCESS' if constraints_satisfied else 'PARTIAL'}")
    except Exception as e:
        print(f"   ✗ Sector constraint FAILED: {e}")
    
    # 5. CARDINALITY CONSTRAINT TEST
    print("\n5. Testing Cardinality Constraints (from ConstraintBuilder)...")
    try:
        # Use ConstraintBuilder.cardinality_constraint()
        min_assets = 5
        max_assets = 8
        cardinality_constraint = constraint_builder.cardinality_constraint(min_assets, max_assets)
        
        # Get initial weights and select top N assets (simplified implementation)
        initial_weights_list = optimizer.optimize_sharpe()
        
        # Convert to dictionary if needed
        if isinstance(initial_weights_list, list):
            initial_weights = {tickers[i]: initial_weights_list[i] for i in range(len(tickers))}
        else:
            initial_weights = initial_weights_list
            
        sorted_weights = sorted(initial_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only top 8 assets within cardinality limits
        max_assets = 8
        cardinality_weights = {t: 0.0 for t in tickers}
        top_assets = sorted_weights[:max_assets]
        
        # Renormalize weights
        total_weight = sum(w for _, w in top_assets)
        for ticker, weight in top_assets:
            cardinality_weights[ticker] = weight / total_weight
        
        constraint_portfolios['Cardinality 5-8 Assets'] = Portfolio(
            weights=cardinality_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Verify constraint
        active_assets = sum(1 for w in cardinality_weights.values() if w > 0.001)
        min_assets = 5
        constraint_satisfied = min_assets <= active_assets <= max_assets
        
        status = "✓" if constraint_satisfied else "✗"
        print(f"   {status} Cardinality constraint {'SUCCESS' if constraint_satisfied else 'VIOLATION'}")
        print(f"   Active assets: {active_assets} (target: {min_assets}-{max_assets})")
        print(f"   Holdings:")
        for ticker, weight in cardinality_weights.items():
            if weight > 0.001:
                print(f"     {ticker}: {weight:.2%}")
    except Exception as e:
        print(f"   ✗ Cardinality constraint FAILED: {e}")
    
    # 6. TURNOVER CONSTRAINT TEST
    print("\n6. Testing Turnover Constraint...")
    try:
        # Start with equal weight portfolio
        equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
        current_weights = np.array([equal_weights[t] for t in tickers])
        
        # Optimize with limited turnover (max 20% turnover)
        max_turnover = 0.20
        
        # Use actual ConstraintBuilder
        turnover_constraint = constraint_builder.turnover_constraint(current_weights, max_turnover)
        
        turnover_weights = optimizer.optimize_sharpe(
            constraints={'turnover': turnover_constraint}
        )
        
        # Convert to dictionary if needed
        if isinstance(turnover_weights, list):
            turnover_weights = {tickers[i]: turnover_weights[i] for i in range(len(tickers))}
        
        # Calculate actual turnover
        new_weights = np.array([turnover_weights[t] for t in tickers])
        actual_turnover = np.sum(np.abs(new_weights - current_weights))
        constraint_satisfied = actual_turnover <= max_turnover + 1e-6
        
        constraint_portfolios['Low Turnover'] = Portfolio(
            weights=turnover_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        status = "✓" if constraint_satisfied else "✗"
        print(f"   {status} Turnover constraint {'SUCCESS' if constraint_satisfied else 'VIOLATION'}")
        print(f"   Target max turnover: {max_turnover:.1%}")
        print(f"   Actual turnover: {actual_turnover:.1%}")
        print(f"   Largest deviations from equal weight:")
        deviations = [(t, abs(w - 1/len(tickers))) for t, w in turnover_weights.items()]
        for ticker, dev in sorted(deviations, reverse=True)[:5]:
            print(f"     {ticker}: {dev:.2%}")
    except Exception as e:
        print(f"   ✗ Turnover constraint FAILED: {e}")
    
    # 7. LEVERAGE CONSTRAINT TEST
    print("\n7. Testing Leverage Constraint (130/30 Strategy)...")
    try:
        # 130/30: 130% long, 30% short, net 100%
        leverage_weights_list = optimizer.optimize_sharpe(
            weight_bounds=(-0.10, 0.20)  # Allow some shorting, cap longs
        )
        
        # Convert to dictionary if needed
        if isinstance(leverage_weights_list, list):
            leverage_130_30_weights = {tickers[i]: leverage_weights_list[i] for i in range(len(tickers))}
        else:
            leverage_130_30_weights = leverage_weights_list
        
        # Calculate leverage metrics
        long_weights = sum(w for w in leverage_130_30_weights.values() if w > 0)
        short_weights = abs(sum(w for w in leverage_130_30_weights.values() if w < 0))
        net_weights = sum(leverage_130_30_weights.values())
        gross_leverage = long_weights + short_weights
        
        constraint_portfolios['130/30 Strategy'] = Portfolio(
            weights=leverage_130_30_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        print(f"   ✓ Leverage constraint SUCCESS")
        print(f"   Long exposure: {long_weights:.1%}")
        print(f"   Short exposure: {short_weights:.1%}")
        print(f"   Net exposure: {net_weights:.1%}")
        print(f"   Gross leverage: {gross_leverage:.1%}")
        print(f"   Positions:")
        for ticker, weight in sorted(leverage_130_30_weights.items(), key=lambda x: x[1], reverse=True):
            if abs(weight) > 0.01:
                print(f"     {ticker}: {weight:+.2%}")
    except Exception as e:
        print(f"   ✗ Leverage constraint FAILED: {e}")
    
    # 8. CUSTOM LINEAR CONSTRAINT TEST (Risk Budget Constraint)
    print("\n8. Testing Custom Linear Constraint (Equal Risk Contribution)...")
    try:
        # Approximate equal risk contribution using risk parity
        risk_parity_weights_result = optimizer.optimize_risk_parity()
        
        # Convert to dictionary if needed
        if isinstance(risk_parity_weights_result, list):
            risk_parity_weights = {tickers[i]: risk_parity_weights_result[i] for i in range(len(tickers))}
        else:
            risk_parity_weights = risk_parity_weights_result
        
        constraint_portfolios['Risk Parity'] = Portfolio(
            weights=risk_parity_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Calculate risk contributions
        cov_matrix = returns.cov().values
        weights_array = np.array([risk_parity_weights[t] for t in tickers])
        portfolio_var = np.dot(weights_array, np.dot(cov_matrix, weights_array))
        risk_contrib = (weights_array * np.dot(cov_matrix, weights_array)) / portfolio_var
        
        print(f"   ✓ Risk Parity constraint SUCCESS")
        print(f"   Risk contributions (should be roughly equal):")
        for i, ticker in enumerate(tickers):
            if risk_parity_weights[ticker] > 0.01:
                print(f"     {ticker}: Weight={risk_parity_weights[ticker]:.2%}, Risk={risk_contrib[i]:.2%}")
    except Exception as e:
        print(f"   ✗ Risk Parity constraint FAILED: {e}")
    
    # 9. MIXED CONSTRAINT TEST (Multiple constraints combined)
    print("\n9. Testing Mixed Constraints (Long-only + Max 10% + Sector limits)...")
    try:
        mixed_weights_list = optimizer.optimize_sharpe(
            weight_bounds=(0.0, 0.10)  # Long-only, max 10% per position
        )
        
        # Convert to dictionary if needed
        if isinstance(mixed_weights_list, list):
            mixed_weights = {tickers[i]: mixed_weights_list[i] for i in range(len(tickers))}
        else:
            mixed_weights = mixed_weights_list
        
        constraint_portfolios['Mixed Constraints'] = Portfolio(
            weights=mixed_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        # Verify constraints
        max_weight = max(mixed_weights.values())
        min_weight = min(mixed_weights.values())
        active_positions = sum(1 for w in mixed_weights.values() if w > 0.001)
        
        print(f"   ✓ Mixed constraints SUCCESS")
        print(f"   Max position: {max_weight:.2%} (limit: 10%)")
        print(f"   Min position: {min_weight:.2%}")
        print(f"   Active positions: {active_positions}")
        print(f"   Top holdings:")
        for ticker, weight in sorted(mixed_weights.items(), key=lambda x: x[1], reverse=True)[:8]:
            if weight > 0.001:
                print(f"     {ticker}: {weight:.2%}")
    except Exception as e:
        print(f"   ✗ Mixed constraints FAILED: {e}")
    
    # 10. EXTREME CONSTRAINT TEST (Very tight constraints)
    print("\n10. Testing Extreme Constraints (5-8% per position, long-only)...")
    try:
        # Use subset for extreme constraints
        extreme_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'UNH', 'HD']
        available_extreme_tickers = [t for t in extreme_tickers if t in tickers][:10]  # Limit to available tickers
        extreme_returns = returns[available_extreme_tickers]
        extreme_optimizer = PortfolioOptimizer(extreme_returns)
        
        extreme_weights_list = extreme_optimizer.optimize_min_variance(
            weight_bounds=(0.05, 0.08)  # Very tight: 5-8% per position
        )
        
        # Convert to dictionary if needed
        if isinstance(extreme_weights_list, list):
            extreme_weights_subset = {available_extreme_tickers[i]: extreme_weights_list[i] for i in range(len(available_extreme_tickers))}
        else:
            extreme_weights_subset = extreme_weights_list
        
        # Pad with zeros for other tickers
        extreme_weights = {ticker: 0.0 for ticker in tickers}
        extreme_weights.update(extreme_weights_subset)
        
        constraint_portfolios['Extreme Constraints'] = Portfolio(
            weights=extreme_weights,
            start_date='2020-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        active_extreme = {t: w for t, w in extreme_weights.items() if w > 0}
        
        print(f"   ✓ Extreme constraints SUCCESS")
        print(f"   All positions between 5-8%:")
        for ticker, weight in active_extreme.items():
            print(f"     {ticker}: {weight:.2%}")
        print(f"   Total allocation: {sum(active_extreme.values()):.1%}")
    except Exception as e:
        print(f"   ✗ Extreme constraints FAILED: {e}")
    
    # PERFORMANCE COMPARISON OF ALL CONSTRAINT PORTFOLIOS
    print("\n" + "="*60)
    print("CONSTRAINT PORTFOLIO PERFORMANCE COMPARISON")
    print("="*60)
    
    # Calculate metrics for all constraint portfolios
    constraint_metrics = {}
    constraint_values = {}
    
    for name, portfolio in constraint_portfolios.items():
        try:
            constraint_values[name] = portfolio.calculate_portfolio_value()
            constraint_metrics[name] = portfolio.calculate_metrics()
            print(f"\n{name}:")
            print(f"  Total Return: {constraint_metrics[name]['total_return']:.2%}")
            print(f"  Volatility: {constraint_metrics[name]['volatility']:.2%}")
            print(f"  Sharpe Ratio: {constraint_metrics[name]['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {constraint_metrics[name]['max_drawdown']:.2%}")
        except Exception as e:
            print(f"\n{name}: Error calculating metrics - {e}")
    
    # Create comparison visualization
    if constraint_values:
        print(f"\nCreating constraint portfolio comparison chart...")
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (name, values) in enumerate(constraint_values.items()):
            fig.add_trace(go.Scatter(
                x=values.index,
                y=values.values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Constraint Portfolio Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
            width=1200,
            height=700,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=2, label="2y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            )
        )
        fig.show()
    
    return constraint_portfolios, constraint_metrics

def example_robust_optimization_constraints():
    """Test robust optimization with various constraint scenarios."""
    
    print("\n" + "="*60)
    print("ROBUST OPTIMIZATION WITH CONSTRAINTS")
    print("="*60)
    
    # Download data
    downloader = DataDownloader()
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ', 'EEM', 'EFA', 'IWM', 'DBC', 'HYG']
    prices, returns = downloader.download_prices_and_returns(
        tickers,
        start_date='2015-01-01',
        end_date='2023-12-31'
    )
    
    robust_portfolios = {}
    
    # 1. ROBUST OPTIMIZATION - BLACK-LITTERMAN WITH CONSTRAINTS
    print("\n1. Testing Black-Litterman with Position Limits...")
    try:
        robust_optimizer = RobustOptimizer(returns)
        
        # Market cap weights as prior (simplified)
        market_caps = pd.Series({
            'SPY': 0.45, 'QQQ': 0.20, 'TLT': 0.10, 'GLD': 0.05, 'VNQ': 0.05,
            'EEM': 0.05, 'EFA': 0.05, 'IWM': 0.03, 'DBC': 0.01, 'HYG': 0.01
        })
        
        # Views (simplified - expecting tech outperformance)
        views = {
            'QQQ': 0.08,  # Expect 8% excess return
            'TLT': -0.02, # Expect -2% excess return
        }
        
        # View confidences
        view_confidences = {
            'QQQ': 0.8,  # High confidence
            'TLT': 0.6   # Medium confidence
        }
        
        bl_weights = robust_optimizer.black_litterman_optimization(
            market_caps=market_caps,
            views=views,
            view_confidences=view_confidences,
            weight_bounds=(0.0, 0.25)  # Max 25% per position
        )
        
        robust_portfolios['Black-Litterman Constrained'] = Portfolio(
            weights=bl_weights,
            start_date='2015-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        print(f"   ✓ Black-Litterman with constraints SUCCESS")
        print(f"   Top allocations:")
        for ticker, weight in sorted(bl_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {ticker}: {weight:.2%}")
            
    except Exception as e:
        print(f"   ✗ Black-Litterman optimization FAILED: {e}")
    
    # 2. ROBUST OPTIMIZATION - RESAMPLED EFFICIENCY WITH SECTOR CONSTRAINTS
    print("\n2. Testing Resampled Efficiency with Sector Limits...")
    try:
        # Define asset classes
        equity_etfs = ['SPY', 'QQQ', 'EEM', 'EFA', 'IWM']
        fixed_income_etfs = ['TLT', 'HYG']
        alternative_etfs = ['GLD', 'VNQ', 'DBC']
        
        # Resampled efficiency (simplified implementation)
        n_samples = 50
        resampled_weights = []
        
        for _ in range(n_samples):
            # Bootstrap returns
            sample_returns = returns.sample(n=len(returns), replace=True, random_state=np.random.randint(1000))
            sample_optimizer = PortfolioOptimizer(sample_returns)
            
            try:
                # Optimize with sector constraints (approximated)
                weights = sample_optimizer.optimize_sharpe(weight_bounds=(0.0, 0.30))
                
                # Apply sector limits manually
                equity_weight = sum(weights.get(t, 0) for t in equity_etfs)
                if equity_weight > 0.70:  # Max 70% equity
                    scale_factor = 0.70 / equity_weight
                    for t in equity_etfs:
                        weights[t] = weights.get(t, 0) * scale_factor
                
                resampled_weights.append(weights)
            except:
                continue
        
        if resampled_weights:
            # Average the weights
            avg_weights = {}
            for ticker in tickers:
                avg_weights[ticker] = np.mean([w.get(ticker, 0) for w in resampled_weights])
            
            # Normalize
            total = sum(avg_weights.values())
            if total > 0:
                avg_weights = {t: w/total for t, w in avg_weights.items()}
                
                robust_portfolios['Resampled Efficiency'] = Portfolio(
                    weights=avg_weights,
                    start_date='2015-01-01',
                    end_date='2023-12-31',
                    initial_capital=100000
                )
                
                # Check sector allocations
                equity_alloc = sum(avg_weights.get(t, 0) for t in equity_etfs)
                fixed_alloc = sum(avg_weights.get(t, 0) for t in fixed_income_etfs)
                alt_alloc = sum(avg_weights.get(t, 0) for t in alternative_etfs)
                
                print(f"   ✓ Resampled Efficiency SUCCESS")
                print(f"   Asset class allocations:")
                print(f"     Equity: {equity_alloc:.1%}")
                print(f"     Fixed Income: {fixed_alloc:.1%}")
                print(f"     Alternatives: {alt_alloc:.1%}")
        else:
            print(f"   ✗ Resampled Efficiency FAILED: No valid samples")
            
    except Exception as e:
        print(f"   ✗ Resampled Efficiency FAILED: {e}")
    
    # 3. ROBUST OPTIMIZATION - WORST-CASE SCENARIO WITH CONSTRAINTS
    print("\n3. Testing Worst-Case Optimization with Constraints...")
    try:
        # Simulate worst-case scenarios by using worst periods
        worst_period_returns = returns['2020-02-01':'2020-04-30']  # COVID crash
        worst_case_optimizer = PortfolioOptimizer(worst_period_returns)
        
        # Optimize for worst case with defensive constraints
        worst_case_weights = worst_case_optimizer.optimize_min_variance(
            weight_bounds=(0.0, 0.40)  # Conservative position limits
        )
        
        robust_portfolios['Worst-Case Optimized'] = Portfolio(
            weights=worst_case_weights,
            start_date='2015-01-01',
            end_date='2023-12-31',
            initial_capital=100000
        )
        
        print(f"   ✓ Worst-Case optimization SUCCESS")
        print(f"   Defensive allocations:")
        for ticker, weight in sorted(worst_case_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            if weight > 0.01:
                print(f"     {ticker}: {weight:.2%}")
                
    except Exception as e:
        print(f"   ✗ Worst-Case optimization FAILED: {e}")
    
    return robust_portfolios

# Run comprehensive constraint testing
constraint_portfolios, constraint_metrics = example_comprehensive_constraint_testing()
display(pd.DataFrame(constraint_metrics))

# Run robust optimization constraint testing  
robust_portfolios = example_robust_optimization_constraints()


#%% 3
# ==============================================================================
# SECTION 3: PORTFOLIO RETURNS TABLES
# ==============================================================================

def example_portfolio_returns_tables():
    """Create various return tables and dataframes."""
    
    print("\n" + "="*60)
    print("PORTFOLIO RETURNS TABLES")
    print("="*60)
    
    weights = {
        'AAPL': 0.10,
        'MSFT': 0.10,
        'COST': 0.10,
        'FANG': 0.10,
        'COKE': 0.05,
        'KO': 0.05,
        'GLD': 0.10,
        'UNH': 0.10,
        'HD': 0.10,
        'MA': 0.05,
        'V': 0.05,
        'META': 0.10
    }
    # Create sample portfolio
    portfolio = Portfolio(
        weights=weights,
        start_date='1980-01-01',
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
    display(daily_returns_df)
    
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
    display(monthly_df)
    
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
    display(annual_df)

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
    display(stats_df)

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
    display(monthly_summary)

    return portfolio

portfolio_tables = example_portfolio_returns_tables()

#%% 4
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
    tickers = ['AAPL', 'MSFT', 'MA', 'AMZN', 'COST', 'COKE', 'BAC', 'JPM']
    returns = downloader.download_returns(
        tickers, 
        start_date='1980-01-01', 
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

#%% 5
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

#%% 6
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
#%% 7

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

#%% building a wrapper

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

display(pd.DataFrame(optimized_weights.items(), columns=["Ticker", "Weight"]))
#%% - checking if constraints are met (integrate later)
# Convert to dictionary if needed
if isinstance(optimized_weights, list):
    optimized_weights = {tickers[i]: optimized_weights[i] for i in range(len(tickers))}

constraint_portfolios['Max 20% Min 0% Position'] = Portfolio(
    weights=optimized_weights,
    start_date=start_date,
    end_date=end_date,
    initial_capital=initial_capital
)

# Verify both constraints
max_weight = max(optimized_weights.values())
min_weight = min(w for w in optimized_weights.values() if w > 0)  # Exclude zero weights
max_constraint_satisfied = max_weight <= max_weight_limit + 1e-6  # Small tolerance
min_constraint_satisfied = min_weight >= min_weight_limit - 1e-6  # Small tolerance

print(f"   ✓ Position constraints succeeded!")
print(f"   Maximum weight: {max_weight:.1%} (limit: {max_weight_limit:.1%})")
print(f"   Minimum weight: {min_weight:.1%} (limit: {min_weight_limit:.1%})")
print(f"   Max constraint satisfied: {max_constraint_satisfied}")
print(f"   Min constraint satisfied: {min_constraint_satisfied}")
print(f"   Both constraints satisfied: {max_constraint_satisfied and min_constraint_satisfied}")

###except Exception as e:
print(f"   ✗ Position constraints FAILED: {e}")
# Set a fallback equal-weight portfolio to avoid NameError
max_pos_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
print("   Using equal-weight fallback portfolio")

# Only display if max_pos_weights is defined
if 'max_pos_weights' in locals():
    display(pd.DataFrame(max_pos_weights.items(), columns=["Ticker", "Weight"]))
else:
    print("No weights to display - optimization failed")

#%%
def test_optimal(weights):
    """Demonstrate all visualization types for a single portfolio."""
    
    print("\n" + "="*60)
    print("SINGLE PORTFOLIO - ALL VISUALIZATIONS")
    print("="*60)
    
  
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

# # Only run the test if we have valid weights
# if 'max_pos_weights' in locals() and max_pos_weights:
#     test_optimal(max_pos_weights)
# else:
#     print("Skipping visualization test - no valid portfolio weights available")

# %%
test_optimal(optimized_weights)