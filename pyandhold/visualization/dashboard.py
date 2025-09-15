"""Streamlit dashboard for PyAndHold portfolio optimization."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional
from pyandhold.portfolio.portfolio import Portfolio
from pyandhold.data.downloader import DataDownloader
from pyandhold.data.universe import StockUniverse
from pyandhold.optimization.optimizers import PortfolioOptimizer
from pyandhold.optimization.robust import RobustOptimizer
from pyandhold.visualization.plots import PortfolioVisualizer


class PortfolioDashboard:
    """Streamlit dashboard for portfolio analysis."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.downloader = DataDownloader()
        self.visualizer = PortfolioVisualizer()
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="PyAndHold",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("📈 PyAndHold Portfolio Analyzer")
        st.markdown("---")
    
    def run(self):
        """Run the dashboard."""
        # Sidebar for inputs
        with st.sidebar:
            st.header("Configuration")
            
            # Asset selection
            st.subheader("1. Select Assets")
            asset_selection = st.selectbox(
                "Choose selection method:",
                ["Manual", "S&P 500", "NASDAQ 100", "Sector ETFs", "Custom List"]
            )
            
            if asset_selection == "Manual":
                tickers_input = st.text_area(
                    "Enter tickers (comma-separated):",
                    value="AAPL,MSFT,GOOGL,AMZN,META"
                )
                tickers = [t.strip() for t in tickers_input.split(",")]
            elif asset_selection == "S&P 500":
                tickers = StockUniverse.get_sp500_tickers()[:20]  # Top 20
            elif asset_selection == "NASDAQ 100":
                tickers = StockUniverse.get_nasdaq100_tickers()[:20]
            elif asset_selection == "Sector ETFs":
                sector_etfs = StockUniverse.get_sector_etfs()
                tickers = list(sector_etfs.values())
            else:
                custom_list = st.text_area("Enter custom tickers:")
                tickers = [t.strip() for t in custom_list.split(",")]
            
            st.write(f"Selected {len(tickers)} assets")
            
            # Date range
            st.subheader("2. Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=3*365)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            # Portfolio settings
            st.subheader("3. Portfolio Settings")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                value=100000,
                step=1000
            )
            
            rebalance_freq = st.selectbox(
                "Rebalancing Frequency",
                ["None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
            )
            
            rebalance_map = {
                "None": None, "Daily": "D", "Weekly": "W",
                "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"
            }
            rebalance_freq = rebalance_map[rebalance_freq]
            
            # Optimization settings
            st.subheader("4. Optimization")
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Equal Weight", "Max Sharpe", "Min Variance", 
                 "Risk Parity", "Max Return", "Min CVaR"]
            )
            
            if optimization_method in ["Max Return"]:
                max_volatility = st.slider(
                    "Max Annual Volatility (%)",
                    min_value=5,
                    max_value=50,
                    value=20
                ) / 100
            
            # Constraints
            st.subheader("5. Constraints")
            min_weight = st.slider(
                "Min Weight per Asset (%)",
                min_value=0,
                max_value=50,
                value=0
            ) / 100
            
            max_weight = st.slider(
                "Max Weight per Asset (%)",
                min_value=10,
                max_value=100,
                value=100
            ) / 100
            
            # Advanced options
            with st.expander("Advanced Options"):
                use_robust = st.checkbox("Use Robust Optimization")
                if use_robust:
                    robust_method = st.selectbox(
                        "Robust Method",
                        ["Cross-Validation", "Bootstrap", "Shrinkage"]
                    )
                
                transaction_cost = st.number_input(
                    "Transaction Cost (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1
                ) / 100
                
                benchmark = st.text_input("Benchmark Ticker", value="^GSPC")
            
            # Run optimization button
            optimize_button = st.button("🚀 Run Analysis", type="primary")
        
        # Main content area
        if optimize_button:
            with st.spinner("Fetching data and optimizing portfolio..."):
                # Download data
                prices = self.downloader.download_data(
                    tickers,
                    start_date,
                    end_date
                )
                
                returns = prices.pct_change().dropna()
                
                # Optimize portfolio
                if optimization_method == "Equal Weight":
                    weights = {ticker: 1/len(tickers) for ticker in tickers}
                else:
                    if use_robust:
                        optimizer = RobustOptimizer(returns)
                        if robust_method == "Cross-Validation":
                            weights = optimizer.cross_validation_optimize(
                                optimization_method.lower().replace(" ", "_"),
                                weight_bounds=(min_weight, max_weight)
                            )
                        elif robust_method == "Bootstrap":
                            weights = optimizer.bootstrap_optimize(
                                optimization_method.lower().replace(" ", "_"),
                                weight_bounds=(min_weight, max_weight)
                            )
                        else:
                            weights = optimizer.shrinkage_covariance_optimize(
                                optimization_method.lower().replace(" ", "_"),
                                weight_bounds=(min_weight, max_weight)
                            )
                    else:
                        optimizer = PortfolioOptimizer(returns)
                        
                        if optimization_method == "Max Sharpe":
                            weights = optimizer.optimize_sharpe(
                                weight_bounds=(min_weight, max_weight)
                            )
                        elif optimization_method == "Min Variance":
                            weights = optimizer.optimize_min_variance(
                                weight_bounds=(min_weight, max_weight)
                            )
                        elif optimization_method == "Risk Parity":
                            weights = optimizer.optimize_risk_parity(
                                weight_bounds=(min_weight, max_weight)
                            )
                        elif optimization_method == "Max Return":
                            weights = optimizer.optimize_max_return(
                                max_volatility=max_volatility,
                                weight_bounds=(min_weight, max_weight)
                            )
                        elif optimization_method == "Min CVaR":
                            weights = optimizer.optimize_cvar(
                                weight_bounds=(min_weight, max_weight)
                            )
                
                # Create portfolio
                portfolio = Portfolio(
                    weights=weights,
                    start_date=start_date,
                    end_date=end_date,
                    benchmark=benchmark,
                    rebalance_frequency=rebalance_freq,
                    transaction_cost=transaction_cost,
                    initial_capital=initial_capital
                )
                
                # Calculate metrics
                metrics = portfolio.calculate_metrics()
                portfolio_value = portfolio.calculate_portfolio_value()
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "📈 Performance", "🎯 Metrics", 
                "⚖️ Weights", "🔄 Optimization"
            ])
            
            with tab1:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{metrics['total_return']*100:.2f}%"
                    )
                with col2:
                    st.metric(
                        "Annual Return (CAGR)",
                        f"{metrics['cagr']*100:.2f}%"
                    )
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics['sharpe_ratio']:.2f}"
                    )
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{metrics['max_drawdown']*100:.2f}%"
                    )
                
                # Performance chart
                st.subheader("Portfolio Performance")
                perf_fig = self.visualizer.plot_performance(
                    portfolio_value,
                    title="Portfolio Value Over Time"
                )
                st.plotly_chart(perf_fig, use_container_width=True)
                
                # Drawdown chart
                st.subheader("Drawdown Analysis")
                dd_fig = self.visualizer.plot_drawdown(portfolio_value)
                st.plotly_chart(dd_fig, use_container_width=True)
            
            with tab2:
                # Returns distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Returns Distribution")
                    dist_fig = self.visualizer.plot_returns_distribution(
                        portfolio.portfolio_returns
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)
                
                with col2:
                    st.subheader("Period Returns")
                    period_returns = pd.DataFrame({
                        'Daily': portfolio.portfolio_returns,
                    })
                    period_fig = self.visualizer.plot_period_returns(
                        period_returns
                    )
                    st.plotly_chart(period_fig, use_container_width=True)
                
                # Rolling metrics
                st.subheader("Rolling Metrics (252-day window)")
                rolling_metrics = portfolio.get_rolling_metrics()
                rolling_fig = self.visualizer.plot_rolling_metrics(rolling_metrics)
                st.plotly_chart(rolling_fig, use_container_width=True)
            
            with tab3:
                # Detailed metrics table
                st.subheader("Performance Metrics")
                
                metrics_df = pd.DataFrame([
                    {"Metric": "Total Return", "Value": f"{metrics['total_return']*100:.2f}%"},
                    {"Metric": "CAGR", "Value": f"{metrics['cagr']*100:.2f}%"},
                    {"Metric": "Volatility (Annual)", "Value": f"{metrics['volatility']*100:.2f}%"},
                    {"Metric": "Sharpe Ratio", "Value": f"{metrics['sharpe_ratio']:.3f}"},
                    {"Metric": "Sortino Ratio", "Value": f"{metrics['sortino_ratio']:.3f}"},
                    {"Metric": "Calmar Ratio", "Value": f"{metrics['calmar_ratio']:.3f}"},
                    {"Metric": "Max Drawdown", "Value": f"{metrics['max_drawdown']*100:.2f}%"},
                    {"Metric": "Ulcer Index", "Value": f"{metrics['ulcer_index']*100:.2f}%"},
                    {"Metric": "VaR (95%)", "Value": f"{metrics['var_95']*100:.2f}%"},
                    {"Metric": "CVaR (95%)", "Value": f"{metrics['cvar_95']*100:.2f}%"},
                    {"Metric": "Omega Ratio", "Value": f"{metrics['omega_ratio']:.3f}"},
                    {"Metric": "K-Ratio", "Value": f"{metrics['k_ratio']:.3f}"}
                ])
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Correlation matrix
                st.subheader("Asset Correlation Matrix")
                corr_matrix = portfolio.get_correlation_matrix()
                corr_fig = self.visualizer.plot_correlation_heatmap(corr_matrix)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            with tab4:
                # Weights visualization
                st.subheader("Portfolio Weights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    pie_fig = self.visualizer.plot_weights_pie(weights)
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                with col2:
                    # Weights table
                    weights_df = pd.DataFrame([
                        {"Asset": ticker, "Weight": f"{weight*100:.2f}%"}
                        for ticker, weight in sorted(
                            weights.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                    ])
                    st.dataframe(weights_df, use_container_width=True)
            
            with tab5:
                # Efficient frontier
                st.subheader("Efficient Frontier")
                
                with st.spinner("Calculating efficient frontier..."):
                    optimizer_ef = PortfolioOptimizer(returns)
                    frontier = optimizer_ef.efficient_frontier(
                        n_portfolios=30,
                        weight_bounds=(min_weight, max_weight)
                    )
                    
                    current_portfolio = {
                        'return': metrics['annualized_return'],
                        'volatility': metrics['volatility']
                    }
                    
                    ef_fig = self.visualizer.plot_efficient_frontier(
                        frontier,
                        current_portfolio
                    )
                    st.plotly_chart(ef_fig, use_container_width=True)
                
                # Optimization comparison
                st.subheader("Strategy Comparison")
                
                strategies = ["Equal Weight", "Max Sharpe", "Min Variance", "Risk Parity"]
                comparison_data = []
                
                for strategy in strategies:
                    if strategy == "Equal Weight":
                        w = {ticker: 1/len(tickers) for ticker in tickers}
                    else:
                        opt = PortfolioOptimizer(returns)
                        if strategy == "Max Sharpe":
                            w = opt.optimize_sharpe((min_weight, max_weight))
                        elif strategy == "Min Variance":
                            w = opt.optimize_min_variance((min_weight, max_weight))
                        elif strategy == "Risk Parity":
                            w = opt.optimize_risk_parity((min_weight, max_weight))
                    
                    # Quick metrics calculation
                    p = Portfolio(w, start_date, end_date, initial_capital=initial_capital)
                    m = p.calculate_metrics()
                    
                    comparison_data.append({
                        'Strategy': strategy,
                        'Return': f"{m['annualized_return']*100:.2f}%",
                        'Volatility': f"{m['volatility']*100:.2f}%",
                        'Sharpe': f"{m['sharpe_ratio']:.3f}",
                        'Max DD': f"{m['max_drawdown']*100:.2f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)


def main():
    """Main function to run the dashboard."""
    dashboard = PortfolioDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()