"""Visualization module for portfolio analysis."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
import seaborn as sns
import matplotlib.pyplot as plt


class PortfolioVisualizer:
    """Create various portfolio visualizations."""
    
    @staticmethod
    def plot_performance(
        portfolio_value: pd.Series,
        benchmark_value: Optional[pd.Series] = None,
        title: str = "Portfolio Performance"
    ) -> go.Figure:
        """
        Plot portfolio performance over time.
        
        Args:
            portfolio_value: Series of portfolio values
            benchmark_value: Optional benchmark values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark line if provided
        if benchmark_value is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_value.index,
                y=benchmark_value,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Plot returns distribution histogram.
        
        Args:
            returns: Series of returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mean = returns.mean()
        std = returns.std()
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Returns",
            yaxis_title="Frequency",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown(
        prices: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> go.Figure:
        """
        Plot drawdown chart.
        
        Args:
            prices: Series of prices
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate drawdown
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat='.1f'),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_rolling_metrics(
        metrics_df: pd.DataFrame,
        title: str = "Rolling Metrics"
    ) -> go.Figure:
        """
        Plot rolling metrics over time.
        
        Args:
            metrics_df: DataFrame with rolling metrics
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=len(metrics_df.columns),
            cols=1,
            subplot_titles=list(metrics_df.columns),
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(metrics_df.columns, 1):
            fig.add_trace(
                go.Scatter(
                    x=metrics_df.index,
                    y=metrics_df[col],
                    mode='lines',
                    name=col,
                    showlegend=False
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(metrics_df.columns),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix"
    ) -> go.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=800
        )
        
        return fig
    
    @staticmethod
    def plot_efficient_frontier(
        frontier_df: pd.DataFrame,
        current_portfolio: Optional[Dict] = None,
        title: str = "Efficient Frontier"
    ) -> go.Figure:
        """
        Plot efficient frontier.
        
        Args:
            frontier_df: DataFrame with frontier portfolios
            current_portfolio: Optional current portfolio metrics
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Efficient frontier line
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=2),
            marker=dict(size=5)
        ))
        
        # Highlight max Sharpe ratio portfolio
        max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
        fig.add_trace(go.Scatter(
            x=[frontier_df.loc[max_sharpe_idx, 'volatility']],
            y=[frontier_df.loc[max_sharpe_idx, 'return']],
            mode='markers',
            name='Max Sharpe Ratio',
            marker=dict(color='green', size=15, symbol='star')
        ))
        
        # Current portfolio if provided
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['volatility']],
                y=[current_portfolio['return']],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='red', size=12, symbol='diamond')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Volatility (Annual %)",
            yaxis_title="Expected Return (Annual %)",
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_weights_pie(
        weights: Dict[str, float],
        title: str = "Portfolio Weights"
    ) -> go.Figure:
        """
        Plot portfolio weights as pie chart.
        
        Args:
            weights: Dictionary of weights
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3
        )])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(title=title)
        
        return fig
    
    @staticmethod
    def plot_period_returns(
        returns: pd.DataFrame,
        title: str = "Period Returns Analysis"
    ) -> go.Figure:
        """
        Plot returns by different periods.
        
        Args:
            returns: DataFrame of returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate period returns
        daily_returns = returns
        weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily Returns', 'Weekly Returns', 'Monthly Returns', 'Yearly Returns']
        )
        
        # Daily
        fig.add_trace(
            go.Bar(x=daily_returns.index, y=daily_returns.iloc[:, 0], name='Daily'),
            row=1, col=1
        )
        
        # Weekly
        fig.add_trace(
            go.Bar(x=weekly_returns.index, y=weekly_returns.iloc[:, 0], name='Weekly'),
            row=1, col=2
        )
        
        # Monthly
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns.iloc[:, 0], name='Monthly'),
            row=2, col=1
        )
        
        # Yearly
        if len(yearly_returns) > 0:
            fig.add_trace(
                go.Bar(x=yearly_returns.index, y=yearly_returns.iloc[:, 0], name='Yearly'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig