"""Visualization module for portfolio analysis."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
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
        Plot portfolio performance over time with date slider.
        
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
        
        # Add date slider
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode='x unified',
            showlegend=True,
            height=500,
            width=1000,
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
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Plot returns distribution histogram with CDF.
        
        Args:
            returns: Series of returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create subplots: histogram and CDF side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Histogram (PDF)', 'Cumulative Distribution (CDF)'],
            horizontal_spacing=0.1
        )
        
        # 1. Histogram (PDF)
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='lightblue',
            opacity=0.7,
            histnorm='probability density'
        ), row=1, col=1)
        
        # Add normal distribution overlay on histogram
        mean = returns.mean()
        std = returns.std()
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ), row=1, col=1)
        
        # 2. Cumulative Distribution Function (CDF)
        sorted_returns = np.sort(returns)
        y_cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        
        fig.add_trace(go.Scatter(
            x=sorted_returns,
            y=y_cdf,
            mode='lines',
            name='Empirical CDF',
            line=dict(color='blue', width=2)
        ), row=1, col=2)
        
        # Add normal CDF overlay
        normal_cdf = stats.norm.cdf(x_range, mean, std)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_cdf,
            mode='lines',
            name='Normal CDF',
            line=dict(color='red', width=2, dash='dash')
        ), row=1, col=2)
        
        fig.update_layout(
            title=title,
            height=500,
            width=1000,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Returns", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        
        return fig
    
    @staticmethod
    def display_returns_statistics(returns: pd.Series) -> pd.DataFrame:
        """
        Calculate and return statistics DataFrame for pretty display.
        Better than plotly table - shows as scrollable dataframe.
        
        Args:
            returns: Series of returns
            
        Returns:
            DataFrame with formatted statistics
        """
        # Calculate statistics
        mean = returns.mean()
        std = returns.std()
        
        # Calculate additional statistics
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        except:
            jb_stat, jb_pvalue = np.nan, np.nan
        
        stats_data = {
            'Mean_Daily_Return': [mean],
            'Standard_Deviation': [std],
            'Annualized_Return': [mean * 252],
            'Annualized_Volatility': [std * np.sqrt(252)],
            'Skewness': [returns.skew()],
            'Kurtosis': [returns.kurtosis()],
            'Minimum_Return': [returns.min()],
            'Maximum_Return': [returns.max()],
            'VaR_5%': [returns.quantile(0.05)],
            'VaR_95%': [returns.quantile(0.95)],
            'Positive_Days_%': [(returns > 0).mean() * 100],
            'JB_Statistic': [jb_stat],
            'JB_p_value': [jb_pvalue],
            'Sharpe_Ratio': [mean / std * np.sqrt(252) if std > 0 else np.nan]
        }
        
        df = pd.DataFrame(stats_data)
        
        # Format for better display
        df = df.round(4)
        df.index = ['Portfolio_Returns']
        
        return df.T  # Transpose for vertical display
    
    @staticmethod
    def plot_drawdown(
        prices: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> go.Figure:
        """
        Plot drawdown chart with date slider.
        
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
        
        # Add date slider
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat='.1f'),
            hovermode='x unified',
            height=500,
            width=1000,
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
        
        return fig
    
    @staticmethod
    def plot_single_metric_timeseries(
        data: pd.Series,
        title: str = "Metric Over Time",
        y_label: str = "Value",
        color: str = "blue",
        show_mean: bool = True
    ) -> go.Figure:
        """
        Universal function to plot any single metric time series with optional mean line.
        
        Args:
            data: Series with datetime index and metric values
            title: Chart title
            y_label: Y-axis label
            color: Line color
            show_mean: Whether to show dashed mean line
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Main metric line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=title.split(' ')[0],  # Use first word as name
            line=dict(color=color, width=2)
        ))
        
        # Add mean line if requested
        if show_mean:
            mean_value = data.mean()
            fig.add_hline(
                y=mean_value,
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7,
                annotation_text=f"Mean: {mean_value:.3f}",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            hovermode='x unified',
            height=400,
            width=1000,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            )
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
        Plot portfolio weights as pie chart with modern pastel colors.
        
        Args:
            weights: Dictionary of weights
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Modern pastel color palette
        pastel_colors = [
            '#FF9999', '#66B2FF', '#99FF99', '#FFCC99', 
            '#FF99CC', '#99CCFF', '#FFB366', '#B3B3FF',
            '#99FFB3', '#FFD700', '#FF6B9D', '#87CEEB'
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3,
            marker=dict(
                colors=pastel_colors[:len(weights)],
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.1%}<br>Percent: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    @staticmethod
    def plot_period_returns(
        returns: pd.DataFrame,
        title: str = "Period Returns Analysis"
    ) -> go.Figure:
        """
        Plot returns by different periods with modern pastel colors.
        
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
        
        # Modern pastel color palette
        pastel_colors = ['#FFB3B3', '#B3D9FF', '#B3FFB3', '#FFE6B3']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily Returns', 'Weekly Returns', 'Monthly Returns', 'Yearly Returns']
        )
        
        # Daily
        fig.add_trace(
            go.Bar(
                x=daily_returns.index[-50:],  # Show last 50 days to avoid clutter
                y=daily_returns.iloc[-50:, 0] * 100,  # Convert to percentage
                name='Daily',
                marker_color=pastel_colors[0]
            ),
            row=1, col=1
        )
        
        # Weekly
        fig.add_trace(
            go.Bar(
                x=weekly_returns.index,
                y=weekly_returns.iloc[:, 0] * 100,
                name='Weekly',
                marker_color=pastel_colors[1]
            ),
            row=1, col=2
        )
        
        # Monthly
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.iloc[:, 0] * 100,
                name='Monthly',
                marker_color=pastel_colors[2]
            ),
            row=2, col=1
        )
        
        # Yearly
        if len(yearly_returns) > 0:
            fig.add_trace(
                go.Bar(
                    x=yearly_returns.index,
                    y=yearly_returns.iloc[:, 0] * 100,
                    name='Yearly',
                    marker_color=pastel_colors[3]
                ),
                row=2, col=2
            )
        
        # Update y-axis labels to show percentage
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns_table(
        returns: pd.Series,
        title: str = "Monthly Returns Table"
    ) -> go.Figure:
        """
        Plot monthly returns as a heatmap table.
        
        Args:
            returns: Series of daily returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create year-month pivot table
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100  # Convert to percentage
        })
        
        # Create pivot table
        pivot_table = monthly_data.pivot(index='Year', columns='Month', values='Return')
        
        # Fill NaN with 0 and ensure all months are present
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table = pivot_table.reindex(columns=range(1, 13), fill_value=np.nan)
        pivot_table.columns = month_names
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)", ticksuffix="%"),
            hoverongaps=False,
            hovertemplate='<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            width=1000,
            height=600
        )
        
        return fig
    
    @staticmethod
    @staticmethod  
    def display_drawdown_periods(portfolio_value: pd.Series, top_n: int = 10) -> pd.DataFrame:
        """
        Calculate and return top drawdown periods DataFrame for pretty display.
        Better than plotly table - shows as scrollable dataframe.
        
        Args:
            portfolio_value: Series of portfolio values
            top_n: Number of top periods to show
            
        Returns:
            DataFrame with formatted drawdown periods
        """
        # Calculate drawdown
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        
        # Find drawdown periods
        is_drawdown = drawdown < -0.001  # 0.1% threshold
        drawdown_periods = []
        
        start_idx = None
        for i, is_dd in enumerate(is_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                end_idx = i - 1
                period_dd = drawdown.iloc[start_idx:end_idx+1]
                min_dd_idx = period_dd.idxmin()
                min_dd = period_dd.min()
                
                drawdown_periods.append({
                    'Peak_Date': portfolio_value.index[start_idx],
                    'Trough_Date': min_dd_idx,
                    'Recovery_Date': portfolio_value.index[end_idx],
                    'Duration_Days': (portfolio_value.index[end_idx] - portfolio_value.index[start_idx]).days,
                    'Max_Drawdown_Pct': min_dd * 100,
                    'Peak_Value': portfolio_value.iloc[start_idx],
                    'Trough_Value': portfolio_value.loc[min_dd_idx],
                    'Recovery_Value': portfolio_value.iloc[end_idx]
                })
                start_idx = None
        
        # Handle case where last period is still in drawdown
        if start_idx is not None:
            period_dd = drawdown.iloc[start_idx:]
            min_dd_idx = period_dd.idxmin()
            min_dd = period_dd.min()
            drawdown_periods.append({
                'Peak_Date': portfolio_value.index[start_idx],
                'Trough_Date': min_dd_idx,
                'Recovery_Date': 'Ongoing',
                'Duration_Days': (portfolio_value.index[-1] - portfolio_value.index[start_idx]).days,
                'Max_Drawdown_Pct': min_dd * 100,
                'Peak_Value': portfolio_value.iloc[start_idx],
                'Trough_Value': portfolio_value.loc[min_dd_idx],
                'Recovery_Value': portfolio_value.iloc[-1]
            })
        
        if not drawdown_periods:
            # Return empty DataFrame with proper columns if no drawdowns
            return pd.DataFrame(columns=['Rank', 'Peak_Date', 'Trough_Date', 'Recovery_Date', 
                                       'Duration_Days', 'Max_Drawdown_Pct', 'Peak_Value', 
                                       'Trough_Value', 'Recovery_Value'])
        
        df = pd.DataFrame(drawdown_periods)
        # Sort by worst drawdown and take top N
        df = df.nsmallest(top_n, 'Max_Drawdown_Pct').reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        df = df[['Rank', 'Peak_Date', 'Trough_Date', 'Recovery_Date', 
                'Duration_Days', 'Max_Drawdown_Pct', 'Peak_Value', 
                'Trough_Value', 'Recovery_Value']]
        
        # Format for better display
        df['Max_Drawdown_Pct'] = df['Max_Drawdown_Pct'].round(2)
        df['Peak_Value'] = df['Peak_Value'].round(2)
        df['Trough_Value'] = df['Trough_Value'].round(2)
        df['Recovery_Value'] = df['Recovery_Value'].round(2)
        
        # Format dates as strings except 'Ongoing'
        for col in ['Peak_Date', 'Trough_Date', 'Recovery_Date']:
            if col == 'Recovery_Date':
                df[col] = df[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
                )
            else:
                df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        return df
    
    @staticmethod
    def plot_cumulative_returns(
        returns: pd.Series,
        title: str = "Cumulative Returns"
    ) -> go.Figure:
        """
        Plot cumulative returns chart with date slider.
        
        Args:
            returns: Series of daily returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='green', width=2)
        ))
        
        # Add date slider
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis_tickformat='.0%',
            hovermode='x unified',
            width=1000,
            height=500,
            title_font_size=16,
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
        
        return fig