"""Backtesting module for portfolio strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Union
from datetime import datetime
from ..metrics.returns import ReturnMetrics
from ..metrics.risk import RiskMetrics
from ..metrics.performance import PerformanceMetrics


class Backtester:
    """Backtest portfolio strategies with advanced features."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        min_trade_size: float = 100
    ):
        """
        Initialize backtester.
        
        Args:
            data: DataFrame of asset prices
            initial_capital: Starting capital
            commission: Commission rate
            slippage: Slippage rate
            min_trade_size: Minimum trade size
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_trade_size = min_trade_size
        
        # Results storage
        self.portfolio_value = None
        self.positions = None
        self.trades = []
        self.metrics = {}
    
    def backtest_fixed_weights(
        self,
        weights: Dict[str, float],
        rebalance_frequency: Optional[str] = None,
        leverage: float = 1.0
    ) -> pd.DataFrame:
        """
        Backtest fixed weight strategy.
        
        Args:
            weights: Target weights
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            leverage: Leverage factor
            
        Returns:
            DataFrame with backtest results
        """
        portfolio_value = self.initial_capital
        values = []
        positions = {ticker: 0 for ticker in weights.keys()}
        
        # Initial allocation
        for ticker, weight in weights.items():
            if ticker in self.data.columns:
                position_value = portfolio_value * weight * leverage
                shares = position_value / self.data[ticker].iloc[0]
                positions[ticker] = shares
                portfolio_value -= self.commission * position_value
        
        # Track portfolio over time
        for date, prices in self.data.iterrows():
            # Calculate current portfolio value
            current_value = sum(
                positions[ticker] * prices[ticker]
                for ticker in positions if ticker in prices.index
            )
            
            # Add cash (if leverage < 1)
            cash = portfolio_value - sum(
                positions[ticker] * self.data[ticker].iloc[0]
                for ticker in positions if ticker in self.data.columns
            )
            current_value += cash
            
            values.append({
                'date': date,
                'value': current_value,
                'returns': (current_value / portfolio_value - 1) if len(values) > 0 else 0
            })
            
            # Rebalance if needed
            if rebalance_frequency and self._should_rebalance(date, rebalance_frequency):
                # Calculate new positions
                for ticker, weight in weights.items():
                    if ticker in prices.index:
                        target_value = current_value * weight * leverage
                        target_shares = target_value / prices[ticker]
                        
                        # Trade if difference is significant
                        share_diff = target_shares - positions[ticker]
                        if abs(share_diff * prices[ticker]) > self.min_trade_size:
                            # Apply costs
                            trade_cost = abs(share_diff * prices[ticker]) * (self.commission + self.slippage)
                            current_value -= trade_cost
                            positions[ticker] = target_shares
                            
                            # Record trade
                            self.trades.append({
                                'date': date,
                                'ticker': ticker,
                                'shares': share_diff,
                                'price': prices[ticker],
                                'cost': trade_cost
                            })
            
            portfolio_value = current_value
        
        self.portfolio_value = pd.DataFrame(values).set_index('date')
        return self.portfolio_value
    
    def backtest_dynamic_strategy(
        self,
        strategy_func: Callable,
        lookback_period: int = 252,
        rebalance_frequency: str = 'M',
        **strategy_kwargs
    ) -> pd.DataFrame:
        """
        Backtest dynamic strategy with changing weights.
        
        Args:
            strategy_func: Function that returns weights given historical data
            lookback_period: Lookback period for strategy
            rebalance_frequency: Rebalancing frequency
            **strategy_kwargs: Additional arguments for strategy function
            
        Returns:
            DataFrame with backtest results
        """
        portfolio_value = self.initial_capital
        values = []
        positions = {}
        
        # Get rebalance dates
        rebalance_dates = self.data.resample(rebalance_frequency).first().index
        
        for i, date in enumerate(self.data.index):
            if i < lookback_period:
                values.append({
                    'date': date,
                    'value': portfolio_value,
                    'returns': 0
                })
                continue
            
            # Get historical data
            historical_data = self.data.iloc[i-lookback_period:i]
            
            # Rebalance if needed
            if date in rebalance_dates:
                # Get new weights from strategy
                weights = strategy_func(historical_data, **strategy_kwargs)
                
                # Calculate current portfolio value
                current_prices = self.data.iloc[i]
                current_value = sum(
                    positions.get(ticker, 0) * current_prices[ticker]
                    for ticker in current_prices.index if ticker in positions
                )
                
                # Update positions
                new_positions = {}
                for ticker, weight in weights.items():
                    if ticker in current_prices.index:
                        target_value = current_value * weight
                        target_shares = target_value / current_prices[ticker]
                        
                        # Apply transaction costs
                        if ticker in positions:
                            share_diff = target_shares - positions[ticker]
                        else:
                            share_diff = target_shares
                        
                        trade_cost = abs(share_diff * current_prices[ticker]) * (self.commission + self.slippage)
                        current_value -= trade_cost
                        new_positions[ticker] = target_shares
                
                positions = new_positions
                portfolio_value = current_value
            
            # Calculate current value
            current_prices = self.data.iloc[i]
            current_value = sum(
                positions.get(ticker, 0) * current_prices[ticker]
                for ticker in current_prices.index if ticker in positions
            )
            
            values.append({
                'date': date,
                'value': current_value,
                'returns': (current_value / values[-1]['value'] - 1) if len(values) > 1 else 0
            })
        
        self.portfolio_value = pd.DataFrame(values).set_index('date')
        return self.portfolio_value
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        if self.portfolio_value is None:
            raise ValueError("Run backtest first")
        
        returns = self.portfolio_value['returns']
        values = self.portfolio_value['value']
        
        # Calculate metrics
        self.metrics = {
            'total_return': (values.iloc[-1] / self.initial_capital) - 1,
            'cagr': ReturnMetrics.cagr(values.to_frame()),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns.to_frame()).iloc[0],
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns.to_frame()).iloc[0],
            'max_drawdown': RiskMetrics.max_drawdown(values.to_frame()).iloc[0],
            'calmar_ratio': PerformanceMetrics.calmar_ratio(values.to_frame()).iloc[0],
            'win_rate': (returns > 0).mean(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'total_trades': len(self.trades),
            'total_commission': sum(t['cost'] for t in self.trades) if self.trades else 0
        }
        
        return self.metrics
    
    def _should_rebalance(self, date: pd.Timestamp, frequency: str) -> bool:
        """Check if should rebalance on given date."""
        if frequency == 'D':
            return True
        elif frequency == 'W':
            return date.weekday() == 0  # Monday
        elif frequency == 'M':
            return date.day == 1
        elif frequency == 'Q':
            return date.month in [1, 4, 7, 10] and date.day == 1
        elif frequency == 'Y':
            return date.month == 1 and date.day == 1
        return False