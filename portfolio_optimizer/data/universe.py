"""Module for managing stock universes and common indices."""

import pandas as pd
from typing import List, Dict
import yfinance as yf


class StockUniverse:
    """Manage common stock universes and indices."""
    
    # Common ETFs representing indices
    INDEX_ETFS = {
        'sp500': 'SPY',
        'nasdaq100': 'QQQ',
        'russell2000': 'IWM',
        'dow30': 'DIA',
        'emerging_markets': 'EEM',
        'developed_markets': 'EFA',
        'bonds': 'AGG',
        'gold': 'GLD',
        'reits': 'VNQ',
        'commodities': 'DBC'
    }
    
    @classmethod
    def get_sp500_tickers(cls) -> List[str]:
        """Get current S&P 500 tickers."""
        # For production, this would fetch from a reliable source
        # Using placeholder for demonstration
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
            'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD'
        ]
    
    @classmethod
    def get_nasdaq100_tickers(cls) -> List[str]:
        """Get current NASDAQ-100 tickers."""
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
            'AVGO', 'PEP', 'COST', 'ADBE', 'CSCO', 'CMCSA', 'INTC', 'AMD'
        ]
    
    @classmethod
    def get_sector_etfs(cls) -> Dict[str, str]:
        """Get sector ETF tickers."""
        return {
            'technology': 'XLK',
            'healthcare': 'XLV',
            'financials': 'XLF',
            'energy': 'XLE',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'industrials': 'XLI',
            'materials': 'XLB',
            'utilities': 'XLU',
            'real_estate': 'XLRE',
            'communication': 'XLC'
        }
    
    @classmethod
    def get_asset_class_etfs(cls) -> Dict[str, str]:
        """Get major asset class ETF tickers."""
        return {
            'us_stocks': 'VTI',
            'international_stocks': 'VXUS',
            'bonds': 'BND',
            'real_estate': 'VNQ',
            'commodities': 'DBC',
            'gold': 'GLD',
            'tips': 'TIP',
            'high_yield': 'HYG',
            'emerging_bonds': 'EMB'
        }