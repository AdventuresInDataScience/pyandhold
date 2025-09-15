"""Module for downloading and managing financial data."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataDownloader:
    """Handle downloading and caching of financial data."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize DataDownloader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def download_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        prepost: bool = False,
        threads: bool = True
    ) -> pd.DataFrame:
        """
        Download historical price data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data (default: today)
            interval: Data interval (1d, 1wk, 1mo)
            auto_adjust: Adjust for dividends and splits
            prepost: Include pre/post market data
            threads: Use multithreading for download
            
        Returns:
            DataFrame with adjusted close prices, columns are tickers
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Convert dates to string format for yfinance
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
            threads=threads,
            progress=False
        )
        
        # Handle single ticker case
        if len(tickers) == 1:
            # For single ticker, check if data['Close'] is Series or DataFrame
            close_data = data['Close']
            if isinstance(close_data, pd.Series):
                prices = close_data.to_frame(tickers[0])
            else:
                # Already a DataFrame, just rename the column
                prices = close_data.copy()
                prices.columns = [tickers[0]]
        else:
            prices = data['Close']
        
        # Remove any tickers with all NaN values
        prices = prices.dropna(axis=1, how='all')
        
        return prices
    
    def download_returns(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download price data and calculate returns.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional arguments for download_data
            
        Returns:
            DataFrame with daily returns
        """
        prices = self.download_data(tickers, start_date, end_date, **kwargs)
        returns = prices.pct_change().dropna()
        return returns
    
    def get_benchmark_data(
        self,
        benchmark: str = "^GSPC",
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None
    ) -> pd.Series:
        """
        Download benchmark data (default: S&P 500).
        
        Args:
            benchmark: Benchmark ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Series with benchmark prices
        """
        data = self.download_data([benchmark], start_date, end_date)
        return data.iloc[:, 0]