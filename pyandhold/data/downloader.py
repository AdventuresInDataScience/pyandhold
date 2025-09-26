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
        threads: bool = True,
        return_both: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
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
            return_both: If True, return tuple of (prices, returns)
            
        Returns:
            DataFrame with adjusted close prices, columns are tickers
            Or if return_both=True: tuple of (prices DataFrame, returns DataFrame)
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
        
        if return_both:
            returns = prices.pct_change().dropna()
            return prices, returns
        
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
    
    def download_prices_and_returns(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download price data and calculate returns in a single efficient call.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional arguments for download_data
            
        Returns:
            Tuple of (prices DataFrame, returns DataFrame)
        """
        return self.download_data(tickers, start_date, end_date, return_both=True, **kwargs)
    
    def download_with_flexible_alignment(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        align_all: bool = False,
        min_history: Optional[int] = None,
        **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download data with flexible alignment options to preserve maximum history.
        
        This method is designed for workflows where you want to:
        1. Download data for a large universe of assets
        2. Optimize/select a subset 
        3. Then align data only for the selected subset
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            align_all: If True, align all assets to common date range (traditional approach)
                      If False, keep individual asset histories intact (recommended)
            min_history: Minimum history per asset (assets with less history will be dropped)
            **kwargs: Additional arguments for download_data
            
        Returns:
            Tuple of (prices DataFrame, returns DataFrame)
            - If align_all=False: Assets may have different start dates (preserves maximum history)
            - If align_all=True: All assets aligned to common date range (traditional approach)
        """
        from .preprocessor import DataPreprocessor
        
        # Download raw data
        prices, returns = self.download_data(tickers, start_date, end_date, return_both=True, **kwargs)
        
        if not align_all:
            # Preserve individual asset histories - only remove assets with insufficient history
            if min_history:
                valid_assets = []
                for col in prices.columns:
                    asset_data = prices[col].dropna()
                    if len(asset_data) >= min_history:
                        valid_assets.append(col)
                
                if len(valid_assets) < len(prices.columns):
                    print(f"Removing {len(prices.columns) - len(valid_assets)} assets with insufficient history (<{min_history} observations)")
                    prices = prices[valid_assets]
                    returns = returns[valid_assets]
            
            # Don't align - preserve maximum history for each asset
            return prices, returns
        else:
            # Traditional approach - align all to common date range
            aligned_prices = DataPreprocessor.align_data(prices, min_history=min_history)
            aligned_returns = DataPreprocessor.align_data(returns, min_history=min_history)
            return aligned_prices, aligned_returns
    
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