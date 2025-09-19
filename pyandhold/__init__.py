"""PyAndHold: A comprehensive library for portfolio analysis and optimization."""

from .portfolio.portfolio import Portfolio
from .portfolio.backtester import Backtester
from .data.downloader import DataDownloader
from .data.universe import StockUniverse
from .data.preprocessor import DataPreprocessor
from .optimization.optimizers import PortfolioOptimizer
from .optimization.robust import RobustOptimizer
from .optimization.constraints import ConstraintBuilder
from .utils.helpers import PortfolioHelpers, Summariser
# Comment out or fix the problematic imports
# from .visualization.visualizer import PortfolioVisualizer
# from .metrics.return_metrics import ReturnMetrics
# from .metrics.risk_metrics import RiskMetrics
# from .metrics.performance_metrics import PerformanceMetrics


__version__ = "0.1.0"
__all__ = [
    "Portfolio", "Backtester", 
    "DataDownloader", "StockUniverse", "DataPreprocessor",
    "PortfolioOptimizer", "RobustOptimizer", "ConstraintBuilder",
    "PortfolioHelpers", "Summariser",
    # Remove or fix these entries as well
    # "PortfolioVisualizer",
    # "ReturnMetrics", "RiskMetrics", "PerformanceMetrics"
]