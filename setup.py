from setuptools import setup, find_packages

setup(
    name="pyandhold",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "yfinance>=0.2.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.2.0",
        "plotly>=5.10.0",
        "streamlit>=1.20.0",
        "cvxpy>=1.3.0",
        "pyportfolioopt>=1.5.0",
        "pandas-datareader>=0.10.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.6.0",
    ],
    python_requires=">=3.8",
)