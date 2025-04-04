from setuptools import find_packages, setup

setup(
    name="market_ml_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "yfinance>=0.2.12",
        "pandas-ta>=0.3.14b0",
        "ta>=0.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "full": [
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "catboost>=1.1.0",
            "tensorflow>=2.10.0",
            "shap>=0.41.0",
            "hyperopt>=0.2.7",
            "optuna>=3.0.0",
            "ccxt>=2.0.0",
            "alpha_vantage>=2.3.1",
            "pyfolio>=0.9.2",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pylint>=2.15.0",
            "black>=23.0.0",  # Added black
            "isort>=5.10.0",  # Added isort
            "pytest-mock>=3.10.0",  # Added for mocker fixture
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning framework for market prediction and trading",
    keywords="finance, machine learning, trading, quantitative analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
