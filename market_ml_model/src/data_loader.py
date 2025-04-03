import yfinance as yf
# import pandas as pd # No longer needed here
# from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
# Features moved to feature_engineering.py
# from ta import add_all_ta_features # Keep commented if not used

# ==============================================================================
# Data Loading Function
# ==============================================================================


def load_data(ticker, start_date, end_date, interval="1d"):
    """
    Loads historical market data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for data retrieval (YYYY-MM-DD).
        end_date (str): The end date for data retrieval (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d').
                        Default '1d'.

    Returns:
        pd.DataFrame: OHLCV data, or None if download fails.
    """
    print(f"Loading {interval} data for {ticker} from {start_date} "
          f"to {end_date}...")
    # Use yfinance to download historical market data
    data = yf.download(
        tickers=ticker,       # Ticker symbol(s)
        start=start_date,     # Start date for data retrieval
        end=end_date,         # End date for data retrieval
        interval=interval,    # Data interval
        auto_adjust=False,    # Keep OHLC as is (no split/dividend adjust)
        progress=False        # Suppress download progress bar
    )
    if data.empty:
        print(f"Could not download data for {ticker}.")
        return None
    return data


# (Removed unused preprocess_data function and example usage block)
