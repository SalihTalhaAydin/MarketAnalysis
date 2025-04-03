import yfinance as yf
import pandas as pd  # Often needed alongside yfinance/TA
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
# from ta import add_all_ta_features # Keep commented if not used

# ==============================================================================
# Data Loading Function
# ==============================================================================


def load_data(ticker, start_date, end_date):
    """
    Loads historical market data for a given ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for data retrieval (YYYY-MM-DD).
        end_date (str): The end date for data retrieval (YYYY-MM-DD).

    Returns:
        pd.DataFrame: OHLCV data, or None if download fails.
    """
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    # Use yfinance to download historical market data
    data = yf.download(
        tickers=ticker,       # Ticker symbol(s)
        start=start_date,     # Start date for data retrieval
        end=end_date,         # End date for data retrieval
        auto_adjust=False,    # Keep OHLC as is (no split/dividend adjust)
        progress=False        # Suppress download progress bar
    )
    if data.empty:
        print(f"Could not download data for {ticker}.")
        return None
    return data


# ==============================================================================
# Data Preprocessing Function
# ==============================================================================


def preprocess_data(df, ticker):
    """
    Preprocesses the input DataFrame by adding technical indicators.

    Args:
        df (pd.DataFrame): DataFrame with market data (expects MultiIndex).
        ticker (str): The ticker symbol corresponding to the data in df.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with features, or None if invalid.
    """
    print("Preprocessing data...")
    if df is None or df.empty:
        print("No data to preprocess.")
        return None

    # Validate input DataFrame structure (assuming yfinance MultiIndex)
    if not isinstance(df.columns, pd.MultiIndex):
        print("Error: Input DataFrame lacks expected MultiIndex columns.")
        return None
    required_cols = [
        ('Open', ticker), ('High', ticker), ('Low', ticker),
        ('Close', ticker), ('Volume', ticker)
    ]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns for {ticker} in DataFrame.")
        return None

    # --- Feature Engineering ---
    # Calculate Simple Moving Average (SMA)
    df[('SMA_20', ticker)] = df[('Close', ticker)].rolling(window=20).mean()

    # Calculate On-Balance Volume (OBV)
    obv_indicator = OnBalanceVolumeIndicator(
        close=df[('Close', ticker)], volume=df[('Volume', ticker)]
    )
    df[('OBV', ticker)] = obv_indicator.on_balance_volume()

    # Calculate Money Flow Index (MFI)
    mfi_indicator = MFIIndicator(
        high=df[('High', ticker)],
        low=df[('Low', ticker)],
        close=df[('Close', ticker)],
        volume=df[('Volume', ticker)]
    )
    df[('MFI', ticker)] = mfi_indicator.money_flow_index()

    # Drop rows with NaN values (often introduced by rolling calculations)
    df.dropna(inplace=True)

    print("Preprocessing complete.")
    return df


# ==============================================================================
# Example Usage (when script is run directly)
# ==============================================================================


if __name__ == '__main__':
    # Define parameters for data loading
    ticker_symbol = 'AAPL'  # Stock ticker symbol
    start_date = '2023-01-01'  # Data start date
    end_date = '2025-01-01'    # Data end date

    # Load the raw market data
    print(f"\n--- Loading Data for {ticker_symbol} ---")
    raw_data = load_data(ticker_symbol, start_date, end_date)

    # Preprocess the data if loading was successful
    if raw_data is not None and not raw_data.empty:
        print("\n--- Preprocessing Data ---")
        # Use copy() to avoid SettingWithCopyWarning on the original DataFrame
        processed_data = preprocess_data(raw_data.copy(), ticker=ticker_symbol)

        # Display results if preprocessing was successful
        if processed_data is not None:
            print("\nProcessed Data Sample:")
            print(processed_data.head())
            print(f"\nShape of processed data: {processed_data.shape}")
            # Save the processed data for later use
            try:
                # Ensure the data directory exists (optional, good practice)
                import os
                data_dir = 'market_ml_model/data'
                os.makedirs(data_dir, exist_ok=True)
                # Construct filename dynamically
                filename = (
                    f'processed_{ticker_symbol}_{start_date}_to_{end_date}.csv'
                )
                save_path = os.path.join(data_dir, filename)
                processed_data.to_csv(save_path)
                print(f"\nProcessed data saved to {save_path}")
            except Exception as e:
                print(f"\nError saving processed data: {e}")
            #     print(f"\nError saving processed data: {e}")
    else:
        print(f"\nCould not load or preprocess data for {ticker_symbol}.")
