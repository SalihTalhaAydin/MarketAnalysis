import pandas as pd


def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Performs initial data cleaning and standardization.

    Args:
        df: Raw OHLCV DataFrame (potentially MultiIndex from yfinance).
        ticker: The ticker symbol for which data is being processed.

    Returns:
        DataFrame with standardized columns and basic cleaning,
        or None if fails.
    """
    print("--- Initial Data Preprocessing ---")
    if df is None or df.empty:
        print("Error: Input DataFrame is empty/None for preprocessing.")
        return None

    processed_df = df.copy()

    # --- Standardize Column Names (lowercase) ---
    # Handle potential MultiIndex from yfinance if multiple tickers were loaded
    if isinstance(processed_df.columns, pd.MultiIndex):
        print(f"Extracting data for '{ticker}' from MultiIndex...")
        # Check if ticker is at level 1 (standard yfinance format)
        if ticker in processed_df.columns.get_level_values(1):
            try:
                idx = pd.IndexSlice
                processed_df = processed_df.loc[:, idx[:, ticker]].copy()
                processed_df.columns = processed_df.columns.droplevel(1)
                processed_df.columns = processed_df.columns.str.lower()
                print(f"Extracted {ticker} data from column level 1.")
            except Exception as e:
                print(f"Error extracting ticker {ticker} from level 1: {e}")
                return None
        # Check if ticker is at level 0 (current test fixture format)
        elif ticker in processed_df.columns.get_level_values(0):
            try:
                processed_df = processed_df.xs(ticker, level=0, axis=1).copy()
                processed_df.columns = processed_df.columns.str.lower()
                print(f"Extracted {ticker} data from column level 0.")
            except KeyError:
                print(f"Error: Ticker {ticker} in level 0 but xs failed.")
                return None
            except Exception as e:
                print(f"Error extracting ticker {ticker} from level 0: {e}")
                return None
        else:
            print(f"Error: Ticker {ticker} not found in MultiIndex "
                  f"columns (levels 0 or 1).")
            print(f"Available columns: {processed_df.columns}")
            return None
    else:
        # Assume single index columns, convert to lowercase
        processed_df.columns = processed_df.columns.str.lower()

    # --- Basic Preprocessing Steps ---
    # 1. Handle missing values (example: forward fill)
    initial_nan_count = processed_df.isnull().sum().sum()
    processed_df.ffill(inplace=True)
    # Handle any remaining NaNs at the beginning (e.g., drop rows)
    processed_df.dropna(inplace=True)
    final_nan_count = processed_df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Handled NaNs (Initial: {initial_nan_count}, "
              f"Final: {final_nan_count})")

    # 2. Ensure correct data types (e.g., numeric for OHLCV)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(
                processed_df[col], errors='coerce'
            )
        else:
            print(f"Warning: Column '{col}' not found for type conversion.")
    processed_df.dropna(inplace=True)  # Drop rows if coercion failed

    print(f"Initial preprocessing complete. Shape: {processed_df.shape}")
    return processed_df