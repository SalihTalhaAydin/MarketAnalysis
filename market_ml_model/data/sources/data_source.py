from enum import Enum


class DataSource(str, Enum):
    """Enum for supported data sources."""

    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    CSV = "csv"
    CRYPTO = "crypto"
    DATAREADER = "datareader"
    # Add other sources as needed
