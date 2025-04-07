"""
Utility functions for feature engineering.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def filter_correlated_features(
    df: pd.DataFrame, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Removes highly correlated features from a DataFrame.

    Iteratively identifies pairs of features with an absolute correlation
    coefficient greater than the threshold and removes one feature from each pair.
    The feature to remove is chosen based on the alphabetical order of the column names.

    Args:
        df: DataFrame containing features (numeric columns only).
            Non-numeric columns will be ignored.
        threshold: The absolute correlation threshold (between 0 and 1).
                   Features with correlation above this value will be considered for removal.

    Returns:
        DataFrame with highly correlated features removed.
    """
    if df is None or df.empty:
        logger.warning(
            "Input DataFrame for correlation filtering is empty. Returning as is."
        )
        return df
    if not isinstance(df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame.")
        return df  # Or raise error

    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        logger.warning(
            "No numeric columns found for correlation filtering. Returning original DataFrame."
        )
        return df

    logger.info(
        f"Starting correlation filtering on {len(numeric_df.columns)} numeric features with threshold {threshold}."
    )

    # Calculate the absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Create a mask for the upper triangle
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_triangle = corr_matrix.where(upper_triangle_mask)

    # Find features with correlation above the threshold
    to_drop = set()
    # Iterate through columns checking correlations
    for col in upper_triangle.columns:
        if col in to_drop:  # Skip if already marked for removal
            continue
        # Find correlations for this column that are above the threshold
        correlated_cols = upper_triangle.index[upper_triangle[col] > threshold].tolist()
        # Add the correlated columns (except the current one if it was somehow included)
        for correlated_col in correlated_cols:
            if correlated_col not in to_drop:  # Only add if not already marked
                # Keep the column that comes first alphabetically
                if col < correlated_col:
                    to_drop.add(correlated_col)
                    logger.debug(
                        f"Marking '{correlated_col}' for removal (correlated with '{col}' > {threshold})"
                    )
                else:
                    to_drop.add(col)
                    logger.debug(
                        f"Marking '{col}' for removal (correlated with '{correlated_col}' > {threshold})"
                    )
                    break  # Move to the next column as 'col' is now marked

    if to_drop:
        logger.info(
            f"Identified {len(to_drop)} features to remove due to high correlation: {sorted(list(to_drop))}"
        )
        # Drop the identified features from the original DataFrame (including non-numeric)
        df_filtered = df.drop(columns=list(to_drop))
        logger.info(f"DataFrame shape after correlation filtering: {df_filtered.shape}")
        return df_filtered
    else:
        logger.info(
            "No features found with correlation above the threshold. No features removed."
        )
        return df
