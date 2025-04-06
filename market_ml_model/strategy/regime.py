# market_ml_model/strategy/regime.py
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not installed. Visualization unavailable.")
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

# Import clustering if needed for regime detection
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE_FOR_REGIME = True
except ImportError:
    logger.warning("scikit-learn not available for clustering regime detection.")
    SKLEARN_AVAILABLE_FOR_REGIME = False

    class KMeans:
        pass  # Dummy class

    class StandardScaler:
        pass  # Dummy class


# --- Market Regime Detection ---


class MarketRegimeDetector:
    """Detects market regimes using various indicators."""

    def __init__(
        self,
        n_regimes: int = 3,  # e.g., Bull, Bear, Sideways
        lookback_window: int = 60,  # Lookback for calculating features
        regime_features: Optional[List[str]] = None,  # Features used for detection
        use_clustering: bool = True,  # Use KMeans for detection
        random_state: int = 42,  # Added random_state for reproducibility
    ):
        """
        Initialize market regime detector.

        Args:
            n_regimes: Number of regimes to detect
            lookback_window: Lookback window for detection features
            regime_features: List of feature names to use (e.g., ['volatility_50', 'sma_slope'])
            use_clustering: Whether to use clustering (KMeans) or rule-based detection
            random_state: Random state for KMeans initialization
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        # Default features if none provided
        self.regime_features = regime_features or [
            "volatility_ratio_20_50",
            "efficiency_ratio_20",
            "adf_pvalue",  # Example features
        ]
        self.use_clustering = use_clustering
        self.random_state = random_state  # Store random state
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE_FOR_REGIME else None
        self.kmeans = (
            KMeans(
                n_clusters=n_regimes, random_state=self.random_state, n_init=10
            )  # Use stored random_state
            if SKLEARN_AVAILABLE_FOR_REGIME
            else None
        )
        self.is_fitted = False
        self.regime_history = []  # Store history for analysis/plotting

    def detect_regime(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect market regime for the latest timestamp in the provided data.
        Requires engineered features to be present in the input DataFrame.

        Args:
            data: DataFrame containing the required regime_features, indexed by time.

        Returns:
            Dictionary with regime information for the latest timestamp, or None if error.
            Example: {'regime': 0, 'regime_name': 'Trending', 'confidence': 0.8, 'timestamp': ...}
        """
        if data is None or data.empty:
            logger.warning("No data provided for regime detection.")
            return None

        missing_features = [f for f in self.regime_features if f not in data.columns]
        if missing_features:
            logger.error(
                f"Missing required features for regime detection: {missing_features}"
            )
            return None

        # Select and clean features for the lookback window ending at the last timestamp
        # Ensure enough data points for lookback
        if len(data) < self.lookback_window:
            logger.warning(
                f"Not enough data ({len(data)}) for regime lookback window ({self.lookback_window})."
            )
            # Optionally fall back to rule-based or return None
            return None  # Cannot reliably detect regime

        # Use data up to the latest point for detection/fitting
        regime_data = data[self.regime_features].dropna()

        # Need sufficient non-NaN data points after dropping NaNs
        if len(regime_data) < self.lookback_window:
            logger.warning(
                f"Not enough non-NaN data ({len(regime_data)}) in lookback window for regime detection."
            )
            return None

        if self.use_clustering:
            regime_info = self._detect_clustering_regime(regime_data)
        else:
            # Implement or call rule-based detection if needed
            logger.warning(
                "Rule-based regime detection not implemented yet. Using clustering."
            )
            regime_info = self._detect_clustering_regime(
                regime_data
            )  # Fallback to clustering

        # Store history
        if regime_info:
            self.regime_history.append(regime_info)
            # Limit history length
            max_history = 2000  # Keep more history if needed
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

        return regime_info

    def _detect_clustering_regime(self, regime_data: pd.DataFrame) -> Optional[Dict]:
        """Detect regimes using KMeans clustering for the latest data point."""
        if (
            not SKLEARN_AVAILABLE_FOR_REGIME
            or self.scaler is None
            or self.kmeans is None
        ):
            logger.error("Clustering requires scikit-learn.")
            return None

        try:
            # Scale features (fit scaler only once or periodically, transform here)
            # For simplicity, fit_transform each time, but could be optimized
            scaled_data = self.scaler.fit_transform(regime_data)

            # Fit KMeans model if not already fitted or periodically refit
            if not self.is_fitted:
                logger.info("Fitting KMeans model for regime detection...")
                self.kmeans.fit(scaled_data)
                self.is_fitted = True
                logger.info(
                    f"KMeans model fitted. Cluster centers:\n{self.kmeans.cluster_centers_}"
                )

            # Predict regime for the latest data point
            latest_scaled_features = scaled_data[-1:, :]
            regime = self.kmeans.predict(latest_scaled_features)[0]

            # Calculate confidence (e.g., distance to cluster center)
            distances = self.kmeans.transform(latest_scaled_features)
            min_distance = distances[0, regime]
            # Normalize confidence (optional, simple approach)
            confidence = 1.0 / (1.0 + min_distance)  # Closer = higher confidence

            # Map cluster label to regime name (requires interpretation)
            # This mapping needs to be defined based on cluster properties
            # Example placeholder mapping:
            regime_names = {
                0: "Regime A",
                1: "Regime B",
                2: "Regime C",
            }  # Needs proper mapping
            regime_name = regime_names.get(regime, f"Unknown Regime {regime}")

            regime_info = {
                "regime": int(regime),
                "regime_name": regime_name,
                "confidence": float(confidence),
                "features": regime_data.iloc[
                    -1
                ].to_dict(),  # Features for the detected point
                "timestamp": regime_data.index[-1],
            }
            return regime_info

        except Exception as e:
            logger.exception(f"Error during clustering regime detection: {e}")
            return None

    # Placeholder for rule-based detection
    def _detect_rule_based_regime(self, data: pd.DataFrame) -> Optional[Dict]:
        logger.error("Rule-based regime detection not implemented.")
        # Example structure:
        # latest_features = data.iloc[-1]
        # if latest_features['volatility_ratio'] > 1.5: regime = 2 ...
        # return {'regime': regime, ...}
        return None

    def plot_regime_history(
        self,
        price_data: Optional[pd.Series] = None,  # Pass close price Series
        figsize: Tuple[int, int] = (15, 8),
        filename: Optional[str] = None,
    ):
        """Plot the detected market regimes over time."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available.")
            return
        if not self.regime_history:
            logger.warning("No regime history to plot.")
            return

        try:
            # Create DataFrame from regime history
            regime_df = pd.DataFrame(self.regime_history)
            if (
                "timestamp" not in regime_df.columns
                or regime_df["timestamp"].isnull().all()
            ):
                logger.warning("No valid timestamps in regime history for plotting.")
                return
            regime_df = regime_df.dropna(subset=["timestamp"]).set_index("timestamp")
            regime_df.index = pd.to_datetime(regime_df.index)  # Ensure datetime index

            plt.figure(figsize=figsize)
            ax1 = plt.gca()
            ax2 = ax1.twinx()  # Create secondary y-axis for regimes

            # Plot price data if available
            if price_data is not None and not price_data.empty:
                # Align price data to regime index
                price_data_aligned = price_data.reindex(regime_df.index).ffill()
                ax1.plot(
                    price_data_aligned.index,
                    price_data_aligned,
                    label="Price",
                    color="black",
                    alpha=0.6,
                    linewidth=1,
                )
                ax1.set_ylabel("Price")
                ax1.legend(loc="upper left")
            else:
                ax1.set_ylabel("Regime Value (if no price)")  # Adjust label if no price

            # Plot regimes as steps
            unique_regimes = sorted(regime_df["regime"].unique())

            # Use step plot for regimes
            ax2.plot(
                regime_df.index,
                regime_df["regime"],
                drawstyle="steps-post",
                label="Detected Regime",
                color="blue",
                linewidth=1.5,
            )

            ax2.set_ylabel("Market Regime")
            ax2.set_yticks(unique_regimes)
            # Optional: Add regime names as labels if meaningful mapping exists
            # regime_name_map = {r: info['regime_name'] for r, info in regime_df.groupby('regime').first().iterrows()}
            # ax2.set_yticklabels([regime_name_map.get(r, r) for r in unique_regimes])
            ax2.set_ylim(min(unique_regimes) - 0.5, max(unique_regimes) + 0.5)
            ax2.legend(loc="upper right")

            plt.title("Market Regime Detection History")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()

            if filename:
                try:
                    # Ensure directory exists
                    output_dir = os.path.dirname(filename)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    logger.info(f"Regime history plot saved to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save regime plot to {filename}: {e}")

            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.exception(f"Error plotting regime history: {e}")
            plt.close()  # Ensure plot is closed
