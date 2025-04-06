# market_ml_model/models/scheduler.py
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Import necessary components from other modules
from .predictor import ModelPredictorBase, get_confidence_levels, predict_with_threshold

# Assuming DataLoader and engineer_features are accessible, adjust imports if needed
# from ..data.loaders import DataLoader
# from ..features.features_engineering import engineer_features

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Class to schedule and run predictions at specified intervals."""

    def __init__(
        self,
        predictor: ModelPredictorBase,
        data_loader: Any,  # Should be DataLoader instance from data.loaders
        feature_engineer: Any,  # Should be engineer_features function
        interval_minutes: int = 60,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the Prediction Scheduler.

        Args:
            predictor: Fitted ModelPredictorBase instance
            data_loader: Instance of DataLoader to fetch new data
            feature_engineer: Function to engineer features
            interval_minutes: Prediction interval in minutes
            output_dir: Directory to save prediction results
        """
        self.predictor = predictor
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.interval = timedelta(minutes=interval_minutes)
        self.output_dir = output_dir
        self.last_prediction_time = None
        self.scheduler_active = False

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def predict_once(
        self,
        ticker: str,
        start_date: Optional[
            str
        ] = None,  # For fetching historical data needed for features
        end_date: Optional[str] = None,  # Typically now
    ) -> Optional[pd.DataFrame]:
        """
        Fetch latest data, engineer features, and make a single prediction.

        Args:
            ticker: Asset ticker symbol
            start_date: Start date for historical data fetch
            end_date: End date for data fetch (defaults to now)

        Returns:
            DataFrame with the latest prediction(s), or None if error
        """
        try:
            logger.info(f"Running single prediction for {ticker}...")
            # Fetch latest data (need enough history for feature calculation)
            # Adjust start/end dates based on feature requirements
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Fetch slightly more data than needed for feature lookbacks
            # This needs knowledge of the feature engineer lookback periods
            # Placeholder: fetch last 200 periods assuming max lookback is ~200
            if start_date is None:
                # Heuristic: go back enough days/hours based on interval
                # This is complex, needs better handling based on actual feature lookbacks
                # Defaulting to a fixed lookback for now
                # This needs the data_loader's interval knowledge
                # start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_periods)).strftime('%Y-%m-%d') # Simplistic daily lookback
                logger.warning(
                    "Start date not provided, fetching recent data. Feature accuracy may be affected."
                )
                # Use data_loader's defaults if possible - requires DataLoader to have config/defaults
                # start_date = self.data_loader.config.default_start_date # Example if DataLoader has config
                # Fallback: Use a reasonable default lookback if loader doesn't provide one
                start_date = (pd.to_datetime(end_date) - timedelta(days=90)).strftime(
                    "%Y-%m-%d"
                )  # Example: 90 days

            # Assuming data_loader has a method like load_data or similar
            # Adjust the call based on the actual DataLoader interface
            raw_data = self.data_loader.load_data(
                symbol=ticker, start_date=start_date, end_date=end_date
            )
            if raw_data is None or raw_data.empty:
                logger.error(f"Failed to load data for {ticker}")
                return None

            # Engineer features
            # Assume feature_engineer function takes the raw data DataFrame and maybe config
            # Adjust call based on actual feature_engineer interface
            features_df, _ = self.feature_engineer(
                raw_data
            )  # Assuming it returns features, target
            if features_df is None or features_df.empty:
                logger.error(f"Failed to engineer features for {ticker}")
                return None

            # Get latest features
            latest_features = features_df.iloc[[-1]]  # Get last row as DataFrame

            # Predict probabilities
            probabilities, class_names = self.predictor.predict_proba(latest_features)
            if probabilities is None:
                logger.error(f"Failed to generate probabilities for {ticker}")
                return None

            # Create result DataFrame
            result_df = pd.DataFrame(
                probabilities,
                index=latest_features.index,
                columns=[
                    f"probability_{c}" for c in class_names
                ],  # Ensure column names match
            )
            # Determine positive class index for thresholding
            positive_class_label = "1"  # Assuming '1' is positive
            try:
                pos_idx = class_names.index(positive_class_label)
            except ValueError:
                pos_idx = 1  # Default if '1' not found

            result_df["prediction"] = predict_with_threshold(
                probabilities, positive_class_index=pos_idx
            )
            result_df["confidence"] = get_confidence_levels(probabilities)

            logger.info(
                f"Prediction for {ticker} at {result_df.index[0]}: {result_df.iloc[0].to_dict()}"
            )

            # Save result if output dir specified
            if self.output_dir:
                self._save_prediction_results(ticker, result_df)

            return result_df

        except Exception as e:
            logger.exception(f"Error during single prediction for {ticker}: {e}")
            return None

    def predict_batch(
        self, tickers: List[str], **kwargs
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Run predict_once for a batch of tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.predict_once(ticker, **kwargs)
        return results

    def start_scheduler(self, tickers_to_predict: List[str]):  # Pass tickers to predict
        """Start the prediction scheduler loop."""
        if not self.scheduler_active:
            logger.info(
                f"Starting prediction scheduler with interval {self.interval}..."
            )
            self.scheduler_active = True
            # Run immediately first time? Optional.
            # logger.info("Running initial prediction cycle...")
            # self.predict_batch(tickers_to_predict)
            # self.last_prediction_time = datetime.now()
            self._run_scheduler(tickers_to_predict)  # Start the loop
        else:
            logger.warning("Scheduler already active.")

    def stop_scheduler(self):
        """Stop the prediction scheduler."""
        if self.scheduler_active:
            logger.info("Stopping prediction scheduler...")
            self.scheduler_active = False
        else:
            logger.warning("Scheduler not active.")

    def _run_scheduler(self, tickers: List[str]):
        """Internal scheduler loop."""
        while self.scheduler_active:
            now = datetime.now()
            should_run = False
            if self.last_prediction_time is None:
                should_run = True  # Run the first time
            else:
                time_since_last = now - self.last_prediction_time
                if time_since_last >= self.interval:
                    should_run = True

            if should_run:
                logger.info(
                    f"Scheduler triggered at {now}. Running predictions for {tickers}..."
                )
                self.predict_batch(tickers)
                self.last_prediction_time = now
                logger.info("Prediction cycle complete.")

            # Sleep until next check
            # Calculate sleep time more accurately to align with interval
            if self.last_prediction_time:
                next_run_time = self.last_prediction_time + self.interval
                sleep_seconds = max(0, (next_run_time - datetime.now()).total_seconds())
            else:
                # If first run hasn't happened, check reasonably frequently
                sleep_seconds = min(
                    60, self.interval.total_seconds()
                )  # Check every minute or interval, whichever is smaller

            # Check frequently if stopping soon, especially for short intervals
            check_interval = (
                min(10, sleep_seconds) if sleep_seconds > 0 else 1
            )  # Check every 10s or less

            # Sleep in smaller chunks to allow faster stopping
            end_sleep_time = time.time() + sleep_seconds
            while time.time() < end_sleep_time:
                if not self.scheduler_active:
                    break  # Exit inner loop if stopped
                time.sleep(min(check_interval, end_sleep_time - time.time()))

            if not self.scheduler_active:
                break  # Exit outer while loop if stopped

    def _save_prediction_results(self, ticker: str, results: pd.DataFrame):
        """Save prediction results to a file."""
        if not self.output_dir or results.empty:
            return
        try:
            # Use timestamp from the prediction index
            timestamp_str = results.index[0].strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_prediction_{timestamp_str}.csv"
            filepath = os.path.join(self.output_dir, filename)
            results.to_csv(filepath)
            logger.info(f"Saved prediction results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save prediction results for {ticker}: {e}")
