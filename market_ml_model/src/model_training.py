import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


def train_classification_model(features: pd.DataFrame, target: pd.Series):
    """Trains a classification model (e.g., RandomForestClassifier).

    Uses the provided features and target data to train a model suitable
    for predicting market direction or other classification tasks.

    Args:
        features: DataFrame of input features for training.
        target: Series containing the target variable (e.g., 0 or 1) for
                training.

    Returns:
        The trained scikit-learn model object, or None if training fails.
    """
    print("--- Training Classification Model ---")
    if features.empty or target.empty:
        print("Error: Features or target data is empty for training.")
        return None
    if len(features) != len(target):
        print(f"Error: Features ({len(features)}) and target ({len(target)}) "
              "length mismatch.")
        return None

    print(f"Training with Features shape: {features.shape}")
    print(f"Training with Target shape: {target.shape}")

    # Check for NaNs before training
    if features.isnull().any().any() or target.isnull().any():
        print("Error: NaN values detected in features or target for training.")
        return None

    # Instantiate the model (example hyperparameters)
    # Consider tuning these parameters later (e.g., using GridSearchCV)
    model = RandomForestClassifier(
        n_estimators=100,     # Number of trees in the forest
        max_depth=10,         # Maximum depth of the trees
        min_samples_split=50,    # Min samples required to split internal node
        min_samples_leaf=25,     # Min samples required to be at a leaf node
        class_weight='balanced',   # Adjust weights for class imbalance
        random_state=42,      # For reproducibility
        n_jobs=-1             # Use all available CPU cores
    )

    try:
        # Train the model
        model.fit(features, target)
        print("Model training complete.")
        # Optional: Check if the model is fitted (basic sanity check)
        model.predict(features.iloc[[0]])  # Try predicting on one sample
        return model
    except NotFittedError:
        # This check might be redundant if fit() succeeded, but good practice.
        print("Error: Model reported as not fitted after training attempt.")
        return None
    except ValueError as ve:
        print(f"Error during model training (ValueError): {ve}")
        # Common issues: NaNs, inf values, non-numeric data
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return None