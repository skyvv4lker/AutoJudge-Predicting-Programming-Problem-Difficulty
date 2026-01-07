import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import extract_features


def train_regressor(csv_path):
    """
    Train a regression model to predict numerical difficulty score
    """
    # Extract features and labels
    X, _, y_score, _ = extract_features(csv_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_score,
        test_size=0.2,
        random_state=42
    )

    # Initialize regressor
    regressor = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )


    # Train model
    regressor.fit(X_train, y_train)

    # Predict on test set
    y_pred = regressor.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5


    print("Regression Results:")
    print("MAE :", mae)
    print("RMSE:", rmse)

    return regressor


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "task_complexity.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

    # Create models directory if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Train regressor
    regressor = train_regressor(DATA_PATH)

    # Save trained model
    regressor_path = os.path.join(MODELS_DIR, "regressor.pkl")
    joblib.dump(regressor, regressor_path)

    print("\nRegressor model saved at:", regressor_path)
