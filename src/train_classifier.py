import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from feature_engineering import extract_features


def train_classifier(csv_path):
    """
    Train a classification model to predict Easy / Medium / Hard
    """
    # Extract features and labels
    X, y_class, _, _ = extract_features(csv_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    # Initialize classifier
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    # Train model
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Classification Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return clf


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "task_complexity.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

    # Create models directory if not exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Train classifier
    classifier = train_classifier(DATA_PATH)

    # Save trained model
    classifier_path = os.path.join(MODELS_DIR, "classifier.pkl")
    joblib.dump(classifier, classifier_path)

    print("\nClassifier model saved at:", classifier_path)
