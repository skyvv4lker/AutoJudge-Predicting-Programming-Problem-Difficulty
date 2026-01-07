import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import preprocess_data


def extract_features(csv_path):
    """
    Preprocess data and extract TF-IDF features
    """
    # Load and preprocess data
    df = preprocess_data(csv_path)

    texts = df["combined_text"].values
    y_class = df["problem_class"].values
    y_score = df["problem_score"].values

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    return X, y_class, y_score, vectorizer


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "task_complexity.csv")
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y_class, y_score, vectorizer = extract_features(DATA_PATH)

    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    joblib.dump(vectorizer, tfidf_path)

    print("Feature extraction completed successfully")
    print("TF-IDF feature shape:", X.shape)
    print("TF-IDF model saved at:", tfidf_path)
