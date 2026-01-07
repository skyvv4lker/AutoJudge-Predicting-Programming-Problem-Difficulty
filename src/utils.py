import joblib
import os
import re


def clean_text(text):
    """
    Same cleaning logic used during training
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_models(models_dir):
    """
    Load trained models and vectorizer
    """
    tfidf = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
    classifier = joblib.load(os.path.join(models_dir, "classifier.pkl"))
    regressor = joblib.load(os.path.join(models_dir, "regressor.pkl"))

    return tfidf, classifier, regressor


def predict_difficulty(
    description,
    input_desc,
    output_desc,
    tfidf,
    classifier,
    regressor
):
    """
    Generate predictions for difficulty class and score
    """
    combined_text = " ".join([
        clean_text(description),
        clean_text(input_desc),
        clean_text(output_desc)
    ])

    X = tfidf.transform([combined_text])

    predicted_class = classifier.predict(X)[0]
    predicted_score = regressor.predict(X)[0]

    return predicted_class, round(float(predicted_score), 2)
