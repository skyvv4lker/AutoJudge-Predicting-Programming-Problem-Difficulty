import pandas as pd
import re
import os


def clean_text(text):
    """
    Clean text by:
    - converting to lowercase
    - removing special characters
    - removing extra spaces
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_to_class(score):
    """
    Convert numerical score to difficulty class
    (used only if problem_class is missing)
    """
    if score <= 3:
        return "Easy"
    elif score <= 6:
        return "Medium"
    else:
        return "Hard"


def preprocess_data(csv_path):
    """
    Load CSV, clean text, combine fields, and prepare labels
    """
    df = pd.read_csv(csv_path)

    # Normalize class labels if present
    if "problem_class" in df.columns:
        df["problem_class"] = (
            df["problem_class"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.capitalize()
        )

    # Ensure required columns exist
    required_columns = [
        "description",
        "input_description",
        "output_description",
        "problem_score"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    # Clean text fields
    df["description"] = df["description"].apply(clean_text)
    df["input_description"] = df["input_description"].apply(clean_text)
    df["output_description"] = df["output_description"].apply(clean_text)

    # Combine all text into one field
    df["combined_text"] = (
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    ).str.strip()

    # Convert score column to numeric
    df["problem_score"] = pd.to_numeric(df["problem_score"], errors="coerce")
    df = df.dropna(subset=["problem_score"])

    # Create classification label ONLY if not present
    if "problem_class" not in df.columns or df["problem_class"].isna().all():
        df["problem_class"] = df["problem_score"].apply(score_to_class)

    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "task_complexity.csv")

    df = preprocess_data(DATA_PATH)

    print("Preprocessing completed successfully")
    print("Total samples:", len(df))
    print(df[["problem_score", "problem_class"]].head())
