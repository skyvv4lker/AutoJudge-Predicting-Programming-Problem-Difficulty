from flask import Flask, render_template, request
import os

from src.utils import load_models, predict_difficulty

app = Flask(__name__)

# Load models once when app starts
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

tfidf, classifier, regressor = load_models(MODELS_DIR)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        description = request.form.get("description", "")
        input_desc = request.form.get("input_description", "")
        output_desc = request.form.get("output_description", "")

        pred_class, pred_score = predict_difficulty(
            description,
            input_desc,
            output_desc,
            tfidf,
            classifier,
            regressor
        )

        prediction = {
            "class": pred_class,
            "score": pred_score
        }

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
