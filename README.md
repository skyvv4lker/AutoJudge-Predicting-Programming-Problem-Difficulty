# AutoJudge – Predicting Programming Problem Difficulty

## Project Overview
Online coding platforms such as Codeforces, CodeChef, and Kattis classify programming problems into difficulty categories like Easy, Medium, and Hard, and often assign a numerical difficulty score. These labels are usually based on human judgment and user feedback.

This is a baseline implementation of AutoJudge, an intelligent system that automatically predicts:
- Problem Difficulty Class (Easy / Medium / Hard)
- Problem Difficulty Score (numerical value on a 0–10 scale)

The predictions are generated using **only the textual description** of a programming problem, without relying on user submissions or historical solving data.

---

## Dataset Used
The model is trained on a dataset containing programming problems with the following attributes:
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical difficulty score)

The original dataset is provided in JSONL format and converted into CSV for preprocessing and training.

---

## Approach

### Data Preprocessing
- Converted all text to lowercase
- Removed special characters and extra whitespace
- Combined the following fields into a single text representation:
  - Problem description
  - Input description
  - Output description
- Ensured valid numerical difficulty scores
- Normalized difficulty class labels

---

### Feature Extraction
- Applied **TF-IDF vectorization** to the combined text
- Configuration:
  - Maximum features: 5000
  - N-gram range: (1, 2)
  - English stopwords removed

TF-IDF converts textual problem descriptions into numerical feature vectors representing term importance.

---

### Models Used

#### Classification Model
- Logistic Regression
- Task: Predict difficulty class (Easy / Medium / Hard)

#### Regression Model
- Random Forest Regressor
- Task: Predict numerical difficulty score (0–10)

Both models are trained independently using the same TF-IDF features.

---

## Evaluation Results

### Classification
- Accuracy: Approximately 48%
- Confusion matrix and classification report were generated to analyze class-wise performance

### Regression
- Mean Absolute Error (MAE): Approximately 1.7
- Root Mean Squared Error (RMSE): Approximately 2.0

These results establishes the model as a baseline difficulty prediction system.

---

## Web Interface
A simple Flask-based web application is provided to demonstrate the model. The interface allows users to:
1. Enter the problem description, input description, and output description
2. Click the **Predict** button
3. View the predicted difficulty class and difficulty score

The interface is intentionally minimal and designed to demonstrate core functionality.

---

## How to Run the Project Locally

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models
Pre-trained models are already included. To retrain the models:
```bash
python src/train_classifier.py
python src/train_regressor.py
```

### Run the Web Application
```bash
python app.py
```
Open the browser and navigate to:
```bash
http://127.0.0.1:5000
```

---

## Project Structure

```
AutoJudge/
├── app.py
├── data/
│   └── task_complexity.csv
├── models/
│   ├── tfidf.pkl
│   ├── classifier.pkl
│   └── regressor.pkl
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── utils.py
├── templates/
│   └── index.html
├── README.md
└── report.pdf
```

### Demo Video Link
```
(Add your demo video link here)
```

---

## Limitations

```
1. The model relies only on textual information and does not account for algorithmic complexity or required implementation skills
2. Some logically simple problems with long descriptions may receive moderate difficulty scores
3. Classification accuracy is limited due to overlapping textual patterns across difficulty categories
```

---

## Conclusion

```
The model serves as a baseline AutoJudge system that demonstrates how textual descriptions alone can be used to estimate programming problem difficulty. While performance is moderate, the system provides a solid foundation for comparison with more advanced models.
```

---

## Author

```
Name: Ammy Sunil Meshram
Enrollment : 22118009 (Fourth Year)
Branch : Metallurgical & Materials Engineering
College : Indian Institute of Technology, Roorkee
```
