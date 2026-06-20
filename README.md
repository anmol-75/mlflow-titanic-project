
# Titanic Survival Prediction with MLflow

A beginner machine learning project that uses **MLflow** to track and compare experiments on the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). The goal is to predict passenger survival and learn how to track, compare, and manage ML experiments using MLflow.

## What this project does

- Loads and cleans the Titanic dataset
- Trains and compares 4 different classification models:
  - Random Forest (100 trees)
  - Random Forest (200 trees, deeper)
  - Logistic Regression
  - Decision Tree
- Logs parameters, metrics (accuracy, precision, recall, F1), and trained models for each run using MLflow
- Lets you visually compare all runs in the MLflow UI

## Tech stack

- Python 3.13
- scikit-learn
- MLflow
- pandas / numpy

## Project structure
  mlflow-titanic-project/

├── data/              # Titanic dataset (not pushed to repo, see Setup)

├── src/

│   └── train.py       # Main training + MLflow tracking script

├── requirements.txt

├── .gitignore

└── README.md

## Setup

1. Clone the repo
```bash
   git clone https://github.com/anmol-75/mlflow-titanic-project.git
   cd mlflow-titanic-project
```

2. Create and activate a virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place `train.csv` inside a `data/` folder.

5. Run the training script
```bash
   python src/train.py
```

6. View experiment results in the MLflow UI
```bash
   mlflow ui
```
   Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Results

The script trains and logs 4 models. Open the MLflow UI, select multiple runs, and click **Compare** to see metrics side by side (accuracy, precision, recall, F1 score) and identify the best-performing model.

## Future improvements

- Add hyperparameter tuning (GridSearchCV)
- Try gradient boosting models (XGBoost, LightGBM)
- Register the best model using the MLflow Model Registry
- Build a simple Flask/FastAPI app to serve predictions

