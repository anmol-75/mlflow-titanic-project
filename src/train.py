import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/train.csv")

# -----------------------------
# 2. BASIC CLEANING / FEATURE PREP
# -----------------------------
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# -----------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. SET UP MLFLOW
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-survival-prediction")

# -----------------------------
# 5. DEFINE EXPERIMENTS TO RUN
# -----------------------------
experiments = [
    {
        "run_name": "random-forest-100",
        "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 5}
    },
    {
        "run_name": "random-forest-200-deep",
        "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 8}
    },
    {
        "run_name": "logistic-regression",
        "model": LogisticRegression(max_iter=1000),
        "params": {"model_type": "LogisticRegression", "max_iter": 1000}
    },
    {
        "run_name": "decision-tree",
        "model": DecisionTreeClassifier(max_depth=5, random_state=42),
        "params": {"model_type": "DecisionTree", "max_depth": 5}
    },
]

# -----------------------------
# 6. RUN EACH EXPERIMENT
# -----------------------------
for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):

        for key, value in exp["params"].items():
            mlflow.log_param(key, value)

        model = exp["model"]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"\n--- {exp['run_name']} ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

print("\nAll runs finished. Launch 'mlflow ui' to compare them.")