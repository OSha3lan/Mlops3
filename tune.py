from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import itertools
import os

mlflow.set_experiment("Telco Churn Prediction")
df = pd.read_csv('data/preprocessed_churn.csv')
X = df.drop('Churn', axis=1); y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No parent run - each combination is an independent run (DagsHub compatible)
n_estimators = [50, 100]
max_depth = [5, 10]

best_accuracy = 0
best_run_id = None
best_params = {}

for n, d in itertools.product(n_estimators, max_depth):
    with mlflow.start_run(run_name=f"RF_n{n}_d{d}"):
        clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", n)
        mlflow.log_param("max_depth", d)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("purpose", "Hyperparameter Tuning")

        print(f"n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")
        
        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = mlflow.active_run().info.run_id
            best_params = {"n_estimators": n, "max_depth": d}
            print(f" New best model!")

# Print summary
print(f"\n{'='*60}")
print(f"Best Run ID: {best_run_id}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Params: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}")
print(f"{'='*60}")

