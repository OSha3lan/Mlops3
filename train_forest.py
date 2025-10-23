from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import joblib  # Used for saving and loading scikit-learn models


mlflow.set_experiment("Telco Churn Prediction")
df = pd.read_csv('data/preprocessed_churn.csv')
X = df.drop('Churn', axis=1); y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Random Forest Baseline"):
 mlflow.set_tag("model_type", "Random Forest")
 params = {"n_estimators": 100, "random_state": 42}
 mlflow.log_params(params)
 
 rf = RandomForestClassifier(**params).fit(X_train, y_train)
 
 y_pred = rf.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
 
 mlflow.log_metric("accuracy", accuracy)
 mlflow.log_metric("auc", auc)
 
 # Save the model using joblib
 joblib.dump(rf, "model.joblib")
 mlflow.log_artifact("model.joblib")