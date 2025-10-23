import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import joblib  # Used for saving and loading scikit-learn models
from dotenv import load_dotenv
load_dotenv()


mlflow.set_experiment("Telco Churn Prediction")
df = pd.read_csv('data/preprocessed_churn.csv')
X = df.drop('Churn', axis=1); y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Logistic Regression Baseline"):
 mlflow.set_tag("model_type", "Logistic Regression")
 params = {"solver": "liblinear", "random_state": 42}
 mlflow.log_params(params)
 
 lr = LogisticRegression(**params).fit(X_train, y_train)
 
 y_pred = lr.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
 
 mlflow.log_metric("accuracy", accuracy)
 mlflow.log_metric("auc", auc)
 
 # Save the model using joblib
 joblib.dump(lr, "model.joblib")
 mlflow.log_artifact("model.joblib")