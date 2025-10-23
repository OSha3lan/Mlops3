import pandas as pd
import sys
import os
from dotenv import load_dotenv
load_dotenv()
 
 # URL for the Telco Customer Churn dataset
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
 
 # Load data
try:
 df = pd.read_csv(url)
 print("Telco Customer Churn dataset loaded successfully!")
except Exception as e:
    print(f"Error loading data from URL: {e}")
    sys.exit(1)

 # Preprocessing Steps
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
categorical_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_processed.drop('customerID', axis=1, inplace=True)
 
 # Save the preprocessed file
os.makedirs('data', exist_ok=True)
df_processed.to_csv('data/preprocessed_churn.csv', index=False)
print("Preprocessing complete and file saved.")