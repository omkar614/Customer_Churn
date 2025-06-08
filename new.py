import pandas as pd
import joblib

# Load the dataset you used for training
df = pd.read_csv("tel_churn.csv")

df = df.drop(columns=['Churn', 'Unnamed: 0'], errors='ignore')


# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# One-hot encode categorical variables
df_processed = pd.get_dummies(df)

# Save the column structure used in training
joblib.dump(df_processed.columns.tolist(), "model_columns.pkl")

print("Saved model_columns.pkl with", len(df_processed.columns), "columns.")
