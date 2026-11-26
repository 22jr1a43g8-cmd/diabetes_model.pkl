import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load your dataset (replace 'diabetes.csv' with your actual CSV path)
df = pd.read_csv("diabetes.csv")

# Replace impossible zero values with NaN (matching your Streamlit app)
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_columns] = df[zero_columns].replace(0, np.nan)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Build and fit the pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# 3. Save the trained model
joblib.dump(model, "diabetes_logistic_model.pkl")
print("Saved diabetes_logistic_model.pkl successfully!")