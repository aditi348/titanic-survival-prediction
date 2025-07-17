# src/model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from src.preprocessing import load_data, preprocess_data

def train_model():
    # Load and preprocess the data
    train_df, _ = load_data()
    train_cleaned = preprocess_data(train_df.copy())

    # Separate features and target
    X = train_cleaned.drop("Survived", axis=1)
    y = train_cleaned["Survived"]

    # Train/test split (for evaluation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n Validation Accuracy: {acc:.4f}\n")

    print(" Classification Report:")
    print(classification_report(y_val, y_pred))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save the model
    joblib.dump(model, "models/random_forest_titanic.pkl")
    print("\n Model saved to models/random_forest_titanic.pkl")

if __name__ == "__main__":
    train_model()
