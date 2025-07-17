# src/predict.py

import pandas as pd
import joblib
from preprocessing import load_data, preprocess_data



def predict():
    # Load test data only
    _, test_df = load_data()

    # Keep PassengerId for submission
    passenger_ids = test_df['PassengerId']

    # Preprocess test data
    test_cleaned = preprocess_data(test_df.copy())

    # Load saved model
    model = joblib.load("models/random_forest_titanic.pkl")

    # Predict survival
    predictions = model.predict(test_cleaned)

    # Prepare submission dataframe
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    # Save to CSV
    submission.to_csv("titanic_predictions.csv", index=False)
    print("Predictions saved to titanic_predictions.csv")

if __name__ == "__main__":
    predict()
