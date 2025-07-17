# Titanic Survival Prediction

## Project Overview
This project predicts which passengers survived the Titanic shipwreck using machine learning. It involves data preprocessing, feature engineering, model training, evaluation, and prediction generation.

## Dataset
The data comes from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

## Features Used
- Pclass (Passenger class)
- Sex
- Age
- SibSp (Number of siblings/spouses aboard)
- Parch (Number of parents/children aboard)
- Fare
- Embarked (Port of Embarkation)

## Steps Implemented
1. **Data Loading:** Load train and test datasets.
2. **Preprocessing:** Handle missing values and encode categorical variables.
3. **Model Training:** Train a Random Forest Classifier.
4. **Evaluation:** Evaluate with accuracy, precision, recall, and confusion matrix.
5. **Prediction:** Predict survival on test data and save results as a CSV for submission.

## How to Run
1. Clone this repository.
2. Create a virtual environment and activate it.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
