# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    return train_df, test_df

def preprocess_data(df):
    # Fix missing Age with median (safe way)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fix missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # For test set: fix missing Fare
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop columns that won't help prediction
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    return df

if __name__ == "__main__":
    train_df, test_df = load_data()
    print("Before preprocessing:")
    print(train_df.head())

    train_cleaned = preprocess_data(train_df.copy())
    print("After preprocessing:")
    print(train_cleaned.head())
