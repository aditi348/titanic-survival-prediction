# main.py

from src.preprocessing import load_data, preprocess_data
from src.model import train_model

def main():
    print(" Starting Titanic Survival Prediction Project")

    # Load & Preview Raw Data
    train_df, test_df = load_data()
    print(f" Loaded training data with shape: {train_df.shape}")
    print(f"Loaded test data with shape: {test_df.shape}")

    # Preprocess + Train model (everything inside train_model)
    train_model()

    print(" Pipeline completed successfully!")

if __name__ == "__main__":
    main()
