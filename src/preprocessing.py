import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(path: str):
    df = pd.read_csv(path)

    # Dividir en Train (70%) y Test (30%)
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # Dividir Train en Train (70%) y Validation (15%)
    train, val = train_test_split(train, test_size=0.15, random_state=42)

    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print("Data preprocessed and saved.")

if __name__ == "__main__":
    load_and_split_data('data/raw/dataset.csv')