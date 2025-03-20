import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params(params_yaml):
    """Load max_features from YAML config file."""
    try:
        with open(params_yaml,'r') as file:
            params  = yaml.safe_load(file)
    except FileNotFoundError:
        print('File not found')
    
    except ValueError:
        print('please add float value')
    
    except Exception as e:
        print(e)
    else:
        return params['data_ingestion']['test_size']
        #return 0.2

def load_data(url: str) -> pd.DataFrame:
    """Loads data from a given URL and handles exceptions."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error loading data from URL '{url}': {e}")
        # return pd.DataFrame()  # Return empty DataFrame to prevent crash

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the DataFrame by filtering and encoding sentiment labels."""
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot process data.")
        
        df = df.drop(columns=['tweet_id'], errors='ignore')  # Ignore KeyError if column is missing
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        return final_df
    except Exception as e:
        print(f"Error processing data: {e}")
        # return pd.DataFrame()  # Return empty DataFrame to prevent crash

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Saves train and test data to CSV files."""
    try:
        if train_data.empty or test_data.empty:
            raise ValueError("Train or test data is empty. Skipping saving.")

        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """Main execution function."""
    try:
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        
        if df.empty:
            raise ValueError("Failed to load data. Exiting program.")

        final_df = process_data(df)
        if final_df.empty:
            raise ValueError("Processed data is empty. Exiting program.")

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()

