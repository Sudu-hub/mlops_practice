import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#create a logger 
#logger = logging.getLogger('data_ingestion_logger')
# After using logger we set level DEBUG (1st level)
#logger.setLevel('DEBUG')

#create a handller
#console_handler = logging.StreamHandler() # -> MASSAGE PRINT ON CONSOLE
#console_handler.setLevel('DEBUG')

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#console_handler.setFormatter(formatter)

#logger.addHandler(console_handler)

#logger.debug('This is a DEBUG message')
#logger.info('This an info message')
#logger.warning('This is a warning')
#logger.error('This is an error')
#logger.critical('This is a critical message')

def load_params(params_yaml):
    """Load max_features from YAML config file."""
    try:
        with open(params_yaml,'r') as file:
            params  = yaml.safe_load(file)
        test_size =  params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    
    except FileNotFoundError:
        logger.error('file not found')
        raise
    
    except ValueError:
        logger.error('Error:unexpected')
        raise

    except Exception as e:
        logger.error('some error is occured')
        raise
        #return 0.2

def load_data(url: str) -> pd.DataFrame:
    """Loads data from a given URL and handles exceptions."""
    try:
        df = pd.read_csv(url)
        logger.debug('data is retrived')
        return df
    except Exception as e:
        logger.error('has some error occured')
        raise
        # return pd.DataFrame()  # Return empty DataFrame to prevent crash

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the DataFrame by filtering and encoding sentiment labels."""
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot process data.")
        
        df = df.drop(columns=['tweet_id'], errors='ignore')  # Ignore KeyError if column is missing
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logger.debug('final_df is retrived')
        return final_df
    except Exception as e:
        logger.error('has some error occured')
        # return pd.DataFrame()  # Return empty DataFrame to prevent crash

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Saves train and test data to CSV files."""
    try:
        if train_data.empty or test_data.empty:
            raise ValueError("Train or test data is empty. Skipping saving.")

        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info('data saved succesfully')
    except Exception as e:
        logger.error('has some error occured')
        raise

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
        logger.error('has some error occured')
        raise

if __name__ == '__main__':
    main()

