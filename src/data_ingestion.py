import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

def load_params(params_path):
    test_size = yaml.safe_load(open(params_path,'r'))['data_ingestion']['test_size']
    return test_size

def load_data(url):
    df = pd.read_csv(url)
    return df

def process_data(df):
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df = final_df.copy()  # Ensure you're modifying the original DataFrame
    final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
    return final_df

def save_data(data_path, train_data, test_data):
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path,'train.csv'))
    test_data.to_csv(os.path.join(data_path,'test.csv'))


def main():
    test_size = load_params('params.yaml')
    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    data_path = os.path.join('data','raw')
    save_data(data_path, train_data, test_data)

if __name__ == '__main__':
    main()
