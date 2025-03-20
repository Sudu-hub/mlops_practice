import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import yaml

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
        return params['feature_engineering']['max_features']

def preprocess_data(train_data, test_data):
    """Fill NaN values in train and test datasets."""
    try:
        if train_data.empty or test_data.empty:
            raise ValueError('Train_data is empty')
        
        train_data = train_data.fillna('')
        test_data = test_data.fillna('')
        
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        return X_train, y_train, X_test, y_test
    except NameError:
        print('variable is not defined')
    
    except Exception as e:
        print(e)

def vectorize_text(X_train, X_test, max_features):
    """Apply CountVectorizer transformation to train and test data."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        return X_train_bow, X_test_bow
    except Exception as e:
        print(e)

def save_transformed_data(X_train_bow, y_train, X_test_bow, y_test, data_path):
    """Save transformed train and test data as CSV files."""
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())

        train_df['label'] = y_train
        test_df['label'] = y_test

        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
    except Exception as e:
        print(e)

def main():
    """Main function to load data, process it, and save the transformed files."""
    try:
        max_features = load_params('params.yaml')
        train_data = pd.read_csv("./data/processed/train_processed.csv")
        if train_data.empty:
            raise ValueError('Train_data is empty')
    
        test_data = pd.read_csv("./data/processed/test_processed.csv")
        if test_data.empty:
            raise ValueError('Test data is empty')
        
        X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
        X_train_bow, X_test_bow = vectorize_text(X_train, X_test, max_features)
        data_path = os.path.join('data','features')
        save_transformed_data(X_train_bow, y_train, X_test_bow, y_test, data_path)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
