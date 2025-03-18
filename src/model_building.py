import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml

def load_params(params_yaml):
    """Load model parameters from YAML configuration file."""
    return yaml.safe_load(open(params_yaml))['model_building']

def load_data(data_path):
    """Load training dataset from CSV file."""
    return pd.read_csv(data_path)

def split_data(data):
    """Extract features and target variables from dataset."""
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def train_model(X_train, y_train, params):
    """Train the Gradient Boosting Classifier model."""
    clf = GradientBoostingClassifier(
        n_estimators=params['n_estimators'], 
        learning_rate=params['learning_rate']
    )
    return clf.fit(X_train, y_train)

def save_model(model, output_path):
    """Save the trained model using pickle."""
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    """Complete pipeline for training and saving the model."""
    params = load_params('params.yaml')
    train_data = load_data('./data/features/train_bow.csv')
    X_train, y_train = split_data(train_data)
    model = train_model(X_train, y_train, params)
    save_model(model, 'model.pkl')

if __name__ == '__main__':
    main()