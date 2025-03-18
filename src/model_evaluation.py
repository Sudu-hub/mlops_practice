import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import json

clf = pickle.load(open('model.pkl','rb'))
def load_data(testdata):
    test_data = pd.read_csv(testdata)
    return test_data

def split_data(test_data):
    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values
    return X_test, y_test

def prediction(Xtest):
    y_pred = clf.predict(Xtest)
    y_pred_proba = clf.predict_proba(Xtest)[:, 1]
    return y_pred, y_pred_proba

# Calculate evaluation metrics
def metrics(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, precision, recall, auc

def metridict(accuracy, precision, recall, auc):
    metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'auc':auc
    }
    return metrics_dict


def main():
    X_df = load_data('./data/features/test_bow.csv')
    X_test, y_test = split_data(X_df)
    y_pred, y_pred_prob = prediction(X_test)
    accuracy, precision, recall, auc = metrics(y_test, y_pred, y_pred_prob)
    metric = metridict(accuracy, precision, recall, auc)
    with open('metrics.json', 'w') as file:
        json.dump(metric, file, indent=4)

if __name__ == '__main__':
    main()