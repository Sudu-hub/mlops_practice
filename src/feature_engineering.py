import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import yaml

max_feature = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data = train_data.fillna('')
test_data = test_data.fillna('')

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_feature)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)
train_df = pd.DataFrame(X_train_bow.toarray())
test_df = pd.DataFrame(X_test_bow.toarray())

train_df['label'] = y_train
test_df['label'] = y_test
data_path = os.path.join('data','features')
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path,'train_bow.csv'))
test_df.to_csv(os.path.join(data_path,'test_bow.csv'))