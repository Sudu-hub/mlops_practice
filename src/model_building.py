import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml

params = yaml.safe_load(open('params.yaml'))['model_building']
# Load training data
train_data = pd.read_csv("./data/features/train_bow.csv")

# Extract features and target
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Train the Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
clf.fit(X_train, y_train)

# Save the trained model
pickle.dump(clf, open('model.pkl', 'wb'))

