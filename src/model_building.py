import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load training data
train_data = pd.read_csv("./data/features/train_bow.csv")

# Extract features and target
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Train the Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
clf.fit(X_train, y_train)

# Save the trained model
pickle.dump(clf, open('model.pkl', 'wb'))

