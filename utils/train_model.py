"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fetch training data and preprocess for modeling
train_df = pd.read_csv(r'C:\Users\Dineo\OneDrive\Documents\Coursework - Explore DataScience\Sprint 5 - May 2021\Predict\df-train_set.csv')

train_subset = train_df[train_df['Commodities'] == 'APPLE GOLDEN DELICIOUS']
y_train = train_subset['avg_price_per_kg']
X_train = train_subset[train_subset.columns.difference(['Date', 'Province', 'Container', 'Size_Grade', 'Commodities', 'avg_price_per_kg'])]
scaler = StandardScaler()
lasso = Lasso(alpha = 0.001, random_state=2)
pipeline = Pipeline([('transformer', scaler), ('model', lasso)])

# Fit model
lm = pipeline.fit(X_train, y_train)


# Pickle model for use within our API
save_path = '../assets/trained-models/lasso_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lasso_model, open(save_path,'wb'))
