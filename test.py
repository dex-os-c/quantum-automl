import pandas as pd
import joblib
import numpy as np

# 1. Load the saved AutoML model
clf = joblib.load("quantum_automl_model.pkl")
print("Model loaded successfully")

# 2. New data (must have the same number of features as training data)
# Example: one new patient with 4 features
X_new = pd.DataFrame([[52, 142, 132, 235]])

# If you know the column names used during training, you can set them:
# X_new.columns = clf.feature_names_in_

# 3. Predict
predictions = clf.predict(X_new)
print(f"Prediction: {predictions[0]}")

# 4. (Optional) Predict on multiple samples
X_batch = np.array([
    [52, 142, 132, 235],
    [45, 120, 100, 200],
    [60, 150, 140, 250]
])
batch_preds = clf.predict(X_batch)
print(f"Batch predictions: {batch_preds}")
