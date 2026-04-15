import pandas as pd
from quantum_automl import QuantumAutoClassifier

# Load data
df = pd.read_csv("patients.csv")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Create model
clf = QuantumAutoClassifier(
    max_qubits=4,
    max_iter=20,   # keep small for testing
    cv_folds=2,
    verbose=True
)

# Train
clf.fit(X, y)

# Results
print("Best Score:", clf.best_score_)

# Test prediction
X_new = [[52, 142, 132, 235]]
print("Prediction:", clf.predict(X_new))
