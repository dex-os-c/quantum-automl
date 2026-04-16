import pandas as pd
import joblib
from quantum_automl import QuantumAutoClassifier

# 1. Load data
df = pd.read_csv("patients.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 2. Create and train AutoML model
clf = QuantumAutoClassifier(
    max_qubits=4,
    max_iter=20,          # small for testing; increase for better results
    cv_folds=2,
    verbose=True
)

print("Training AutoML model...")
clf.fit(X, y)

# 3. Save the entire AutoML object (includes best model + preprocessor)
joblib.dump(clf, "quantum_automl_model.pkl")
print(f"Model saved to quantum_automl_model.pkl")
print(f"Best CV score: {clf.best_score_:.4f}")
print(f"Best model: {clf.best_model_name_}")
