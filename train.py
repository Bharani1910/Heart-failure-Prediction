import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/Ifeoluwa-hub/Heart-Failure-Prediction-and-Deployment-with-Flask-and-Heroku/master/heart_failure_clinical_records_dataset.csv"

print("Downloading dataset...")
try:
    df = pd.read_csv(url)
    print("Dataset downloaded successfully.")
except Exception as e:
    print("Could not download dataset from raw URL, generating synthetic data...", e)
    # Generate synthetic data if download fails
    np.random.seed(42)
    n_samples = 300
    data = {
        'age': np.random.randint(40, 95, n_samples),
        'anaemia': np.random.randint(0, 2, n_samples),
        'creatinine_phosphokinase': np.random.randint(23, 7861, n_samples),
        'diabetes': np.random.randint(0, 2, n_samples),
        'ejection_fraction': np.random.randint(14, 80, n_samples),
        'high_blood_pressure': np.random.randint(0, 2, n_samples),
        'platelets': np.random.randint(25000, 850000, n_samples),
        'serum_creatinine': np.random.uniform(0.5, 9.4, n_samples),
        'serum_sodium': np.random.randint(113, 148, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'smoking': np.random.randint(0, 2, n_samples),
        'time': np.random.randint(4, 285, n_samples),
        'DEATH_EVENT': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Saved model.pkl successfully!")
