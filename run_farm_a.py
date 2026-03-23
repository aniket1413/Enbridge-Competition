# run_farm_a.py
# Run everything in one go

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path
import joblib

print("="*50)
print("WIND FARM A ANALYSIS")
print("="*50)

# Paths
farm_a_path = Path("CARE_To_Compare/Wind Farm A")
datasets_path = farm_a_path / "datasets"

# Load event info
events = pd.read_csv(farm_a_path / "event_info.csv")
print(f"Loaded event info: {len(events)} events")

# Load all data files
csv_files = list(datasets_path.glob("*.csv"))
print(f"Loading {len(csv_files)} files...")

all_data = []
for file in csv_files:
    df = pd.read_csv(file, sep=';')
    df['source_file'] = file.name
    all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
print(f"Total data: {len(df)} rows")

# Prepare data
exclude_cols = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id', 'source_file']
sensor_cols = [col for col in df.columns if col not in exclude_cols]

train_data = df[df['train_test'] == 'train'].copy()
test_data = df[df['train_test'] == 'test'].copy()
train_normal = train_data[train_data['status_type_id'] == 0].copy()

# Fill missing values
for col in sensor_cols:
    if train_normal[col].isnull().any():
        train_normal[col].fillna(train_normal[col].median(), inplace=True)

# Train model
features_to_use = sensor_cols[:20]  # First 20 sensors
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(train_normal[features_to_use].values)

# Predict
test_data_clean = test_data[features_to_use].fillna(0)
test_data['is_anomaly'] = model.predict(test_data_clean.values)

# Results
print("\nRESULTS:")
print(f"Anomalies detected: {sum(test_data['is_anomaly'] == -1)}")

# Save
output_path = Path("output")
output_path.mkdir(exist_ok=True)
test_data.to_csv(output_path / "results.csv", index=False)
joblib.dump(model, output_path / "model.pkl")

print("\n✅ Done! Check the 'output' folder")