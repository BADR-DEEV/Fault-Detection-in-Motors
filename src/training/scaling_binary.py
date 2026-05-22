# save_final_model.py
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train_paper_repro import section_dataset, build_window_dataset

RESULTS_DIR = r"C:\DEV\Personal\Ai-Driven-Vibrations-motor\src\training\Results_pipeline_A"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading dataset...")
healthy, faulty = section_dataset()
files = build_window_dataset(healthy, faulty)

X_list, y_list = [], []
for fobj in files:
    for wsets in fobj["window_features"]:
        X_list.append(wsets["F7"]) # Using F7 (All 21 features)
        y_list.append(fobj["label"])

X_train = np.vstack(X_list)
y_train = np.array(y_list, dtype=int)

print(f"Training on {X_train.shape[0]} windows, {X_train.shape[1]} features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Using a standard RBF SVM with probability enabled
model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_scaled, y_train)

joblib.dump(scaler, os.path.join(RESULTS_DIR, "binary_scaler.pkl"))
joblib.dump(model, os.path.join(RESULTS_DIR, "binary_model.pkl"))
print(f"✅ Saved model and scaler to {RESULTS_DIR}")