import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------
# Load dataset
# --------------------------
data_path = "data/3_Region_Dataset.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

if "Region" not in data.columns:
    raise KeyError("Column 'Region' not found. Please check your CSV headers.")

print("📘 Loaded dataset with shape:", data.shape)
print("📊 Columns:", list(data.columns))

# --------------------------
# Prepare features and labels
# --------------------------
X = data[["Absorbance", "Concentration"]]
y = data["Region"].astype(str)

# --------------------------
# Encode labels
# --------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_
print("🏷️ Classes:", list(class_names))

# --------------------------
# K-Fold Cross Validation Setup
# --------------------------
k = 5  # Number of folds (you can change this)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
best_model = None
best_acc = 0.0
fold = 1

# --------------------------
# Train and evaluate per fold
# --------------------------
for train_index, val_index in skf.split(X, y_encoded):
    print(f"\n🚀 Fold {fold}/{k} ---------------------------")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accuracies.append(acc)

    print(f"✅ Fold {fold} Accuracy: {acc * 100:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    # Save best model
    if acc > best_acc:
        best_acc = acc
        best_model = model

    fold += 1

# --------------------------
# Summary
# --------------------------
avg_acc = np.mean(fold_accuracies)
print("\n==============================")
print(f"📈 Average Cross-Validation Accuracy: {avg_acc * 100:.2f}%")
print(f"🏆 Best Fold Accuracy: {best_acc * 100:.2f}%")
print("==============================")

# --------------------------
# Save model and encoder in model/ folder
# --------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

best_model_path = os.path.join(model_dir, "best_region_model.pkl")
encoder_path = os.path.join(model_dir, "region_encoder_cv.pkl")

if best_model is not None:
    joblib.dump(best_model, best_model_path)
    joblib.dump(encoder, encoder_path)
    print(f"\n💾 Best model saved to: {best_model_path}")
    print(f"💾 Encoder saved to: {encoder_path}")

print("\n✅ K-Fold cross-validation completed successfully!")
