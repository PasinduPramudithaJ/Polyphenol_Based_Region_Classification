import pandas as pd
import os
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --------------------------
# Load region classes
# --------------------------
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print("📘 Loaded region classes:", class_names)

# --------------------------
# Function to load dataset from subfolders
# --------------------------
def load_dataset(base_path):
    data = []
    for region in os.listdir(base_path):
        region_path = os.path.join(base_path, region)
        if os.path.isdir(region_path):
            csv_files = glob(os.path.join(region_path, "*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df["Region"] = region  # assign region based on folder name
                    data.append(df)
                except Exception as e:
                    print(f"⚠️ Skipped {csv_file}: {e}")
    if not data:
        raise ValueError(f"No data found in {base_path} folder!")
    return pd.concat(data, ignore_index=True)

# --------------------------
# Load train and validation datasets
# --------------------------
train = load_dataset("train")
val = load_dataset("val")

print(f"✅ Loaded {len(train)} training samples from {train['Region'].nunique()} regions.")
print(f"✅ Loaded {len(val)} validation samples from {val['Region'].nunique()} regions.")
print("Unique regions in train:", train["Region"].unique())
print("Unique regions in val:", val["Region"].unique())
print("Classes.txt:", class_names)

# --------------------------
# Prepare features and labels
# --------------------------
X_train = train[['Absorbance', 'Concentration']]
y_train = train['Region'].astype(str)
X_val = val[['Absorbance', 'Concentration']]
y_val = val['Region'].astype(str)

# --------------------------
# Encode labels using predefined classes
# --------------------------
encoder = LabelEncoder()
encoder.fit(class_names)

# Filter out samples not in predefined classes
train_mask = y_train.isin(class_names)
val_mask = y_val.isin(class_names)

X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]
X_val = X_val.loc[val_mask]
y_val = y_val.loc[val_mask]

# Encode
y_train_enc = encoder.transform(y_train)
y_val_enc = encoder.transform(y_val)

# --------------------------
# Train Random Forest model
# --------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train_enc)

# --------------------------
# Validate model
# --------------------------
y_pred = model.predict(X_val)
acc = accuracy_score(y_val_enc, y_pred)

print(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:\n",
      classification_report(y_val_enc, y_pred, target_names=encoder.classes_))

# --------------------------
# Save model and encoder inside 'model' folder
# --------------------------
os.makedirs("model", exist_ok=True)

model_path = os.path.join("model", "region_model.pkl")
encoder_path = os.path.join("model", "region_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Encoder saved to: {encoder_path}")
