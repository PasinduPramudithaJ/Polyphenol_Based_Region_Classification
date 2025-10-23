import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("data/3_Region_Dataset.csv")
data.columns = data.columns.str.strip()  # remove spaces
print("📊 Columns:", data.columns)

if 'Region' not in data.columns:
    raise KeyError("Column 'Region' not found. Please check your CSV headers.")

# ---------------------------
# Shuffle the dataset
# ---------------------------
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# Stratified split: 60% train, 30% val, 10% test
# ---------------------------
# First split: 60% train, 40% temp (val+test)
train_df, temp_df = train_test_split(
    data, test_size=0.4, random_state=42, stratify=data['Region']
)

# Second split: from 40% → 30% val (out of total) and 10% test
# That means val:test = 3:1 ratio → test_size = 0.25 of temp_df
val_df, test_df = train_test_split(
    temp_df, test_size=0.25, random_state=42, stratify=temp_df['Region']
)

splits = {"train": train_df, "val": val_df, "test": test_df}

# ---------------------------
# Save each split into folders per region + combined CSV
# ---------------------------
for split_name, split_data in splits.items():
    split_path = split_name
    os.makedirs(split_path, exist_ok=True)
    
    # Save per-region CSVs
    for region in split_data['Region'].unique():
        region_path = os.path.join(split_path, region)
        os.makedirs(region_path, exist_ok=True)
        
        # Filter rows for this region and shuffle
        region_df = split_data[split_data['Region'] == region].copy()
        region_df = region_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        region_csv_path = os.path.join(region_path, f"{region}.csv")
        region_df.to_csv(region_csv_path, index=False)
    
    # Save combined CSV for the split
    combined_csv_path = os.path.join(split_path, f"{split_name}_combined.csv")
    split_data.to_csv(combined_csv_path, index=False)
    print(f"💾 {split_name.capitalize()} combined CSV saved: {combined_csv_path}")

print("✅ Fully randomized dataset split into 60/30/10 per region with combined CSVs!")
print(f"📚 Train size: {len(train_df)}")
print(f"🧪 Validation size: {len(val_df)}")
print(f"🔬 Test size: {len(test_df)}")
