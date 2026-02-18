import os
import pandas as pd
import joblib
from glob import glob
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# Load model and encoder from model folder
# --------------------------
model_path = os.path.join("model", "region_model.pkl")
encoder_path = os.path.join("model", "region_encoder.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# --------------------------
# Load and predict all test CSVs
# --------------------------
test_base = "test"
all_predictions = []

for region in os.listdir(test_base):
    region_path = os.path.join(test_base, region)
    if os.path.isdir(region_path):
        csv_files = glob(os.path.join(region_path, "*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check required columns
                if not {'Absorbance', 'Concentration', 'Sample Name'}.issubset(df.columns):
                    print(f"⚠️ Skipped {csv_file} (missing required columns)")
                    continue
                
                # Predict
                X = df[['Absorbance', 'Concentration','Dry matter content','Caffiene Content']]
                preds = model.predict(X)
                df['Predicted_Region'] = encoder.inverse_transform(preds)
                
                # Add true region column
                df['Actual_Region'] = region
                
                all_predictions.append(df[['Sample Name', 'Absorbance', 'Concentration','Dry matter content','Caffiene Content', 'Actual_Region', 'Predicted_Region']])
                
            except Exception as e:
                print(f"⚠️ Failed {csv_file}: {e}")

if not all_predictions:
    raise ValueError("❌ No valid test samples found!")

# Combine all predictions into one DataFrame
final_df = pd.concat(all_predictions, ignore_index=True)

# --------------------------
# Print prediction data
# --------------------------
print("\n🌱 All Test Predictions:")
print(final_df)

# --------------------------
# Calculate overall accuracy
# --------------------------
overall_acc = accuracy_score(final_df['Actual_Region'], final_df['Predicted_Region'])
print(f"\n✅ Overall Test Accuracy: {overall_acc * 100:.2f}%")

# --------------------------
# Detailed classification report
# --------------------------
report_str = classification_report(
    final_df['Actual_Region'],
    final_df['Predicted_Region'],
    target_names=encoder.classes_,
    zero_division=0
)
print("\n📊 Classification Report:")
print(report_str)

# --------------------------
# Save final combined CSV and report
# --------------------------
os.makedirs("predict", exist_ok=True)

final_csv_path = os.path.join("predict", "test_predictions.csv")
final_report_path = os.path.join("predict", "test_predictions_report.txt")

final_df.to_csv(final_csv_path, index=False)
with open(final_report_path, "w") as f:
    f.write(f"Overall Test Accuracy: {overall_acc * 100:.2f}%\n\n")
    f.write(report_str)

print(f"\n💾 Combined predictions saved to: {final_csv_path}")
print(f"💾 Classification report saved to: {final_report_path}")
