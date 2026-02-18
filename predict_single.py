import joblib
import pandas as pd
import os

# --------------------------
# Load model and encoder from model folder
# --------------------------
model_path = os.path.join("model", "region_model.pkl")
encoder_path = os.path.join("model", "region_encoder.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

def predict_region(absorbance, concentration,Dry_matter_content,Caffiene_Content):
    """Predict tea region from single absorbance and concentration input."""
    # Create DataFrame with same column names as training data
    features = pd.DataFrame([[absorbance, concentration,Dry_matter_content,Caffiene_Content]], columns=["Absorbance", "Concentration","Dry matter content","Caffiene Content"])
    pred = model.predict(features)[0]
    region = encoder.inverse_transform([pred])[0]
    return region

if __name__ == "__main__":
    abs_val = float(input("Enter Absorbance: "))
    conc_val = float(input("Enter Concentration: "))
    dry_matter_val = float(input("Enter Dry Matter Content: "))
    caffiene_val = float(input("Enter Caffiene Content: "))
    result = predict_region(abs_val, conc_val,dry_matter_val,caffiene_val)
    print(f"🌱 Predicted Region: {result}")
