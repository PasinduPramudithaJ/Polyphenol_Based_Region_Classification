from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import socket
import os
import traceback
from flask_cors import CORS

# ----------------------------------------------------------
# Initialize Flask App
# ----------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

MODELS_DIR = "models"

# ----------------------------------------------------------
# Load Models and Encoders
# ----------------------------------------------------------
models = {
    "Train/Val Model": {
        "model": joblib.load(os.path.join(MODELS_DIR, "region_model.pkl")),
        "encoder": joblib.load(os.path.join(MODELS_DIR, "region_encoder.pkl")),
    },
    "K-Fold CV Model": {
        "model": joblib.load(os.path.join(MODELS_DIR, "best_region_model.pkl")),
        "encoder": joblib.load(os.path.join(MODELS_DIR, "region_encoder_cv.pkl")),
    },
}

# Default model for API
rf_model = models["K-Fold CV Model"]["model"]
encoder = models["K-Fold CV Model"]["encoder"]

TRAINED_FEATURES = list(rf_model.feature_names_in_)
print("Model expects features:", TRAINED_FEATURES)

# ----------------------------------------------------------
# HTML Template (UNCHANGED DESIGN)
# ----------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tea Region Predictor</title>
    <style>
        body { font-family: Arial; background: #f0f5f5; text-align: center; padding-top: 40px; }
        .box { background: white; padding: 30px; margin: auto; width: 1000px; border-radius: 12px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        input, select { width: 85%; padding: 10px; margin: 8px; border: 1px solid #ccc; border-radius: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background-color: #45a049; }
        table { margin: 20px auto; border-collapse: collapse; width: 95%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="box">
        <h2>Tea Region Prediction</h2>
        <form method="post" enctype="multipart/form-data">

            <h4>Single Sample Prediction:</h4>
            <input type="number" step="0.0001" name="absorbance" placeholder="Absorbance"><br>
            <input type="number" step="0.0001" name="concentration" placeholder="Concentration"><br>
            <input type="number" step="0.0001" name="dry_matter" placeholder="Dry matter content"><br>
            <input type="number" step="0.0001" name="caffiene" placeholder="Caffiene Content"><br>

            <h4>OR Upload CSV</h4>
            <p>CSV must include trained model features.</p>
            <input type="file" name="csv_file" accept=".csv"><br>

            <select name="model_name" required>
                <option value="">-- Select Model --</option>
                {% for name in models %}
                    <option value="{{ name }}">{{ name }}</option>
                {% endfor %}
            </select><br>

            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <h3>🌱 Predicted Region: <span style="color:green">{{ prediction }}</span></h3>
        {% endif %}

        {% if accuracy %}
            <h3>🎯 Accuracy: {{ accuracy }}%</h3>
        {% endif %}

        {% if table %}
            <h3>📊 Batch Prediction Results:</h3>
            {{ table|safe }}
        {% endif %}

        {% if error %}
            <h3 style="color:red;">{{ error }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

# ----------------------------------------------------------
# Home Route
# ----------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            model_name = request.form.get("model_name")
            selected_model = models.get(model_name)

            if not selected_model:
                return render_template_string(
                    HTML_TEMPLATE,
                    models=models,
                    error="Please select a model"
                )

            model = selected_model["model"]
            encoder = selected_model["encoder"]
            trained_features = list(model.feature_names_in_)

            file = request.files.get("csv_file")

            absorbance = request.form.get("absorbance")
            concentration = request.form.get("concentration")
            dry_matter = request.form.get("dry_matter")
            caffiene = request.form.get("caffiene")

            # --------------------------------------------------
            # CSV Upload
            # --------------------------------------------------
            if file:
                df = pd.read_csv(file)

                if not set(trained_features).issubset(df.columns):
                    return render_template_string(
                        HTML_TEMPLATE,
                        models=models,
                        error=f"CSV must contain: {trained_features}"
                    )

                X = df[trained_features]
                preds = model.predict(X)
                df["Predicted Region"] = encoder.inverse_transform(preds)

                # ✔ Compare with actual Region if exists
                if "Region" in df.columns:
                    df["Result"] = np.where(
                        df["Predicted Region"] == df["Region"],
                        "✔",
                        "✘"
                    )

                    accuracy = round((df["Result"] == "✔").mean() * 100, 2)

                    return render_template_string(
                        HTML_TEMPLATE,
                        models=models,
                        table=df.to_html(index=False),
                        accuracy=accuracy
                    )

                return render_template_string(
                    HTML_TEMPLATE,
                    models=models,
                    table=df.to_html(index=False)
                )

            # --------------------------------------------------
            # Single Prediction
            # --------------------------------------------------
            if absorbance or concentration or dry_matter or caffiene:

                input_data = {
                    "Absorbance": float(absorbance) if absorbance else 0,
                    "Concentration": float(concentration) if concentration else 0,
                    "Dry matter content": float(dry_matter) if dry_matter else 0,
                    "Caffiene Content": float(caffiene) if caffiene else 0
                }

                df_input = pd.DataFrame([input_data])
                df_input = df_input[trained_features]

                pred = model.predict(df_input)
                decoded = encoder.inverse_transform(pred)[0]

                return render_template_string(
                    HTML_TEMPLATE,
                    models=models,
                    prediction=decoded
                )

        except Exception as e:
            traceback.print_exc()
            return render_template_string(
                HTML_TEMPLATE,
                models=models,
                error=str(e)
            )

    return render_template_string(HTML_TEMPLATE, models=models)


# ----------------------------------------------------------
# API Endpoint (UNCHANGED LOGIC, CLEANED)
# ----------------------------------------------------------
@app.route("/predict_polyphenol_region", methods=["POST"])
def predict_polyphenol_region():
    try:
        req_data = request.get_json()
        data = req_data.get("data", [])

        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid or empty data"}), 400

        df = pd.DataFrame(data)

        if not set(TRAINED_FEATURES).issubset(df.columns):
            return jsonify({"error": f"Required fields: {TRAINED_FEATURES}"}), 400

        X = df[TRAINED_FEATURES]

        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)
        decoded_preds = encoder.inverse_transform(preds)

        results = []
        for i, p in enumerate(decoded_preds):
            confidence = float(np.max(probs[i]))
            results.append({
                "sample": f"Sample-{i+1}",
                "prediction": str(p),
                "confidence": round(confidence, 4)
            })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------
# Run Server
# ----------------------------------------------------------
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    local_ip = get_local_ip()
    print("Server running on:")
    print(f"  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{local_ip}:5000")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
