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
# ===== Enable CORS for all origins (supports HTTP & HTTPS) =====
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ----------------------------------------------------------
# Load Models and Encoders
# ----------------------------------------------------------
models = {
    "Train/Val Model": {
        "model": joblib.load(os.path.join("models", "region_model.pkl")),
        "encoder": joblib.load(os.path.join("models", "region_encoder.pkl")),
    },
    "K-Fold CV Model": {
        "model": joblib.load(os.path.join("models", "best_region_model.pkl")),
        "encoder": joblib.load(os.path.join("models", "region_encoder_cv.pkl")),
    },
}

MODELS_DIR = "models"

# ----------------------------------------------------------
# HTML Template
# ----------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tea Region Predictor</title>
    <style>
        body { font-family: Arial; background: #f0f5f5; text-align: center; padding-top: 40px; }
        .box { background: white; padding: 30px; margin: auto; width: 750px; border-radius: 12px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        input, select { width: 85%; padding: 10px; margin: 8px; border: 1px solid #ccc; border-radius: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background-color: #45a049; }
        table { margin: 20px auto; border-collapse: collapse; width: 95%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        #spinner { display: none; margin: 20px auto; }
        .lds-dual-ring { display: inline-block; width: 64px; height: 64px; }
        .lds-dual-ring:after { content: " "; display: block; width: 46px; height: 46px; margin: 1px; border-radius: 50%; border: 5px solid #4CAF50; border-color: #4CAF50 transparent #4CAF50 transparent; animation: lds-dual-ring 1.2s linear infinite; }
        @keyframes lds-dual-ring { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
    <script>
        function showSpinner() {
            document.getElementById('spinner').style.display = 'block';
        }
        function clearResults() {
            document.getElementById('prediction').innerHTML = '';
            document.getElementById('table').innerHTML = '';
            document.querySelector('form').reset();
        }
    </script>
</head>
<body>
    <div class="box">
        <h2>Tea Region Prediction</h2>
        <form method="post" enctype="multipart/form-data" onsubmit="showSpinner()">
            <h4>Single Sample Prediction:</h4>
            <input type="number" step="0.0001" name="absorbance" placeholder="Enter Absorbance"><br>
            <input type="number" step="0.0001" name="concentration" placeholder="Enter Concentration"><br>
            
            <h4>OR Upload CSV (with columns: Sample Name, Absorbance, Concentration [, Region]):</h4>
            <input type="file" name="csv_file" accept=".csv"><br>
            
            <select name="model_name" required>
                <option value="">-- Select Model --</option>
                {% for name in models %}
                    <option value="{{ name }}">{{ name }}</option>
                {% endfor %}
            </select><br>
            
            <button type="submit">Predict</button>
            <button type="button" onclick="clearResults()">Clear</button>
        </form>

        <div id="spinner">
            <div class="lds-dual-ring"></div>
            <p>Processing...</p>
        </div>

        <div id="prediction">
            {% if prediction %}
                <h3>🌱 Predicted Region: <span style="color:green">{{ prediction }}</span></h3>
            {% endif %}
        </div>

        <div id="table">
            {% if table %}
                <h3>📊 Batch Prediction Results:</h3>
                {{ table|safe }}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

# ----------------------------------------------------------
# Home Page (Optional for Local Preview)
# ----------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            model_name = request.form.get("model_name")
            selected_model = models.get(model_name)
            if not selected_model:
                return render_template_string(HTML_TEMPLATE, models=models, prediction=None, table=None)

            model = selected_model["model"]
            encoder = selected_model["encoder"]

            # Handle CSV upload
            file = request.files.get("csv_file")
            absorbance = request.form.get("absorbance")
            concentration = request.form.get("concentration")

            if file:
                df = pd.read_csv(file)
                if not {"Absorbance", "Concentration"}.issubset(df.columns):
                    return render_template_string(HTML_TEMPLATE, models=models, prediction=None, table="<p>❌ Invalid CSV columns</p>")

                X = df[["Absorbance", "Concentration"]]
                preds = model.predict(X)
                df["Predicted Region"] = encoder.inverse_transform(preds)
                table_html = df.to_html(index=False)
                return render_template_string(HTML_TEMPLATE, models=models, prediction=None, table=table_html)

            elif absorbance and concentration:
                X = pd.DataFrame([[float(absorbance), float(concentration)]], columns=["Absorbance", "Concentration"])
                pred = model.predict(X)
                decoded = encoder.inverse_transform(pred)[0]
                return render_template_string(HTML_TEMPLATE, models=models, prediction=decoded, table=None)

        except Exception as e:
            print("Error in form:", e)
            traceback.print_exc()

    return render_template_string(HTML_TEMPLATE, models=models, prediction=None, table=None)


# ----------------------------------------------------------
# Load trained RandomForest model and label encoder
# ----------------------------------------------------------
rf_model = joblib.load(os.path.join(MODELS_DIR, "best_region_model.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "region_encoder_cv.pkl"))

@app.route("/predict_polyphenol_region", methods=["POST"])
def predict_polyphenol_region():
    try:
        req_data = request.get_json()
        data = req_data.get("data", [])
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid or empty data"}), 400
        df = pd.DataFrame(data)
        if not {"Absorbance", "Concentration"}.issubset(df.columns):
            return jsonify({"error": "Missing required fields"}), 400
        X = df[["Absorbance", "Concentration"]]
        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)
        decoded_preds = encoder.inverse_transform(preds)
        results = []
        for i, p in enumerate(decoded_preds):
            conf = float(np.max(probs[i])) if len(probs.shape) > 1 else None
            results.append({
                "sample": f"Sample-{i+1}",
                "prediction": str(p),
                "confidence": round(conf,4) if conf is not None else None,
                "error": None
            })
        return jsonify(results)
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================= Server Setup =====================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
    except:
        ip="127.0.0.1"
    finally:
        s.close()
    return ip

if __name__=="__main__":
    local_ip = get_local_ip()
    print(f"Server running on:\n  Localhost: http://127.0.0.1:5000\n  Network: http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

# ----------------------------------------------------------
# Utility to Get Local IP (for LAN testing)
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

# ----------------------------------------------------------
# Run Server
# ----------------------------------------------------------
if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Server running on:\n  Local: http://127.0.0.1:5000\n  Network: http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
