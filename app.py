from flask import Flask, render_template_string, request
import joblib
import pandas as pd
import os

# --------------------------
# Load models and encoders
# --------------------------
models = {
    "Train/Val Model": {
        "model": joblib.load(os.path.join("model", "region_model.pkl")),
        "encoder": joblib.load(os.path.join("model", "region_encoder.pkl"))
    },
    "K-Fold CV Model": {
        "model": joblib.load(os.path.join("model", "best_region_model.pkl")),
        "encoder": joblib.load(os.path.join("model", "region_encoder_cv.pkl"))
    }
}

# --------------------------
# HTML template with styling
# --------------------------
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
        .match-ok { color: green; font-weight: bold; }
        .match-error { color: red; font-weight: bold; }
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

        <!-- Loading Spinner -->
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

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)

@app.route("/predict_polyphenol_region", methods=["POST"])
def predict_polyphenol_region():
    try:
        # Receive JSON data from frontend
        req_data = request.get_json()
        data = req_data.get("data", [])

        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid or empty data"}), 400

        # Convert JSON list into DataFrame
        df = pd.DataFrame(data)

        # Ensure correct columns
        if not {"Absorbance", "Concentration"}.issubset(df.columns):
            return jsonify({"error": "Missing required fields"}), 400

        # Run predictions
        X = df[["Absorbance", "Concentration"]]
        preds = model.predict(X)
        probs = model.predict_proba(X)

        # Map predictions back to labels
        decoded_preds = encoder.inverse_transform(preds)

        # Prepare response
        results = []
        for i, p in enumerate(decoded_preds):
            conf = float(np.max(probs[i])) if len(probs.shape) > 1 else None
            results.append({
                "sample": f"Sample-{i+1}",
                "prediction": str(p),
                "confidence": round(conf, 4) if conf is not None else None,
                "error": None
            })

        return jsonify(results)

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)