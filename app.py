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
# HTML template with Clear button
# --------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tea Region Predictor</title>
    <style>
        body { font-family: Arial; background: #f0f5f5; text-align: center; padding-top: 40px; }
        .box { background: white; padding: 30px; margin: auto; width: 700px; border-radius: 12px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
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
            
            <h4>OR Upload CSV (with columns: Sample Name, Absorbance, Concentration):</h4>
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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    table = None
    
    if request.method == "POST":
        try:
            model_name = request.form["model_name"]
            if model_name not in models:
                prediction = "❌ Invalid model selected"
            else:
                model = models[model_name]["model"]
                encoder = models[model_name]["encoder"]
                
                # Check CSV upload
                if "csv_file" in request.files and request.files["csv_file"].filename != "":
                    file = request.files["csv_file"]
                    df = pd.read_csv(file)
                    required_cols = {"Sample Name", "Absorbance", "Concentration"}
                    if not required_cols.issubset(df.columns):
                        prediction = "❌ CSV must contain columns: Sample Name, Absorbance, Concentration"
                    else:
                        features = df[["Absorbance", "Concentration"]]
                        preds = model.predict(features)
                        df["Predicted_Region"] = encoder.inverse_transform(preds)
                        table = df.to_html(classes="table table-striped", index=False)
                
                # Single prediction
                elif request.form.get("absorbance") and request.form.get("concentration"):
                    absorbance = float(request.form["absorbance"])
                    concentration = float(request.form["concentration"])
                    features = pd.DataFrame([[absorbance, concentration]], columns=["Absorbance", "Concentration"])
                    pred = model.predict(features)[0]
                    prediction = encoder.inverse_transform([pred])[0]
                
                else:
                    prediction = "❌ Please enter values or upload a CSV file"
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction, table=table, models=models.keys())

# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
