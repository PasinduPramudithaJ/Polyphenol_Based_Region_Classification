import time

from flask import Flask, render_template, request, jsonify, render_template_string
import joblib
from matplotlib import pyplot as plt
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
        
.nav-button {
        background-color: #4CAF50;  /* green like Predict button */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
        transition: 0.3s;  /* smooth hover effect */
    }

.nav-button:hover {
        background-color: #007BFF;  /* blue on hover */
        color: white;               /* text stays white */
    }
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
            <a href="/electro"><button type="button" class="nav-button">TPC Analyzer</button></a>
            
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
    

PLOT_FOLDER = os.path.join("static", "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route("/electro", methods=["GET", "POST"])
def index():

    data = None
    error = None

    if request.method == "POST":

        if "clear" in request.form:
            return render_template("index.html")

        try:

            iv_file = request.files.get("iv_csv")
            ic_file = request.files.get("ic_csv")

            if not iv_file or not ic_file:
                raise ValueError("Upload both CSV files.")

            df_iv = pd.read_csv(iv_file)
            df_ic = pd.read_csv(ic_file)

            if 'V' not in df_iv.columns or 'I' not in df_iv.columns:
                raise ValueError("I-V CSV must contain V and I columns")

            if 'C' not in df_ic.columns or 'I' not in df_ic.columns:
                raise ValueError("Calibration CSV must contain C and I columns")

            # -------- TARGET VOLTAGES --------

            V1_target = 0.4
            V2_target = 0.5

            idx1 = (df_iv["V"] - V1_target).abs().idxmin()
            idx2 = (df_iv["V"] - V2_target).abs().idxmin()

            V1 = df_iv.loc[idx1, "V"]
            I1 = df_iv.loc[idx1, "I"]

            V2 = df_iv.loc[idx2, "V"]
            I2 = df_iv.loc[idx2, "I"]

            # -------- PEAK --------

            imax_index = df_iv["I"].idxmax()

            Imax = df_iv.loc[imax_index, "I"]
            Vmax = df_iv.loc[imax_index, "V"]

            # -------- BASELINE --------

            m1 = (I2 - I1) / (V2 - V1)
            c1 = I1 - m1 * V1

            # Baseline current at Vmax
            Imax_prime = (m1 * Vmax) + c1

            # Net peak current
            I_net = Imax - Imax_prime

            # -------- CALIBRATION --------

            C_vals = df_ic["C"]
            I_vals = df_ic["I"]

            M, C_intercept = np.polyfit(C_vals, I_vals, 1)

            TPC = (I_net - C_intercept) / M

            # -------- UNIQUE FILE NAMES --------

            timestamp = int(time.time())

            iv_file_name = f"iv_plot_{timestamp}.png"
            ic_file_name = f"ic_plot_{timestamp}.png"

            iv_path = os.path.join(PLOT_FOLDER, iv_file_name)
            ic_path = os.path.join(PLOT_FOLDER, ic_file_name)

            # -------- PLOT 1 : I-V --------

            plt.figure(figsize=(7,5))

            plt.plot(df_iv["V"], df_iv["I"], label="Sample Curve")

            plt.scatter([V1,V2,Vmax],[I1,I2,Imax])

            plt.text(V1,I1,"  I1")
            plt.text(V2,I2,"  I2")
            plt.text(Vmax,Imax,"  Imax")

            # baseline
            x_base = np.linspace(V1,V2,50)
            y_base = m1*x_base + c1

            plt.plot(x_base,y_base,"--",label="Baseline")

            # projected baseline current
            plt.scatter(Vmax,Imax_prime)
            plt.text(Vmax,Imax_prime,"  Imax'")

            plt.xlabel("Voltage (V)")
            plt.ylabel("Current (µA)")
            plt.title("I-V Curve Analysis")

            plt.legend()
            plt.grid(True)

            plt.savefig(iv_path)
            plt.close()

            # -------- PLOT 2 : CALIBRATION --------

            plt.figure(figsize=(7,5))

            plt.scatter(C_vals,I_vals,label="Standards")

            x_line=np.linspace(min(C_vals),max(C_vals),100)
            y_line=M*x_line+C_intercept

            plt.plot(x_line,y_line,label="Linear Fit")

            plt.xlabel("Concentration")
            plt.ylabel("Current")
            plt.title("Calibration Curve")

            plt.legend()
            plt.grid(True)

            plt.savefig(ic_path)
            plt.close()

            # -------- SEND DATA --------

            data = {

                "V1":round(V1,4),
                "V2":round(V2,4),

                "I1":round(I1,6),
                "I2":round(I2,6),

                "Vmax":round(Vmax,6),
                "Imax":round(Imax,6),

                "m1":round(m1,6),
                "c1":round(c1,6),

                "Imax_prime":round(Imax_prime,6),
                "I_net":round(I_net,6),

                "M":round(M,6),
                "C_intercept":round(C_intercept,6),

                "TPC":round(TPC,4),

                "iv_plot":f"static/plots/{iv_file_name}",
                "ic_plot":f"static/plots/{ic_file_name}"
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html",data=data,error=error)

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
