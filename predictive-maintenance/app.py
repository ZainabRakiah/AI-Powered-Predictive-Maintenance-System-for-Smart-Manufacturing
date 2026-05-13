import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

# Short JSON keys from the form -> columns expected by the trained pipeline
JSON_TO_COL = {
    "type": "Type",
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

app = Flask(__name__)
_artifact = None


def load_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH.name}. Run: python train_model.py"
            )
        raw = joblib.load(MODEL_PATH)
        if isinstance(raw, dict) and "model" in raw:
            _artifact = raw
            m = _artifact["model"]
            if hasattr(m, "set_params") and "clf" in getattr(m, "named_steps", {}):
                m.set_params(clf__n_jobs=1)
        else:
            raise TypeError(
                "model.pkl must be the dict saved by train_model.py "
                "(keys: model, feature_cols, target_col)."
            )
    return _artifact


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected JSON body"}), 400

    artifact = load_artifact()
    feature_cols = artifact["feature_cols"]
    model = artifact["model"]

    row = {}
    for json_key, col_name in JSON_TO_COL.items():
        if json_key not in payload:
            return jsonify({"error": f"Missing field: {json_key}"}), 400
        val = payload[json_key]
        if json_key == "type":
            if val not in ("L", "M", "H"):
                return jsonify({"error": 'type must be "L", "M", or "H"'}), 400
            row[col_name] = val
        else:
            try:
                row[col_name] = float(val)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid number for {json_key}"}), 400

    X = pd.DataFrame([row])[feature_cols]
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    label = "High Failure Risk" if pred == 1 else "Machine Healthy"

    return jsonify(
        {
            "prediction": label,
            "failure_probability": proba,
        }
    )


if __name__ == "__main__":
    load_artifact()
    host, port = "127.0.0.1", 5000
    if os.environ.get("FLASK_DEBUG") == "1":
        print("FLASK_DEBUG=1: using Flask dev server (no reloader).")
        app.run(host=host, port=port, debug=True, use_reloader=False)
    else:
        try:
            from waitress import serve

            print(f"Serving on http://{host}:{port} (waitress, Ctrl+C to stop)")
            serve(app, host=host, port=port, threads=4)
        except ImportError:
            print("waitress not installed; pip install waitress  (or set FLASK_DEBUG=1)")
            app.run(host=host, port=port, debug=False, use_reloader=False)
