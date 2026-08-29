import os
import json
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import os
import urllib.request
import pandas as pd
from flask import Flask, jsonify

DATA_PATH = "creditcard.csv"
DATA_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

def ensure_data_exists():
    if not os.path.exists(DATA_PATH):
        print("📥 Mengunduh creditcard.csv...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("✅ Unduhan selesai!")

# Unduh data SEBELUM memuat dataset ke pandas / memory
ensure_data_exists()

app = Flask(__name__)
df = pd.read_csv(DATA_PATH)

@app.route("/")
def index():
    return jsonify({"total_rows": len(df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# ─── App & Prometheus Setup ───────────────────────────────────────────────────
app = Flask(__name__)

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent on model prediction',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
)

FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total fraud predictions',
    ['result']   # 'fraud' or 'normal'
)

# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(
    os.path.dirname(__file__), '..', 'serving_model_dir', 'cc-fraud-model'
))

def load_latest_model(base_dir: str):
    """Load the latest versioned SavedModel from base_dir."""
    # SavedModel versions are stored as integer-named subdirectories
    versions = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: int(x) if x.isdigit() else 0,
        reverse=True
    )
    if not versions:
        raise FileNotFoundError(f"No model version found in {base_dir}")
    model_path = os.path.join(base_dir, versions[0])
    print(f"[INFO] Loading model from: {model_path}")
    return tf.saved_model.load(model_path)

model = load_latest_model(MODEL_DIR)
infer = model.signatures['serving_default']

# ─── Feature Constants ────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def make_tf_example(feature_dict: dict) -> bytes:
    """Convert a feature dictionary to a serialized tf.Example."""
    feature = {}
    for key in NUMERICAL_FEATURES:
        val = float(feature_dict.get(key, 0.0))
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[val])
        )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    REQUEST_COUNT.labels('GET', '/', 200).inc()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON with feature values and returns fraud prediction.
    Expected input format:
    {
        "V1": -1.359807, "V2": -0.072781, ..., "V28": -0.021053,
        "Time": 0.0, "Amount": 149.62
    }
    """
    start = time.time()
    try:
        data = request.get_json(force=True)
        if data is None:
            REQUEST_COUNT.labels('POST', '/predict', 400).inc()
            return jsonify({'error': 'Invalid JSON input'}), 400

        # Build serialized TF Example
        serialized = make_tf_example(data)
        input_tensor = tf.constant([serialized], dtype=tf.string)

        # Run inference
        output = infer(examples=input_tensor)
        # The output key depends on model signature; usually 'output_0' or 'dense_X'
        output_key = list(output.keys())[0]
        probability = float(output[output_key].numpy()[0][0])

        is_fraud = probability >= 0.5
        result_label = 'fraud' if is_fraud else 'normal'

        FRAUD_PREDICTIONS.labels(result=result_label).inc()
        elapsed = time.time() - start
        PREDICTION_LATENCY.observe(elapsed)
        REQUEST_COUNT.labels('POST', '/predict', 200).inc()

        return jsonify({
            'fraud_probability': round(probability, 6),
            'is_fraud': is_fraud,
            'label': result_label,
            'latency_seconds': round(elapsed, 4)
        })

    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    REQUEST_COUNT.labels('GET', '/health', 200).inc()
    return jsonify({'status': 'ok', 'model_loaded': True})


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
