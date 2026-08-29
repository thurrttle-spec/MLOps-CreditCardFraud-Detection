import os
import time
import pandas as pd
import tensorflow as tf
import kagglehub
from flask import Flask, request, jsonify, render_template
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ─── 1. Unduh Dataset via Kagglehub ──────────────────────────────────────────
print("📥 Mengunduh dataset via kagglehub...")
try:
    dataset_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(dataset_dir, "creditcard.csv")
    print("📖 Membaca dataset...")
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset berhasil dimuat! Total baris: {len(df)}")
except Exception as e:
    print(f"⚠️ Gagal memuat dataset: {e}")

# ─── 2. Inisialisasi Aplikasi Flask & Metrics ────────────────────────────────
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
    ['result']
)

# ─── 3. Pembuatan Path Model Secara Presisi ──────────────────────────────────
# Menggunakan absolute path langsung ke folder serving_model_dir di dalam /app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    'MODEL_DIR', 
    os.path.join(BASE_DIR, 'serving_model_dir', 'cc-fraud-model')
)

def load_latest_model(base_dir: str):
    """Memuat SavedModel versi terbaru dari base_dir."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Folder model tidak ditemukan di: {base_dir}")

    versions = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: int(x) if x.isdigit() else 0,
        reverse=True
    )
    if not versions:
        raise FileNotFoundError(f"Tidak ada versi model ditemukan di: {base_dir}")
        
    model_path = os.path.join(base_dir, versions[0])
    print(f"[INFO] Loading model dari: {model_path}")
    return tf.saved_model.load(model_path)

model = load_latest_model(MODEL_DIR)
infer = model.signatures['serving_default']

# ─── 4. Constants & Helper ───────────────────────────────────────────────────
NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

def make_tf_example(feature_dict: dict) -> bytes:
    feature = {}
    for key in NUMERICAL_FEATURES:
        val = float(feature_dict.get(key, 0.0))
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[val])
        )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

# ─── 5. Routes API ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    REQUEST_COUNT.labels('GET', '/', 200).inc()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    try:
        data = request.get_json(force=True)
        if data is None:
            REQUEST_COUNT.labels('POST', '/predict', 400).inc()
            return jsonify({'error': 'Invalid JSON input'}), 400

        serialized = make_tf_example(data)
        input_tensor = tf.constant([serialized], dtype=tf.string)

        output = infer(examples=input_tensor)
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
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
