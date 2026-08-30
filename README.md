# Submission 2: Credit Card Fraud Detection — MLOps Pipeline
Nama: Muhammad Fathurrohman

Username dicoding: M_Fathurrohman

---

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Credit Card Fraud Detection — Dataset publik dari Kaggle yang berisi 284.807 transaksi kartu kredit yang terjadi selama 2 hari pada September 2013 oleh pemegang kartu kredit Eropa. Dataset memiliki 30 fitur numerik: 28 fitur hasil PCA anonim (V1–V28), fitur Time (detik sejak transaksi pertama), fitur Amount (nilai transaksi), dan label biner Class (0 = normal, 1 = fraud). Dataset sangat tidak seimbang (imbalanced): hanya 0,172% transaksi yang merupakan penipuan (492 dari 284.807). |
| Masalah | Penipuan kartu kredit (credit card fraud) merupakan masalah serius yang merugikan nasabah maupun lembaga keuangan secara finansial. Tantangan utama dalam deteksi penipuan adalah ketidakseimbangan kelas yang ekstrem — jumlah transaksi penipuan jauh lebih sedikit dibandingkan transaksi normal — sehingga model cenderung bias ke kelas mayoritas. Dibutuhkan sistem machine learning yang mampu secara akurat mengidentifikasi transaksi penipuan di antara jutaan transaksi normal secara real-time. |
| Solusi machine learning | Solusi yang dikembangkan adalah sebuah sistem machine learning end-to-end berbasis TensorFlow Extended (TFX) yang mampu: (1) memproses data secara otomatis dan konsisten, (2) melatih model klasifikasi biner Deep Neural Network (DNN) dengan hyperparameter tuning otomatis via KerasTuner, dan (3) men-deploy model ke production environment di Railway yang dapat diakses melalui REST API. Target kinerja model: Binary Accuracy ≥ 99% dan AUC-ROC ≥ 0.95 pada data evaluasi. |
| Metode pengolahan | MMetode pengolahan data menggunakan tf.Transform (TFX Transform component): seluruh 30 fitur numerik (V1–V28, Time, Amount) dinormalisasi menggunakan Z-score standardization (tft.scale_to_z_score()) agar setiap fitur berada pada skala yang seragam (mean=0, std=1). Label target Class di-cast ke float32 untuk kompatibilitas dengan fungsi loss Binary Crossentropy. Dataset dibagi dengan rasio 80% training : 20% evaluasi menggunakan hash-based splitting pada CsvExampleGen. |
| Arsitektur model | Model yang dibangun adalah Feedforward Deep Neural Network (DNN) berbasis Keras dengan arsitektur adaptif melalui hyperparameter tuning (tfx.components.Tuner via KerasTuner): Input Layer — 30 neuron (satu per fitur numerik yang telah di-transform); Hidden Layers — 1–3 layer Dense dengan aktivasi ReLU, diikuti BatchNormalization dan Dropout untuk regularisasi; Output Layer — 1 neuron dengan aktivasi Sigmoid untuk klasifikasi biner. Optimizer: Adam dengan BinaryCrossentropy sebagai loss function. |
| Metrik evaluasi | Model dievaluasi menggunakan metrik berikut melalui TensorFlow Model Analysis (TFMA): BinaryAccuracy (akurasi keseluruhan), AUC (Area Under ROC Curve — penting untuk dataset imbalanced), Precision (ketepatan prediksi fraud), Recall (kemampuan mendeteksi semua transaksi fraud), serta matriks kebingungan (TruePositives / FalsePositives / TrueNegatives / FalseNegatives). Model dianggap layak di-deploy apabila BinaryAccuracy ≥ 0.99 dan AUC ≥ 0.95.|
| Performa model | Model menghasilkan performa yang baik: Binary Accuracy ≈ 99% pada data evaluasi. Nilai AUC model berada di kisaran 0.97–0.98, menandakan kemampuan diskriminasi yang sangat baik secara keseluruhan. Indikasi overfitting/bias terdeteksi karena ketidakseimbangan kelas yang ekstrem (Precision ≈ 82% dan Recall ≈ 17%), sehingga ke depannya dapat diterapkan teknik penyeimbangan kelas seperti SMOTE atau class weighting. |
| Opsi deployment | Model di-deploy sebagai Flask Web API (Python) yang menerima input transaksi dalam format JSON melalui REST API endpoint /predict dan /health. Aplikasi dibungkus menggunakan Docker Container dan di-deploy ke platform cloud Railway (Project: awake-flexibility). Model yang di-serve adalah SavedModel TensorFlow hasil ekspor komponen TFX Pusher dengan signature serving_default. |
| Tautan web app | https://mlops-creditcardfraud-detection-production.up.railway.app/ |
| Hasil monitoring | Monitoring infrastruktur MLOps dilakukan dengan mengintegrasikan Prometheus dan Grafana (Dashboard: CC-FRAUD-PREDICT-API-DASHBOARD) sebagai service terpisah di Railway. Hasil pemantauan real-time menunjukkan:


• Total Request: Tercatat 5 request ke endpoint POST /predict dan 1 request GET /.


• Total Prediksi Fraud: fraud_predict_Total mencatat 5 kali pengujian prediksi secara konsisten.


• Tingkat Kegagalan (failures_total): 0 (sistem berjalan stabil tanpa error).


• Performa Server: Rata-rata latensi eksekusi sangat rendah (~0.005–0.006 detik atau 5-6ms) dengan penggunaan CPU server yang stabil. |

---

## Struktur Proyek

```
M_Fathurrohman-pipeline 1/
├── M_Fathurrohman-pipeline/    # TFX Pipeline artifacts (ExampleGen, StatisticsGen, SchemaGen, ...)
├── modules/
│   ├── cc_fraud_transform.py   # TFX Transform module (preprocessing_fn)
│   ├── cc_fraud_trainer.py     # TFX Trainer module (run_fn, model_builder)
│   └── cc_fraud_tuner.py       # TFX Tuner module (hyperparameter tuning)
├── cc_data/
│   └── creditcard.csv          # Dataset Credit Card Fraud Detection
├── serving_model_dir/
│   └── cc-fraud-model/         # Exported SavedModel dari TFX Pusher
├── app/
│   ├── main.py                 # Flask serving app dengan Prometheus metrics
│   └── templates/
│       └── index.html          # Simple web UI
├── monitoring/
│   ├── prometheus.yml          # Konfigurasi Prometheus
│   └── docker-compose.yml      # Docker Compose untuk Prometheus + Grafana
├── credit_card_fraud_pipeline.ipynb  # Notebook utama TFX Pipeline
├── Dockerfile                  # Docker image untuk Flask app
├── Procfile                    # Heroku entry point
└── requirements.txt            # Dependencies
```

---

## Cara Menjalankan Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan Jupyter Notebook
jupyter notebook credit_card_fraud_pipeline.ipynb
```



