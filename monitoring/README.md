# Monitoring — Prometheus & Grafana

Direktori ini berisi konfigurasi untuk memantau sistem machine learning Credit Card Fraud Detection menggunakan **Prometheus** dan **Grafana**.

---

## Opsi 1: Menjalankan Prometheus di Windows (NATIVE TANPA DOCKER)

Ini adalah cara paling praktis jika Anda tidak menginstal Docker Desktop:

1. **Download Prometheus Binary:**
   - Buka: https://prometheus.io/download/
   - Cari bagian **Prometheus** dan download file bertipe **Windows** (`prometheus-...windows-amd64.zip`).
2. **Ekstrak file zip** tersebut ke suatu folder.
3. **Copy file `prometheus.yml`** dari direktori ini ke dalam folder hasil ekstraksi Prometheus.
4. **Jalankan Flask App terlebih dahulu:**
   ```bash
   python app/main.py
   ```
5. **Jalankan Prometheus:**
   Buka terminal di folder Prometheus dan jalankan:
   ```cmd
   .\prometheus.exe --config.file=prometheus.yml
   ```
6. **Akses Web UI:**
   - Prometheus Dashboard: http://localhost:9090
   - Target Status: http://localhost:9090/targets (status harus `UP`)

---

## Opsi 2: Menjalankan Menggunakan Docker Compose (Jika Ada Docker)

```bash
cd monitoring/
docker-compose up -d
```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin)

---

## Metrics yang Dipantau

| Metric | Type | Deskripsi |
|---|---|---|
| `http_requests_total` | Counter | Total HTTP request berdasarkan method, endpoint, status |
| `prediction_latency_seconds` | Histogram | Latensi prediksi model dalam detik |
| `fraud_predictions_total` | Counter | Total prediksi fraud vs normal |

## Query Contoh di Prometheus (PromQL)

```promql
# Request rate per detik
rate(http_requests_total[1m])

# 95th percentile prediction latency
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# Total prediksi fraud terdeteksi
fraud_predictions_total{result="fraud"}
```
