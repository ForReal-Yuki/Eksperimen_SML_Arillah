from flask import Flask, Response, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import requests
import time
import psutil
import os

app = Flask(__name__)

# --- DEFINISI 10 METRIK UNTUK LEVEL ADVANCE ---
# 1. Total request masuk
REQ_TOTAL = Counter('model_requests_total', 'Total request ke model')
# 2. Total request sukses
REQ_SUCCESS = Counter('model_requests_success', 'Total request berhasil')
# 3. Total request gagal
REQ_FAILED = Counter('model_requests_failed', 'Total request gagal')
# 4. Latensi/Waktu proses prediksi
LATENCY = Histogram('model_inference_duration_seconds', 'Waktu proses inferensi')

# 5. Nilai prediksi harga sewa (Output model)
PREDICTION_VALUE = Gauge('prediction_value_rent', 'Hasil prediksi rent')
# 6. Fitur Input: BHK
FEAT_BHK = Gauge('feature_bhk', 'Input fitur BHK')
# 7. Fitur Input: Size
FEAT_SIZE = Gauge('feature_size', 'Input fitur Size')
# 8. Fitur Input: Bathroom
FEAT_BATHROOM = Gauge('feature_bathroom', 'Input fitur Bathroom')

# 9. Penggunaan CPU sistem
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU oleh sistem')
# 10. Penggunaan Memory/RAM sistem
RAM_USAGE = Gauge('system_memory_usage_bytes', 'Penggunaan RAM oleh sistem')

MLFLOW_URL = "http://127.0.0.1:5001/invocations"


@app.route('/predict', methods=['POST'])
def predict():
    REQ_TOTAL.inc()
    start_time = time.time()

    # Update metrik hardware (Metrik 9 & 10)
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().used)

    data = request.json
    try:
        # Ekstrak fitur dari JSON pandas split (asumsi input dari inference)
        input_data = data['dataframe_split']['data'][0]
        # Urutan kolom berdasarkan House Rent Dataset: BHK, Rent(target), Size, dll.
        # Kita ambil indeks kasarnya atau gunakan dictionary. Di inference kita kirim dict saja biar mudah.

        # Nerusin data ke Docker MLflow
        response = requests.post(MLFLOW_URL, json=data)

        if response.status_code == 200:
            REQ_SUCCESS.inc()
            pred_val = response.json()['predictions'][0]
            PREDICTION_VALUE.set(pred_val)

            # Catat fitur input (Metrik 6, 7, 8)
            # Karena MLflow butuh format spesifik, di sini kita tangkap langsung dari kolom
            columns = data['dataframe_split']['columns']
            row = data['dataframe_split']['data'][0]
            if 'BHK' in columns: FEAT_BHK.set(row[columns.index('BHK')])
            if 'Size' in columns: FEAT_SIZE.set(row[columns.index('Size')])
            if 'Bathroom' in columns: FEAT_BATHROOM.set(row[columns.index('Bathroom')])

        else:
            REQ_FAILED.inc()

        LATENCY.observe(time.time() - start_time)
        return response.json()

    except Exception as e:
        REQ_FAILED.inc()
        return {"error": str(e)}, 500


@app.route('/metrics')
def metrics():
    # Rute khusus yang akan ditarik datanya oleh Prometheus setiap 5 detik
    return Response(generate_latest(), mimetype="text/plain")


if __name__ == '__main__':
    # Jalan di port 8000, sesuai dengan file 2.prometheus.yml kita
    app.run(host='0.0.0.0', port=8000)