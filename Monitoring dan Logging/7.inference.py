import pandas as pd
import requests
import time
import random

# Ganti ini kalau nama file CSV lo beda, dan pastikan path-nya bener (misal ambil dari folder depan)
csv_path = 'Data Prepocessing.csv'

# Load data, drop target column buat input
df = pd.read_csv(csv_path)
df_features = df.drop(columns=['Rent'])

EXPORTER_URL = "http://localhost:8000/predict"

print("Memulai proses inference massal untuk generate metrics...")
for i in range(100):  # Kirim 100 data simulasi
    # Ambil 1 baris random
    sample = df_features.sample(1)

    # Format payload sesuai standar MLflow pyfunc 'dataframe_split'
    payload = {
        "dataframe_split": {
            "columns": list(sample.columns),
            "data": sample.values.tolist()
        }
    }

    try:
        response = requests.post(EXPORTER_URL, json=payload)
        print(f"Request {i + 1}: Status {response.status_code}")
    except Exception as e:
        print(f"Request {i + 1} Gagal: {e}")

    # Jeda 2-5 detik biar grafik Prometheus-nya nanti kelihatan natural bergelombang
    time.sleep(random.uniform(2, 5))

print("Inference selesai!")