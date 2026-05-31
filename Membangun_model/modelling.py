import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("Data Prepocessing.csv")
X = df.drop(columns="Rent")
y = df["Rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Lokal")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic Model Random Forest"):
    # Pakai RandomForestRegressor buat regresi (tebak harga)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    print("Model Random Forest selesai dilatih dengan Autolog!")