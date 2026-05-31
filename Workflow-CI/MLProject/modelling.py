import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns_temp"))

# Load Data
df = pd.read_csv('Data Prepocessing.csv')
X = df.drop(columns=['Rent'])
y = df['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="CI_Automated_Run"):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Siapkan folder
    if os.path.exists("model_dir"):
        shutil.rmtree("model_dir")

    mlflow.sklearn.save_model(model, "model_dir")
    print("Training selesai dan model fisik disimpan di model_dir")