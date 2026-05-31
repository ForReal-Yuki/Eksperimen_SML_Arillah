import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data Prepocessing.csv")
X = df.drop(columns="Rent")
y = df["Rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="CI_Automated"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model Linier selesai dilatih")

    if os.path.exist("model_dir"):
        shutil.rmtree("model_dir")
    mlflow.sklearn.log_model(model, "model_dir")
    print("Task Selesai ada di model_dir")