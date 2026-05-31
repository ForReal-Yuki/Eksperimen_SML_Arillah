import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data Prepocessing.csv")
X = df.drop(columns="Rent")
y = df["Rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic Model Linier"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model Linier selesai dilatih")