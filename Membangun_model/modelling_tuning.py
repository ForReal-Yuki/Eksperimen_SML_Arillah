import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dagshub
import matplotlib.pyplot as plt

dagshub.init(repo_owner='ForReal-Yuki', repo_name='Eksperimen_SML_Arillah', mlflow=True)
# 1. Load Data
df = pd.read_csv('Data Prepocessing.csv')

X = df.drop(columns=['Rent'])
y = df['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup Hyperparameter Tuning (Syarat 1)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# 3. MANUAL LOGGING MLFLOW (Syarat 2)
with mlflow.start_run(run_name="Advance_DagsHub_Model"):
    print("Mulai proses training dan tuning...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Hitung metrik
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Logging parameter dan metrik ke DagsHub...")
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(best_model, "model")

    # --- ARTEFAK TAMBAHAN UNTUK ADVANCE (Minimal 2) ---
    print("Membuat dan mengunggah artefak tambahan...")

    # Artefak 1: Plot Feature Importance (Melihat fitur mana yang paling ngaruh ke harga sewa)
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")  # Kirim gambar ke DagsHub
    plt.close()

    # Artefak 2: Plot Actual vs Predicted (Melihat seberapa akurat tebakan model)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Harga Sewa Asli (Actual)")
    plt.ylabel("Tebakan Model (Predicted)")
    plt.title("Actual vs Predicted Rent")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")  # Kirim gambar ke DagsHub
    plt.close()

    # Artefak 3: Plot Residual (Melihat sebaran error/selisih prediksi)
    plt.figure(figsize=(8, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Tebakan Model (Predicted)")
    plt.ylabel("Error/Selisih (Residuals)")
    plt.title("Residual Plot (Sebaran Error)")
    plt.tight_layout()
    plt.savefig("residual_plot.png")
    mlflow.log_artifact("residual_plot.png")  # Kirim gambar ke DagsHub
    plt.close()

    # Artefak 4: CSV Sample Hasil Prediksi
    sample_results = pd.DataFrame({
        'Actual_Rent': y_test,
        'Predicted_Rent': y_pred,
        'Selisih_Harga': y_test - y_pred
    })
    # Simpan 20 baris pertama aja sebagai sample
    sample_results.head(20).to_csv("sample_predictions.csv", index=False)
    mlflow.log_artifact("sample_predictions.csv")  # Kirim file CSV ke DagsHub