import pandas as pd
import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

def prepocessing(input_path, output_path):
    df = pd.read_csv(input_path)
    categorical_feature = df.select_dtypes(include='object').columns.to_list()
    numerical_feature = df.select_dtypes(include='int64').columns

    Scaler = StandardScaler()
    df_scaled = Scaler.fit_transform(df[numerical_feature])
    df = pd.get_dummies(df, columns=categorical_feature, drop_first=True, dtype=int)
    df_cleaned = df.copy()
    previous_row_count = 0
    while len(df_cleaned) != previous_row_count:
        previous_row_count = len(df_cleaned)

        for col in numerical_feature:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)

if __name__ == "__main__":
    RAW = "House_Rent_Dataset.csv"
    PROCESS = "prepocessing/Data Prepocessing.csv"
    prepocessing(RAW, PROCESS)
