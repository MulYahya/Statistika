# =====================================================
# PREDIKSI HARGA LAPTOP MENGGUNAKAN SVR DAN KNN
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("Laptop_price.csv")

print("="*50)
print("5 DATA PERTAMA")
print(df.head())

print("\nINFO DATASET")
print(df.info())

print("\nSTATISTIK DESKRIPTIF")
print(df.describe())

# =====================================================
# 2. EDA
# =====================================================

print("\nJumlah Missing Value:")
print(df.isnull().sum())

# Distribusi Harga
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribusi Harga Laptop")
plt.show()

# =====================================================
# 3. ENCODING BRAND
# =====================================================

le = LabelEncoder()

df["Brand"] = le.fit_transform(df["Brand"])

# =====================================================
# 4. KORELASI FITUR
# =====================================================

plt.figure(figsize=(10,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Korelasi Antar Fitur")
plt.show()

# =====================================================
# 5. FEATURE DAN TARGET
# =====================================================

X = df.drop("Price", axis=1)
y = df["Price"]

# =====================================================
# 6. SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah Data Train :", len(X_train))
print("Jumlah Data Test  :", len(X_test))

# =====================================================
# 7. STANDARDISASI
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 8. HYPERPARAMETER TUNING SVR
# =====================================================

print("\nMelakukan Tuning SVR...")

svr_param = {
    'kernel': ['rbf'],
    'C': [1, 10, 100],
    'gamma': ['scale', 0.1, 0.01]
}

grid_svr = GridSearchCV(
    SVR(),
    svr_param,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_svr.fit(X_train_scaled, y_train)

best_svr = grid_svr.best_estimator_

print("\nBest Parameter SVR")
print(grid_svr.best_params_)

# =====================================================
# 9. HYPERPARAMETER TUNING KNN
# =====================================================

print("\nMelakukan Tuning KNN...")

knn_param = {
    'n_neighbors': [3,5,7,9,11],
    'weights': ['uniform','distance']
}

grid_knn = GridSearchCV(
    KNeighborsRegressor(),
    knn_param,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_knn.fit(X_train_scaled, y_train)

best_knn = grid_knn.best_estimator_

print("\nBest Parameter KNN")
print(grid_knn.best_params_)

# =====================================================
# 10. TRAINING MODEL
# =====================================================

best_svr.fit(X_train_scaled, y_train)
best_knn.fit(X_train_scaled, y_train)

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# =====================================================
# 11. PREDIKSI
# =====================================================

pred_svr = best_svr.predict(X_test_scaled)
pred_knn = best_knn.predict(X_test_scaled)
pred_lr  = linear_model.predict(X_test_scaled)

# =====================================================
# 12. EVALUASI
# =====================================================

def evaluate(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {model_name} =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2

svr_mae, svr_rmse, svr_r2 = evaluate(
    y_test,
    pred_svr,
    "SVR"
)

knn_mae, knn_rmse, knn_r2 = evaluate(
    y_test,
    pred_knn,
    "KNN"
)

lr_mae, lr_rmse, lr_r2 = evaluate(
    y_test,
    pred_lr,
    "Linear Regression"
)

# =====================================================
# 13. TABEL PERBANDINGAN
# =====================================================

comparison = pd.DataFrame({
    'Model':['SVR','KNN','Linear Regression'],
    'MAE':[svr_mae,knn_mae,lr_mae],
    'RMSE':[svr_rmse,knn_rmse,lr_rmse],
    'R2':[svr_r2,knn_r2,lr_r2]
})

print("\nPERBANDINGAN MODEL")
print(comparison)

# =====================================================
# 14. VISUALISASI PREDIKSI SVR
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(y_test, pred_svr)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("SVR: Actual vs Predicted")

plt.show()

# =====================================================
# 15. VISUALISASI PREDIKSI KNN
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(y_test, pred_knn)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("KNN: Actual vs Predicted")

plt.show()

# =====================================================
# 15B. VISUALISASI PREDIKSI LINEAR REGRESSION
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(y_test, pred_lr)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")

plt.show()

# =====================================================
# 16. GRAFIK PERBANDINGAN MODEL
# =====================================================

plt.figure(figsize=(8,5))

bars = plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.title("Perbandingan Nilai R²")
plt.ylabel("R² Score")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha='center'
    )

plt.show()

# =====================================================
# 17. FEATURE IMPORTANCE
# =====================================================

print("\nMenghitung Feature Importance...")

perm = permutation_importance(
    best_svr,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42
)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm.importances_mean
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance (Permutation Importance)")
plt.show()

# =====================================================
# KOEFISIEN LINEAR REGRESSION
# =====================================================

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_model.coef_
})

coef_df = coef_df.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nKoefisien Linear Regression")
print(coef_df)

plt.figure(figsize=(8,5))

sns.barplot(
    data=coef_df,
    x='Coefficient',
    y='Feature'
)

plt.title("Koefisien Linear Regression")
plt.show()

# =====================================================
# SELESAI
# =====================================================

print("\nProgram selesai dijalankan.")