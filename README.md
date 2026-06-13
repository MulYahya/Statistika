# Prediksi Harga Laptop Menggunakan SVR dan KNN Regression

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi harga laptop berdasarkan spesifikasi perangkat keras menggunakan metode Machine Learning Regression.

Metode yang digunakan adalah:

1. Support Vector Regression (SVR)
2. K-Nearest Neighbor Regression (KNN)

Kedua metode dibandingkan untuk mengetahui model yang menghasilkan performa prediksi terbaik terhadap harga laptop.

---

# Dataset

Dataset yang digunakan adalah:

**Laptop_price.csv**

Dataset berisi informasi spesifikasi laptop dan harga.

## Atribut Dataset

| Fitur | Keterangan |
|---------|---------|
| Brand | Merek laptop |
| Processor_Speed | Kecepatan prosesor |
| RAM_Size | Kapasitas RAM |
| Storage_Capacity | Kapasitas penyimpanan |
| Screen_Size | Ukuran layar |
| Weight | Berat laptop |
| Price | Harga laptop (Target) |

---

# Tujuan Penelitian

1. Membangun model prediksi harga laptop menggunakan SVR.
2. Membangun model prediksi harga laptop menggunakan KNN Regression.
3. Membandingkan performa kedua model.
4. Mengetahui fitur yang paling berpengaruh terhadap harga laptop.

---

# Metode yang Digunakan

## 1. Support Vector Regression (SVR)

Support Vector Regression merupakan pengembangan dari Support Vector Machine (SVM) yang digunakan untuk menyelesaikan permasalahan regresi.

Keunggulan SVR:

- Mampu menangani hubungan non-linear.
- Cocok untuk dataset berukuran kecil hingga menengah.
- Memiliki generalisasi yang baik.

---

## 2. K-Nearest Neighbor Regression (KNN)

KNN Regression memprediksi nilai berdasarkan rata-rata dari sejumlah data tetangga terdekat.

Keunggulan KNN:

- Sederhana dan mudah dipahami.
- Tidak memerlukan proses pelatihan yang kompleks.
- Efektif pada data yang memiliki pola lokal.

---

# Alur Program

Program berjalan melalui beberapa tahapan berikut:

## 1. Load Dataset

Dataset dibaca menggunakan library Pandas.

```python
df = pd.read_csv("Laptop_price.csv")
```

---

## 2. Exploratory Data Analysis (EDA)

Dilakukan untuk memahami karakteristik data.

Analisis yang dilakukan:

- Menampilkan data awal
- Menampilkan informasi dataset
- Statistik deskriptif
- Pengecekan missing value
- Visualisasi distribusi harga

---

## 3. Encoding Data Kategorikal

Kolom Brand berbentuk teks sehingga perlu dikonversi menjadi angka menggunakan Label Encoding.

```python
LabelEncoder()
```

Contoh:

| Brand | Hasil Encoding |
|---------|---------|
| Asus | 0 |
| Dell | 1 |
| HP | 2 |

---

## 4. Analisis Korelasi

Menggunakan Heatmap untuk mengetahui hubungan antar fitur.

Tujuan:

- Melihat fitur yang paling berhubungan dengan harga.
- Mengidentifikasi kemungkinan multikolinearitas.

---

## 5. Pemisahan Data

Dataset dibagi menjadi:

- 80% Data Training
- 20% Data Testing

```python
train_test_split()
```

---

## 6. Standardisasi Data

Menggunakan StandardScaler.

Tujuan:

- Menyamakan skala seluruh fitur.
- Sangat penting untuk SVR dan KNN.

```python
StandardScaler()
```

---

## 7. Hyperparameter Tuning

### SVR

Parameter yang diuji:

- Kernel
- C
- Gamma

Menggunakan:

```python
GridSearchCV()
```

### KNN

Parameter yang diuji:

- n_neighbors
- weights

Menggunakan:

```python
GridSearchCV()
```

---

## 8. Training Model

Model terbaik hasil tuning digunakan untuk proses pelatihan.

### Model SVR

```python
SVR()
```

### Model KNN

```python
KNeighborsRegressor()
```

---

## 9. Prediksi

Model melakukan prediksi terhadap data testing.

```python
model.predict(X_test)
```

---

## 10. Evaluasi Model

Evaluasi dilakukan menggunakan:

### Mean Absolute Error (MAE)

Mengukur rata-rata selisih absolut antara nilai aktual dan prediksi.

Semakin kecil MAE semakin baik.

---

### Root Mean Squared Error (RMSE)

Mengukur besar error prediksi.

Semakin kecil RMSE semakin baik.

---

### R² Score

Mengukur kemampuan model menjelaskan variasi data.

Interpretasi:

| Nilai R² | Keterangan |
|-----------|-----------|
| 0.00 | Sangat buruk |
| 0.50 | Cukup |
| 0.70 | Baik |
| 0.90 | Sangat baik |
| 1.00 | Sempurna |

---

## 11. Visualisasi Hasil

Visualisasi yang dihasilkan:

### Scatter Plot

Menampilkan hubungan:

- Harga Aktual
- Harga Prediksi

Semakin dekat titik ke garis diagonal, semakin baik model.

---

### Grafik Perbandingan Model

Membandingkan:

- MAE
- RMSE
- R²

antara SVR dan KNN.

---

## 12. Feature Importance

Menggunakan:

```python
permutation_importance()
```

Tujuan:

- Menentukan fitur yang paling berpengaruh terhadap harga laptop.
- Memberikan interpretasi terhadap model.

---

# Struktur Folder

```

Project/
│
├── Laptop_price.csv
├── regresi_laptop.py
├── README.md
│
└── output/

```

---

# Library yang Digunakan

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Instalasi:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# Cara Menjalankan Program

Buka terminal pada folder project.

Jalankan:

```bash
python regresi_laptop.py
```

atau

```bash
py regresi_laptop.py
```

---

# Output Program

Program menghasilkan:

1. Informasi dataset
2. Statistik deskriptif
3. Heatmap korelasi
4. Distribusi harga
5. Hyperparameter terbaik SVR
6. Hyperparameter terbaik KNN
7. Nilai MAE
8. Nilai RMSE
9. Nilai R²
10. Grafik aktual vs prediksi
11. Grafik perbandingan model
12. Feature Importance

---

# Kesimpulan

Melalui proyek ini dilakukan implementasi dan perbandingan dua metode regresi yaitu Support Vector Regression (SVR) dan K-Nearest Neighbor Regression (KNN) untuk memprediksi harga laptop berdasarkan spesifikasinya.

Performa kedua model dievaluasi menggunakan MAE, RMSE, dan R² Score sehingga dapat diketahui model yang memberikan hasil prediksi terbaik pada dataset yang digunakan.
