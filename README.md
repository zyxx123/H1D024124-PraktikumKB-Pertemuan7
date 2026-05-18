# H1D024124 - Praktikum Kecerdasan Buatan
## Pertemuan 7 - Jaringan Syaraf Tiruan 2

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange) ![Keras](https://img.shields.io/badge/Keras-3.14.1-red)

---

## Deskripsi
Praktikum ini mengimplementasikan **Jaringan Syaraf Tiruan (JST)** menggunakan **TensorFlow** dan **Keras** untuk klasifikasi spesies bunga Iris berdasarkan fitur morfologinya.

---

## Dataset
- **Nama:** Iris Dataset
- **Sumber:** UCI Machine Learning Repository
- **Jumlah sampel:** 150
- **Fitur:** Sepal length, Sepal width, Petal length, Petal width
- **Kelas:** Iris-setosa, Iris-versicolor, Iris-virginica

---

## Arsitektur Model
```
Input Layer  →  4 fitur
Dense Layer  →  1000 neuron (ReLU)
Dense Layer  →  500 neuron  (ReLU)
Dense Layer  →  300 neuron  (ReLU)
Output Layer →  3 neuron    (Softmax)

Total Parameter: 656.703
```

---

## Konfigurasi Training
| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |
| Split Data | 80% train / 20% test |

---

## Hasil
| Metrik | Nilai |
|--------|-------|
| Akurasi Test | 100% |
| Loss Test | 0.0686 |

### Confusion Matrix
| | Setosa | Versicolor | Virginica |
|---|---|---|---|
| **Setosa** | 10 | 0 | 0 |
| **Versicolor** | 0 | 9 | 0 |
| **Virginica** | 0 | 0 | 11 |

---

## Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/zyxx123/H1D024124-PraktikumKB-Pertemuan7.git
cd H1D024124-PraktikumKB-Pertemuan7
```

### 2. Install library
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 3. Jalankan program
```bash
python iris_jst.py
```

---

## Library yang Digunakan
- TensorFlow 2.21.0
- Keras 3.14.1
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Struktur File
```
H1D024124-PraktikumKB-Pertemuan7/
├── iris_jst.py           # Source code utama
├── iris.data             # Dataset Iris
├── training_history.png  # Grafik akurasi & loss
├── confusion_matrix.png  # Confusion matrix
└── README.md             # Dokumentasi
```

---

**Nama:** Zaki Fatah Alfikri 
**NIM:** H1D024124  
**Mata Kuliah:** Praktikum Kecerdasan Buatan  
