"""
PRAKTIKUM 7 - JARINGAN SYARAF TIRUAN 2
Klasifikasi Spesies Bunga Iris menggunakan TensorFlow & Keras
Dataset: iris.data (UCI Machine Learning Repository)
"""

# ============================================================
# LANGKAH 1: Import Library
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow versi:", tf.__version__)

# ============================================================
# LANGKAH 2: Muat Dataset
# ============================================================
# Muat dataset iris dari file lokal
dataset = pd.read_csv('iris.data', header=None, sep=',')

# Tampilkan info dataset
print("\n=== INFO DATASET ===")
print("Jumlah baris dan kolom:", dataset.shape)
print("\nLima baris pertama:")
print(dataset.head())

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values   # 4 kolom pertama sebagai fitur
y = dataset.iloc[:, -1].values    # Kolom terakhir sebagai label

print("\nFitur (X) shape:", X.shape)
print("Label (y) shape:", y.shape)
print("Kelas unik:", np.unique(y))

# ============================================================
# LANGKAH 3: Konversi Label ke Numerik
# ============================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Mengubah label jadi 0, 1, 2

print("\n=== ENCODING LABEL ===")
print("Kelas  :", label_encoder.classes_)
print("Mapping: 0 =", label_encoder.classes_[0],
      "| 1 =", label_encoder.classes_[1],
      "| 2 =", label_encoder.classes_[2])

# ============================================================
# LANGKAH 4: Split Data Training dan Testing
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== SPLIT DATA ===")
print("Data Training :", X_train.shape[0], "sampel")
print("Data Testing  :", X_test.shape[0], "sampel")

# ============================================================
# LANGKAH 5: Buat Arsitektur Model Neural Network
# ============================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),      # Input: 4 fitur
    Dense(1000, activation='relu'),         # Hidden Layer 1
    Dense(500,  activation='relu'),         # Hidden Layer 2
    Dense(300,  activation='relu'),         # Hidden Layer 3
    Dense(3,    activation='softmax')       # Output Layer: 3 kelas
])

# Tampilkan ringkasan arsitektur
print("\n=== ARSITEKTUR MODEL ===")
model.summary()

# ============================================================
# LANGKAH 6: Kompilasi Model
# ============================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("\nModel berhasil dikompilasi.")

# ============================================================
# LANGKAH 7: Latih Model
# ============================================================
print("\n=== PELATIHAN MODEL ===")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ============================================================
# LANGKAH 8: Evaluasi Model
# ============================================================
print("\n=== EVALUASI MODEL ===")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================
# LANGKAH 9: Visualisasi Training History
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History - Iris JST', fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2196F3')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#FF5722')
axes[0].set_title('Akurasi per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', color='#2196F3')
axes[1].plot(history.history['val_loss'], label='Val Loss', color='#FF5722')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Grafik training history disimpan: training_history.png")

# LANGKAH 10: Prediksi Data Test
print("\n=== PREDIKSI DATA TEST ===")
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

print("Prediksi  :", predicted_classes)
print("Label Asli:", y_test)

# Perbandingan prediksi vs asli
df_hasil = pd.DataFrame({
    'Label Asli'  : label_encoder.inverse_transform(y_test),
    'Prediksi'    : label_encoder.inverse_transform(predicted_classes),
    'Benar/Salah' : ['✓' if p == a else '✗' for p, a in zip(predicted_classes, y_test)]
})
print("\nDetail Prediksi:")
print(df_hasil.to_string(index=False))

# LANGKAH 11: Confusion Matrix
cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix - Iris JST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix disimpan: confusion_matrix.png")

# LANGKAH 12: Prediksi Data Baru (Input Manual)
def predict_new_data():
    print("\n=== PREDIKSI SPESIES BUNGA BARU ===")
    sepal_length = float(input("Masukkan sepal length (cm): "))
    sepal_width  = float(input("Masukkan sepal width  (cm): "))
    petal_length = float(input("Masukkan petal length (cm): "))
    petal_width  = float(input("Masukkan petal width  (cm): "))

    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data, verbose=0)
    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    print(f"\nHasil Prediksi: {predicted_label[0]}")
    print(f"Probabilitas  : Setosa={prediction[0][0]:.4f} | "
          f"Versicolor={prediction[0][1]:.4f} | Virginica={prediction[0][2]:.4f}")

# Contoh prediksi otomatis (tanpa input manual)
print("\n=== CONTOH PREDIKSI OTOMATIS ===")
contoh_data = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Iris-setosa
    [6.0, 2.9, 4.5, 1.5],  # Iris-versicolor
    [6.5, 3.0, 5.8, 2.2],  # Iris-virginica
])
for i, sample in enumerate(contoh_data):
    pred = model.predict(sample.reshape(1, -1), verbose=0)
    kelas = label_encoder.inverse_transform(pred.argmax(axis=1))[0]
    print(f"Input {sample} → Prediksi: {kelas}")

# Uncomment baris berikut untuk prediksi interaktif:
# predict_new_data()