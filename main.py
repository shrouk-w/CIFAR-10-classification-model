import os
from pathlib import Path

base_export_dir = str((Path(__file__).parent / 'files').resolve())
os.makedirs(base_export_dir, exist_ok=True)

print(f"Pliki będą zapisywane w:\n {base_export_dir}")

# =========================================================
# --- 0. Import bibliotek ---
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import datetime

from keras.datasets import cifar10
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Dropout
)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# --- 1. Importowanie danych CIFAR10 ---
# =========================================================
# Wczytanie zbioru danych CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Wyświetlenie przykładowego obrazu (opcjonalne)
print("Etykieta przykładowego obrazu:", y_train[0])
plt.imshow(x_train[0])
plt.axis('off')
plt.show()

# =========================================================
# --- 2. Wstępne przetwarzanie danych ---
# =========================================================

# Normalization 0-255 -> 0-1 = easier for model to process
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding -> class 3 -> [0,0,1,0,0,...] = to calculate loss func and represent solution as probability vector [0.01, 0.3, 0.7, 0.2, ...]
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# lista nazw klas CIFAR-10 (użyteczna w wyświetleniach)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# =========================================================
# --- 3. Definicja modelu sieci neuronowej ---
# =========================================================
input_shape = (32, 32, 3)  # CIFAR-10 to 32x32 RGB

model = Sequential([
    Input(shape=input_shape),

    Conv2D(32, (3, 3)), #creates feature maps 32 filters go through and catch features
    BatchNormalization(),  #scales values [-12,0,20] -> [-1, 0, 2] more stabilization
    Activation('swish'),  #swish is x * Q(x) where Q(x) = 1/(1+e^-x ) looks a bit like relu but catches some < 0 values
    MaxPooling2D((2, 2)), #goes through with small windows(2x2), gets max values, losses redundant information and smallens the picture

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('swish'),
    MaxPooling2D((2, 2)),

    Flatten(),  # gets all values into 1 long vector
    Dense(64, activation='swish'), #64 neurons learn to recognize shapes
    Dense(num_classes, activation='softmax') #10 outputs with probability of each class
])

# =========================================================
# --- 4. Kompilacja modelu ---
# =========================================================
# Optymalizator Adam, funkcja straty categorical_crossentropy, metryka accuracy


model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Wyświetlenie podsumowania modelu
model.summary()

# =========================================================
# --- 5. Trenowanie modelu ---
# =========================================================
es = EarlyStopping(
    monitor='val_loss',   # metryka, którą obserwujemy (np. val_loss, val_accuracy)
    patience=15,               # liczba epok bez poprawy, po których zatrzymujemy trening
    min_delta=1e-8,           # minimalna wymagana zmiana, by uznać, że jest „poprawa”
    mode='min',                # 'min' jeśli monitorujemy straty, 'max' jeśli dokładność
    restore_best_weights=True
)

rlp = ReduceLROnPlateau(
    monitor='val_loss',   # metryka do obserwacji
    factor=0.5,               # ile razy zmniejszyć LR (tu: o połowę)
    patience=6,               # liczba epok bez poprawy przed zmniejszeniem LR (zwykle < ES patience)
    min_delta=1e-8,           # próg czułości jak wyżej
    min_lr=1e-14,              # dolna granica learning rate
    mode='min'                # 'min' dla strat, 'max' dla dokładności
)

datagen = ImageDataGenerator(  #moves and rotates picture a bit so its less likely to overtrain
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    epochs=200,
    validation_data=(x_test, y_test_cat),
    callbacks=[es, rlp],
    verbose=1
)

# Data skończenia treningu - timestamp - znacznik do zapisywania plików
ts = datetime.datetime.now().strftime("_%Y%m%d_%H%M")

# =========================================================
# --- 6. Ewaluacja modelu ---
# =========================================================
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=1)
print('Dokładność na zbiorze testowym:', f"{accuracy:.2f}")
print('Strata na zbiorze testowym:', f"{loss:.2f}")

# =========================================================
# --- 7. Wizualizacja przebiegu treningu (loss) ---
# =========================================================
fig_loss_acc = plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss (train)')
plt.plot(history.history['val_loss'], label='loss (val)')
plt.xlabel('Epoka')
plt.ylabel('categorical_crossentropy')
plt.title(f'Strata na zbiorze testowym to {loss:.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy (train)')
plt.plot(history.history['val_accuracy'], label='accuracy (val)')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.title(f"Dokładność na zbiorze testowym to {accuracy:.2f}")
plt.legend()
plt.tight_layout()

# Zapis wykresu
loss_acc_path = os.path.join(base_export_dir, f'training_loss_accuracy{ts}.png')
fig_loss_acc.savefig(loss_acc_path)
print("Zapisano wykres loss/accuracy ->", loss_acc_path)

plt.show()

# =========================================================
# --- 8. Wizualizacja błędnych klasyfikacji i macierz pomyłek ---
# =========================================================
# Predykcje (etykiety)
pred_probs = model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test_cat, axis=1)  # lub po prostu y_test

# Indeksy błędnych klasyfikacji
incorrect_indices = np.nonzero(pred_labels != true_labels)[0]

# Wyświetlenie kilku błędnych przykładów
n_show = min(3, len(incorrect_indices))
for i in range(n_show):
    idx = incorrect_indices[i]
    plt.figure(figsize=(3, 3))
    plt.imshow(x_test[idx])  # obrazy RGB, bez cmap='gray'
    plt.title(f"Prawidłowo: {class_names[true_labels[idx]]}  -> Predykcja: {class_names[pred_labels[idx]]}")
    plt.axis('off')
    plt.show()

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Macierz pomyłek')

# Zapis wykresu
cm_path = os.path.join(base_export_dir, f'confusion_matrix{ts}.png')
fig_cm.savefig(cm_path)
print("Zapisano macierz pomyłek ->", cm_path)
plt.show()

# =========================================================
# --- 9. Zapis modelu ---
# =========================================================
model_path = os.path.join(base_export_dir, f"my_model{ts}.keras")
model.save(model_path)
print("Model zapisany jako", model_path)
