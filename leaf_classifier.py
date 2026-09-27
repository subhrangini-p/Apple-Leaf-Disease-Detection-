# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
def load_and_preprocess_data(dataset_path, img_size=224):
    X, y, file_paths = [], [], []
    categories = ['healthy', 'disease']
    
    print("Loading image dataset...")
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            continue
        class_num = categories.index(category)
        for img_name in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            try:
                img_array = cv2.imread(img_path)
                if img_array is None:
                    continue
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (img_size, img_size))
                X.append(img_resized)
                y.append(class_num)
                file_paths.append(img_name)
            except Exception as e:
                pass

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y, dtype='int32')
    file_paths = np.array(file_paths)
    return X, y, file_paths

X, y, file_paths = load_and_preprocess_data(".", img_size=224)

# ==========================================
# 2. BALANCED TRAIN/TEST SPLIT
# ==========================================
healthy_idx = np.where(y == 0)[0]
disease_idx = np.where(y == 1)[0]

test_count = max(2, int(len(y) * 0.2 // 2))

X_test = np.concatenate([X[healthy_idx[-test_count:]], X[disease_idx[-test_count:]]], axis=0)
y_test = np.concatenate([y[healthy_idx[-test_count:]], y[disease_idx[-test_count:]]], axis=0)
paths_test = np.concatenate([file_paths[healthy_idx[-test_count:]], file_paths[disease_idx[-test_count:]]], axis=0)

X_train = np.concatenate([X[healthy_idx[:-test_count]], X[disease_idx[:-test_count]]], axis=0)
y_train = np.concatenate([y[healthy_idx[:-test_count]], y[disease_idx[:-test_count]]], axis=0)

# ==========================================
# 3. TRANSFER LEARNING MODEL
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining Transfer Learning Model...")
history = model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test))

# ==========================================
# 4. TRAINING PROGRESS GRAPH
# ==========================================
print("\nGenerating Accuracy & Loss Plots..."),
epochs_range = range(1, 16)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', color='forestgreen', linewidth=2)
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', color='darkorange', linewidth=2)
plt.title('Accuracy Curve', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss', color='crimson', linewidth=2)
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', color='mediumpurple', linewidth=2)
plt.title('Loss Curve', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ==========================================
# 5. CONFUSION MATRIX & EVALUATION REPORT
# ==========================================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- HIGH ACCURACY METRICS REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease'], zero_division=0))

print("\nDrawing Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'],
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Confusion Matrix', fontweight='bold')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()

# ==========================================
# 6. TERMINAL PREDICTION OUTPUT
# ==========================================
categories = ['Healthy', 'Disease']
random_idx = np.random.randint(0, len(X_test))

sample_name = paths_test[random_idx]
actual_label = categories[y_test[random_idx]]
predicted_label = categories[y_pred[random_idx]]
confidence = y_pred_probs[random_idx][y_pred[random_idx]] * 100
is_correct = "PASS" if actual_label == predicted_label else "FAIL"

print("\n==============================================")
print("       RANDOM LEAF PREDICTION RESULT          ")
print("==============================================")
print(f" Image Filename : {sample_name}")
print(f" Actual Class   : {actual_label}")
print(f" Predicted Class: {predicted_label}")
print(f" Confidence     : {confidence:.2f}%")
print(f" Match Result   : [{is_correct}]")
print("==============================================\n")
# Add this at the bottom of leaf_classifier.py
model.save("plant_model.keras")
print("✅ Model successfully saved as plant_model.keras!")
# =====================================================================
